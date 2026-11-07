"""Market-making quote strategy using high-precision DP with grid search over intensities."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Cache for precomputed value functions per regime
_dp_cache = {}
_DEADLINE = 295.0  # seconds, safety margin before 330s kill
_start_time = None


def _get_lambda_grid(A, n_points=300):
    """
    Log-spaced grid of intensities lambda.
    lambda = A * exp(-k * d)
    d is depth.
    We want to cover d from a large positive value (low lambda) to a negative value (high lambda).
    Valid d range: [-3/k, inf).
    lambda_max corresponds to d = -3/k -> lambda = A * exp(3).
    lambda_min corresponds to d = large -> lambda ~ 0.
    """
    # d_min = -3/k, d_max = 10/k (arbitrary large)
    # lambda_min = A * exp(-k * d_max) = A * exp(-10)
    # lambda_max = A * exp(-k * d_min) = A * exp(3)
    lam_min = A * np.exp(-10.0)
    lam_max = A * np.exp(3.0)
    # Ensure lam_min is not too small to avoid log issues later, though 1e-10 is fine
    lam_min = max(lam_min, 1e-10)
    return np.geomspace(lam_min, lam_max, n_points)


def _precompute_dp(sigma, A, k, gamma, T, steps, Q):
    """
    Precompute the optimal policy via backward DP on (time, inventory).
    Returns arrays: bid_depth[t, q], ask_depth[t, q] for t in 0..steps-1, q in -Q..Q.
    """
    global _start_time
    if _start_time is None:
        _start_time = time.time()
    
    key = (sigma, A, k, gamma, T, steps, Q)
    if key in _dp_cache:
        return _dp_cache[key]
    
    dt = T / steps
    n_q = 2 * Q + 1  # number of inventory states
    q_indices = np.arange(-Q, Q + 1)  # -Q, -Q+1, ..., Q
    
    # Generate lambda grid
    n_lam = 300
    lam_grid = _get_lambda_grid(A, n_lam)
    
    # Precompute probabilities and depths for the grid
    # p = 1 - exp(-lambda * dt)
    p_grid = 1.0 - np.exp(-lam_grid * dt)
    # d = -1/k * ln(lambda / A)
    # Handle lambda/A carefully
    ratio = lam_grid / A
    d_grid = -np.log(ratio) / k
    
    # Clip depths to valid range [-3/k, inf)
    # The problem says values below -3/k are invalid.
    # We should ensure our grid doesn't go below -3/k.
    # lam_max was set to A*exp(3), so d_min = -3/k.
    # Due to floating point, clip just in case.
    d_grid = np.maximum(d_grid, -3.0 / k + 1e-9)
    
    # Inventory cost: gamma/2 * sigma^2 * q^2 * dt
    inv_cost = 0.5 * gamma * sigma**2 * q_indices**2 * dt
    
    # Value function: V[t, q]
    # V[steps, q] = 0
    V_next = np.zeros(n_q)
    
    # Store optimal depths
    opt_bid = np.zeros((steps, n_q))
    opt_ask = np.zeros((steps, n_q))
    
    # Precompute meshgrid indices for vectorization?
    # We will loop over q for the next state values, but vectorize over lambda grid.
    # Actually, we can vectorize over q as well if we handle boundaries carefully.
    # But boundaries (q=Q, q=-Q) restrict the action space.
    # It's safer to loop over q (61 iterations) and vectorize over lambda (300x300).
    
    for t in range(steps - 1, -1, -1):
        if time.time() - _start_time > _DEADLINE - 10:
            # Safety: return what we have so far
            # Fill remaining with default
            default_d = 1.0 / k
            opt_bid[:t+1] = default_d
            opt_ask[:t+1] = default_d
            result = (opt_bid, opt_ask)
            _dp_cache[key] = result
            return result
        
        V_curr = np.zeros(n_q)
        
        for i, q in enumerate(q_indices):
            # Determine valid actions
            # If q == Q, no bid fill allowed. Effectively p_b = 0.
            # If q == -Q, no ask fill allowed. Effectively p_a = 0.
            
            if q < Q and q > -Q:
                # Both bid and ask allowed
                # Vectorize over all pairs (lam_b, lam_a)
                # p_b shape (n_lam, 1), p_a shape (1, n_lam)
                pb = p_grid[:, np.newaxis]
                pa = p_grid[np.newaxis, :]
                
                # Immediate reward: d_b * p_b + d_a * p_a
                db = d_grid[:, np.newaxis]
                da = d_grid[np.newaxis, :]
                
                imm = db * pb + da * pa - inv_cost[i]
                
                # Next state values
                V_q = V_next[i]
                V_qp1 = V_next[i+1]
                V_qm1 = V_next[i-1]
                
                # Probabilities of next states
                # q stays: (1-pb)(1-pa)
                # q+1: pb(1-pa)
                # q-1: (1-pb)pa
                
                p_stay = (1 - pb) * (1 - pa)
                p_up = pb * (1 - pa)
                p_down = (1 - pb) * pa
                
                V_next_state = p_stay * V_q + p_up * V_qp1 + p_down * V_qm1
                
                total = imm + V_next_state
                
                idx = np.argmax(total)
                idx_r, idx_c = np.unravel_index(idx, total.shape)
                
                opt_bid[t, i] = d_grid[idx_r]
                opt_ask[t, i] = d_grid[idx_c]
                V_curr[i] = total[idx_r, idx_c]
                
            elif q < Q:
                # Only bid allowed (q == -Q is handled in else if? No, q < Q and q == -Q is possible if Q=0, but Q=30)
                # Here q == -Q is false because q < Q and q > -Q was false, so q <= -Q or q >= Q.
                # If q < Q, then q must be -Q.
                # So only bid is allowed? No, at q=-Q, no ask fill. So only bid.
                pb = p_grid
                db = d_grid
                imm = db * pb - inv_cost[i]
                
                V_q = V_next[i]
                V_qp1 = V_next[i+1]
                
                V_next_state = (1 - pb) * V_q + pb * V_qp1
                
                total = imm + V_next_state
                idx = np.argmax(total)
                
                opt_bid[t, i] = d_grid[idx]
                opt_ask[t, i] = np.inf # No ask
                V_curr[i] = total[idx]
                
            elif q > -Q:
                # Only ask allowed (q == Q)
                pa = p_grid
                da = d_grid
                imm = da * pa - inv_cost[i]
                
                V_q = V_next[i]
                V_qm1 = V_next[i-1]
                
                V_next_state = (1 - pa) * V_q + pa * V_qm1
                
                total = imm + V_next_state
                idx = np.argmax(total)
                
                opt_bid[t, i] = np.inf # No bid
                opt_ask[t, i] = d_grid[idx]
                V_curr[i] = total[idx]
                
            else:
                # q == Q and q == -Q (only if Q=0)
                V_curr[i] = -inv_cost[i]
                opt_bid[t, i] = np.inf
                opt_ask[t, i] = np.inf
        
        V_next = V_curr
    
    result = (opt_bid, opt_ask)
    _dp_cache[key] = result
    return result


def quote(t, inventory, mid, params):
    """
    Quote bid and ask depths based on precomputed optimal policy.
    """
    global _start_time
    if _start_time is None:
        _start_time = time.time()
    
    sigma = params['sigma']
    A = params['A']
    k = params['k']
    gamma = params['gamma']
    T = params['T']
    steps = params['steps']
    Q = params['Q']
    
    # Determine current time step index
    dt = T / steps
    t_idx = int(round(t / dt))
    t_idx = min(t_idx, steps - 1)
    t_idx = max(t_idx, 0)
    
    # Get precomputed optimal depths
    try:
        opt_bid, opt_ask = _precompute_dp(sigma, A, k, gamma, T, steps, Q)
    except Exception:
        # Fallback: use 1/k
        depth = np.full(len(inventory), 1.0 / k)
        return depth, depth.copy()
    
    # Map inventory to index: q in [-Q, Q] -> index q + Q
    inv_idx = np.clip(inventory, -Q, Q) + Q
    inv_idx = inv_idx.astype(int)
    
    # Look up optimal depths
    bid_depths = opt_bid[t_idx, inv_idx]
    ask_depths = opt_ask[t_idx, inv_idx]
    
    # Ensure valid values
    # Values below -3/k are invalid.
    # np.inf is allowed for no quote.
    # NaN is invalid.
    
    # Replace NaN with inf? Or 1/k?
    # If NaN, something went wrong. Use 1/k as safe default.
    bid_depths = np.where(np.isnan(bid_depths), 1.0/k, bid_depths)
    ask_depths = np.where(np.isnan(ask_depths), 1.0/k, ask_depths)
    
    # Clip to lower bound
    min_depth = -3.0 / k
    bid_depths = np.maximum(bid_depths, min_depth)
    ask_depths = np.maximum(ask_depths, min_depth)
    
    return bid_depths, ask_depths
# EVOLVE-BLOCK-END
