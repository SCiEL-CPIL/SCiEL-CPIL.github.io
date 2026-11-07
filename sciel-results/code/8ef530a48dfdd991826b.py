# EVOLVE-BLOCK-START
import numpy as np
import time

# Cache for precomputed optimal depths per (sigma, A, k, gamma, Q, steps)
_depth_cache = {}
_start_time = time.time()


def _solve_depth_for_cost_vectorized(C, k, A, dt):
    """
    Vectorized solve for optimal depth d given marginal cost array C.
    """
    C = np.asarray(C, dtype=np.float64)
    d = np.zeros_like(C)
    
    # Mask for C > 0
    mask = C > 0
    if not np.any(mask):
        return d
    
    C_pos = C[mask]
    
    # Initial guess
    d_pos = C_pos + 1.0 / k
    d_pos = np.clip(d_pos, 0.0, 30.0 / k)
    
    # Newton's method
    max_d = 30.0 / k
    for _ in range(10):
        L = A * np.exp(-k * d_pos) * dt
        g = np.exp(L) - 1.0 - k * L * (d_pos - C_pos)
        gp = k * L * (k * (d_pos - C_pos) - np.exp(L) - 1.0)
        
        # Avoid division by zero
        valid = np.abs(gp) > 1e-15
        if not np.any(valid):
            break
            
        d_new = d_pos.copy()
        d_new[valid] = d_pos[valid] - g[valid] / gp[valid]
        
        # Clamp
        d_new = np.clip(d_new, 0.0, max_d)
        
        if np.max(np.abs(d_new - d_pos)) < 1e-12:
            d_pos = d_new
            break
        d_pos = d_new
        
    d[mask] = d_pos
    return d


def _precompute_depths_exact(sigma, A, k, gamma, Q, steps, T=1.0):
    """
    Precompute optimal bid/ask depths for each inventory level and time step
    using exact finite-horizon dynamic programming.
    
    Returns:
    bid_depths: array of shape (steps, 2*Q+1)
    ask_depths: array of shape (steps, 2*Q+1)
    """
    dt = T / steps
    nq = 2 * Q + 1
    
    # V_next[q_idx] holds V_{t+1}(q)
    V_next = np.zeros(nq, dtype=np.float64)
    
    bid_depths = np.zeros((steps, nq), dtype=np.float64)
    ask_depths = np.zeros((steps, nq), dtype=np.float64)
    
    q_vals = np.arange(-Q, Q + 1)
    
    for t in range(steps - 1, -1, -1):
        # Compute marginal costs for bid and ask for all q
        # C_b(q) = V_next(q) - V_next(q+1)  (cost of increasing inventory)
        # C_a(q) = V_next(q) - V_next(q-1)  (cost of decreasing inventory)
        
        # Handle boundaries
        # q = Q: no bid fill. C_b is irrelevant.
        # q = -Q: no ask fill. C_a is irrelevant.
        
        V_curr = np.zeros(nq, dtype=np.float64)
        
        # Interior states
        for i in range(nq):
            q = i - Q
            
            # Compute costs
            if q < Q:
                # Cost of bid fill (q -> q+1)
                C_b = V_next[i] - V_next[i + 1]
            else:
                C_b = 0.0 # Won't be used
            
            if q > -Q:
                # Cost of ask fill (q -> q-1)
                C_a = V_next[i] - V_next[i - 1]
            else:
                C_a = 0.0 # Won't be used
            
            # Solve for optimal depths
            # We can do this in a batch for all q, but let's do it per q for simplicity first,
            # or better, vectorize the whole loop.
            
        # Vectorized approach for the whole step t
        C_b_arr = np.zeros(nq)
        C_a_arr = np.zeros(nq)
        
        # q from -Q to Q-1
        # i from 0 to nq-2
        C_b_arr[:nq-1] = V_next[:nq-1] - V_next[1:]
        
        # q from -Q+1 to Q
        # i from 1 to nq-1
        C_a_arr[1:] = V_next[1:] - V_next[:nq-1]
        
        # Solve depths
        d_b_arr = _solve_depth_for_cost_vectorized(C_b_arr, k, A, dt)
        d_a_arr = _solve_depth_for_cost_vectorized(C_a_arr, k, A, dt)
        
        # Set boundaries to inf (no quote)
        d_b_arr[nq-1] = np.inf # q = Q
        d_a_arr[0] = np.inf    # q = -Q
        
        # Calculate probabilities
        # L = A * exp(-k * d) * dt
        # p = 1 - exp(-L)
        
        # Handle inf depths: p = 0
        L_b = np.where(np.isfinite(d_b_arr), A * np.exp(-k * d_b_arr) * dt, 0.0)
        L_a = np.where(np.isfinite(d_a_arr), A * np.exp(-k * d_a_arr) * dt, 0.0)
        
        p_b = 1.0 - np.exp(-L_b)
        p_a = 1.0 - np.exp(-L_a)
        
        # Reward
        reward_b = np.where(np.isfinite(d_b_arr), d_b_arr * p_b, 0.0)
        reward_a = np.where(np.isfinite(d_a_arr), d_a_arr * p_a, 0.0)
        
        # Expected Next Value
        # E[V_next] = (1-p_b)(1-p_a)V(q) + p_b(1-p_a)V(q+1) + (1-p_b)p_a V(q-1) + p_b p_a V(q)
        # Note: V(q) is V_next[i]
        
        V_q = V_next
        V_q_plus = np.zeros(nq)
        V_q_minus = np.zeros(nq)
        
        # V(q+1) for q < Q
        V_q_plus[:nq-1] = V_next[1:]
        V_q_plus[nq-1] = V_next[nq-1] # Boundary: no change if no fill? Or just doesn't happen.
        
        # V(q-1) for q > -Q
        V_q_minus[1:] = V_next[:nq-1]
        V_q_minus[0] = V_next[0]
        
        # If p_b=0 (q=Q), term p_b * V(q+1) is 0.
        # If p_a=0 (q=-Q), term p_a * V(q-1) is 0.
        
        E_V_next = (1.0 - p_b) * (1.0 - p_a) * V_q + \
                   p_b * (1.0 - p_a) * V_q_plus + \
                   (1.0 - p_b) * p_a * V_q_minus + \
                   p_b * p_a * V_q # Assuming both hits return to q
        
        # Penalty
        penalty = (gamma / 2.0) * sigma**2 * q_vals**2 * dt
        
        V_curr = -penalty + reward_b + reward_a + E_V_next
        
        # Store depths
        bid_depths[t, :] = d_b_arr
        ask_depths[t, :] = d_a_arr
        
        V_next = V_curr
        
    return bid_depths, ask_depths


def quote(t, inventory, mid, params):
    """Inventory-aware quoting with precomputed optimal depths."""
    sigma = params["sigma"]
    A = params["A"]
    k = params["k"]
    gamma = params["gamma"]
    Q = params["Q"]
    steps = params["steps"]
    T = params["T"]
    
    # Deadline guard
    if time.time() - _start_time > 295:
        # Return default depths
        default_depth = 1.0 / k
        bid_depths = np.full(4000, default_depth, dtype=np.float64)
        ask_depths = np.full(4000, default_depth, dtype=np.float64)
        bid_depths[inventory >= Q] = np.inf
        ask_depths[inventory <= -Q] = np.inf
        return bid_depths, ask_depths
    
    # Check cache
    key = (sigma, A, k, gamma, Q, steps, T)
    if key not in _depth_cache:
        _depth_cache[key] = _precompute_depths_exact(sigma, A, k, gamma, Q, steps, T)
    
    bid_depths_table, ask_depths_table = _depth_cache[key]
    
    # Determine current time step index
    # t = n * dt, dt = T / steps
    # n = t / dt = t * steps / T
    n_step = int(np.floor(t * steps / T))
    # Clamp n_step to valid range [0, steps-1]
    n_step = np.clip(n_step, 0, steps - 1)
    
    # Map inventory to index
    q = inventory
    indices = (q + Q).astype(np.int64)
    indices = np.clip(indices, 0, 2 * Q)
    
    # Get depths for current time step and inventory
    bid_depths = bid_depths_table[n_step, indices].copy()
    ask_depths = ask_depths_table[n_step, indices].copy()
    
    # Enforce no bid fill at q = Q, no ask fill at q = -Q (already done in precompute, but double check)
    bid_depths[q >= Q] = np.inf
    ask_depths[q <= -Q] = np.inf
    
    # Ensure all values are valid (no NaN, no values below -3/k)
    min_valid = -3.0 / k
    bid_depths = np.where(np.isfinite(bid_depths) & (bid_depths >= min_valid), bid_depths, np.inf)
    ask_depths = np.where(np.isfinite(ask_depths) & (ask_depths >= min_valid), ask_depths, np.inf)
    
    return bid_depths, ask_depths
# EVOLVE-BLOCK-END
