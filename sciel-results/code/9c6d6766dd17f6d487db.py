"""Market-making quote strategy using precomputed DP value function."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Cache for precomputed value functions per regime
_dp_cache = {}
_DEADLINE = 300.0  # seconds, safety margin before 330s kill
_start_time = None


def _get_depth_grid(k, n_points=200):
    """Log-spaced depth grid with fine resolution at low depths."""
    # k*d ranges from ~0.01 to ~10
    # d ranges from 1/k * 0.01 to 1/k * 10
    # Use log-spaced in k*d space
    kd_min = 0.01
    kd_max = 10.0
    kd = np.geomspace(kd_min, kd_max, n_points)
    return kd / k


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
    
    # Depth grid
    depth_grid = _get_depth_grid(k, n_points=150)
    n_d = len(depth_grid)
    
    # Precompute fill probabilities for each depth
    # p_bid(d) = 1 - exp(-A * exp(-k*d) * dt)
    # p_ask(d) = 1 - exp(-A * exp(-k*d) * dt)
    exp_kd = np.exp(-k * depth_grid)
    fill_prob = 1.0 - np.exp(-A * exp_kd * dt)
    
    # Inventory cost: gamma/2 * sigma^2 * q^2 * dt
    inv_cost = 0.5 * gamma * sigma**2 * q_indices**2 * dt
    
    # Value function: V[t, q] = optimal expected reward from time t to T given inventory q
    # V[steps, q] = 0 (no terminal charge)
    V_next = np.zeros(n_q)
    
    # Store optimal depths
    opt_bid = np.zeros((steps, n_q))
    opt_ask = np.zeros((steps, n_q))
    
    # Backward DP
    for t in range(steps - 1, -1, -1):
        if time.time() - _start_time > _DEADLINE - 30:
            # Safety: return what we have so far (use 1/k as default)
            default_d = 1.0 / k
            opt_bid[:t+1] = default_d
            opt_ask[:t+1] = default_d
            result = (opt_bid, opt_ask)
            _dp_cache[key] = result
            return result
        
        V_curr = np.zeros(n_q)
        
        for i, q in enumerate(q_indices):
            # No bid fill at q = Q, no ask fill at q = -Q
            # If we quote bid at depth d_b and ask at depth d_a:
            # Expected immediate reward = d_b * p_bid(d_b) + d_a * p_ask(d_a) - inv_cost[q]
            # Next state: q+1 with prob p_bid (if q < Q), q-1 with prob p_ask (if q > -Q)
            # V = max over (d_b, d_a) of [immediate + (1-p_b)(1-p_a)*V_next[q] + p_b*(1-p_a)*V_next[q+1] + (1-p_b)*p_a*V_next[q-1]]
            
            # Vectorize over depth choices
            # For bid: if q < Q, we can quote; else no bid
            # For ask: if q > -Q, we can quote; else no ask
            
            # Try all combinations of bid and ask depths
            # This is O(n_d^2) per state, which might be slow. Let's optimize.
            # Actually, the problem separates if we consider them independently? No, they interact through next state.
            
            # Let's do a grid search but vectorized
            # bid_depths: (n_d,) or (1,) if no bid
            # ask_depths: (n_d,) or (1,) if no ask
            
            if q < Q:
                bid_d = depth_grid
                p_b = fill_prob
                # Reward from bid: d_b * p_b
                bid_reward = bid_d * p_b
            else:
                bid_d = np.array([0.0])
                p_b = np.array([0.0])
                bid_reward = np.array([0.0])
            
            if q > -Q:
                ask_d = depth_grid
                p_a = fill_prob
                ask_reward = ask_d * p_a
            else:
                ask_d = np.array([0.0])
                p_a = np.array([0.0])
                ask_reward = np.array([0.0])
            
            # Create meshgrid for all (bid, ask) combinations
            # n_d_bid x n_d_ask
            if q < Q and q > -Q:
                bb, aa = np.meshgrid(bid_d, ask_d, indexing='ij')
                pbb = np.meshgrid(p_b, p_a, indexing='ij')
                pb = pbb[0]
                pa = pbb[1]
                
                # Immediate reward
                imm = bid_reward[:, None] + ask_reward[None, :] - inv_cost[i]
                
                # Next state value
                # q stays: (1-pb)(1-pa) * V_next[i]
                # q+1: pb*(1-pa) * V_next[i+1]  (if q+1 <= Q)
                # q-1: (1-pb)*pa * V_next[i-1]  (if q-1 >= -Q)
                
                V_q = V_next[i]
                V_qp1 = V_next[i+1] if q + 1 <= Q else 0.0
                V_qm1 = V_next[i-1] if q - 1 >= -Q else 0.0
                
                V_next_state = ((1 - pb) * (1 - pa) * V_q +
                               pb * (1 - pa) * V_qp1 +
                               (1 - pb) * pa * V_qm1)
                
                total = imm + V_next_state
                idx = np.argmax(total)
                idx = np.unravel_index(idx, total.shape)
                opt_bid[t, i] = bid_d[idx[0]]
                opt_ask[t, i] = ask_d[idx[1]]
                V_curr[i] = total[idx]
            elif q < Q:  # only bid
                p_b = fill_prob
                pb = p_b
                V_q = V_next[i]
                V_qp1 = V_next[i+1] if q + 1 <= Q else 0.0
                
                imm = bid_reward - inv_cost[i]
                V_next_state = (1 - pb) * V_q + pb * V_qp1
                total = imm + V_next_state
                idx = np.argmax(total)
                opt_bid[t, i] = bid_d[idx]
                opt_ask[t, i] = 0.0  # no ask
                V_curr[i] = total[idx]
            elif q > -Q:  # only ask
                p_a = fill_prob
                pa = p_a
                V_q = V_next[i]
                V_qm1 = V_next[i-1] if q - 1 >= -Q else 0.0
                
                imm = ask_reward - inv_cost[i]
                V_next_state = (1 - pa) * V_q + pa * V_qm1
                total = imm + V_next_state
                idx = np.argmax(total)
                opt_bid[t, i] = 0.0  # no bid
                opt_ask[t, i] = ask_d[idx]
                V_curr[i] = total[idx]
            else:
                # q == Q and q == -Q simultaneously? Only if Q=0
                V_curr[i] = -inv_cost[i]
                opt_bid[t, i] = 0.0
                opt_ask[t, i] = 0.0
        
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
    
    # Ensure valid values: no negative, no inf, no nan
    bid_depths = np.where(bid_depths > 0, bid_depths, np.inf)
    ask_depths = np.where(ask_depths > 0, ask_depths, np.inf)
    
    return bid_depths, ask_depths
# EVOLVE-BLOCK-END
