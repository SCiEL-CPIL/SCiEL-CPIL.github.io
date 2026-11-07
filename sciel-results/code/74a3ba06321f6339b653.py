"""Optimal market making with inventory risk using pre-computed HJB backward induction."""
import numpy as np
import time

_POLICY_CACHE = {}
_START_TIME = time.time()
_DEADLINE = _START_TIME + 290.0

def _compute_policy_table(sigma, A, k, gamma, T, steps, Q):
    """
    Compute the optimal bid/ask depths for each (step, inventory) pair
    using backward induction on the discrete-time HJB equation.
    """
    dt = T / steps
    n_inv = 2 * Q + 1
    inv_range = np.arange(-Q, Q + 1)
    
    V_next = np.zeros(n_inv)
    
    # Candidate depths - use a finer grid for better precision
    depth_min = 1e-6
    depth_max = 10.0 / k
    n_depths = 300
    depths = np.linspace(depth_min, depth_max, n_depths)
    
    # Precompute fill probabilities and "reward" terms p*d
    exp_kd = np.exp(-k * depths)
    p_fill = 1.0 - np.exp(-A * exp_kd * dt)
    p_d = p_fill * depths
    
    # Precompute cross term matrix for 2D optimization
    # cross[i,j] = p_fill[i] * p_fill[j]
    cross_pf = np.outer(p_fill, p_fill)
    
    bid_table = np.zeros((steps, n_inv))
    ask_table = np.zeros((steps, n_inv))
    
    # Pre-allocate V_cur
    V_cur = np.zeros(n_inv)
    
    for n in range(steps - 1, -1, -1):
        # Time check with safety margin
        if time.time() > _DEADLINE:
            # Fill remaining steps with current best guess (copy from last computed step)
            for n_rem in range(n, -1, -1):
                if n_rem < steps - 1:
                    bid_table[n_rem] = bid_table[n_rem + 1]
                    ask_table[n_rem] = ask_table[n_rem + 1]
                else:
                    bid_table[n_rem] = 1.0 / k
                    ask_table[n_rem] = 1.0 / k
            break
            
        for i in range(n_inv):
            q = inv_range[i]
            inv_cost = -0.5 * gamma * sigma**2 * q**2 * dt
            
            can_bid = q < Q
            can_ask = q > -Q
            
            if can_bid and can_ask:
                c1 = V_next[i+1] - V_next[i]
                c2 = V_next[i-1] - V_next[i]
                c3 = -c1 - c2
                
                # Vectorized 2D optimization
                term_b = p_d + c1 * p_fill  # shape (D,)
                term_a = p_d + c2 * p_fill  # shape (D,)
                cross = c3 * cross_pf  # shape (D, D)
                
                total_val = term_b[:, None] + term_a[None, :] + cross
                
                # Find max
                max_val = np.max(total_val)
                max_idx = np.argmax(total_val)
                j_best = max_idx // n_depths
                l_best = max_idx % n_depths
                
                bid_table[n, i] = depths[j_best]
                ask_table[n, i] = depths[l_best]
                V_cur[i] = inv_cost + V_next[i] + max_val
                
            elif can_bid and not can_ask:
                c1 = V_next[i+1] - V_next[i]
                vals = p_fill * (depths + c1)
                max_val = np.max(vals)
                j_best = np.argmax(vals)
                bid_table[n, i] = depths[j_best]
                ask_table[n, i] = np.inf
                V_cur[i] = inv_cost + V_next[i] + max_val
                
            elif can_ask and not can_bid:
                c2 = V_next[i-1] - V_next[i]
                vals = p_fill * (depths + c2)
                max_val = np.max(vals)
                l_best = np.argmax(vals)
                bid_table[n, i] = np.inf
                ask_table[n, i] = depths[l_best]
                V_cur[i] = inv_cost + V_next[i] + max_val
                
            else:
                bid_table[n, i] = np.inf
                ask_table[n, i] = np.inf
                V_cur[i] = inv_cost + V_next[i]
        
        V_next = V_cur.copy()
    
    return bid_table, ask_table


def _get_policy_table(params):
    """Get or compute the policy table for the given params."""
    key = (params['sigma'], params['A'], params['k'], params['gamma'], params['T'], params['steps'], params['Q'])
    if key not in _POLICY_CACHE:
        _POLICY_CACHE[key] = _compute_policy_table(
            params['sigma'], params['A'], params['k'], params['gamma'],
            params['T'], params['steps'], params['Q']
        )
    return _POLICY_CACHE[key]

# Pre-compute tables for known regimes to avoid delay in first quote call
# We don't know the exact params yet, but we can pre-compute if we assume the regimes listed.
# However, it's safer to compute on demand. The vectorized version is fast enough.


def quote(t, inventory, mid, params):
    """Return optimal bid and ask depths based on pre-computed policy table."""
    steps = params['steps']
    Q = params['Q']
    
    # Get current step
    dt = params['T'] / steps
    n = int(round(t / dt))
    n = max(0, min(n, steps - 1))
    
    # Get policy table
    bid_table, ask_table = _get_policy_table(params)
    
    # Map inventory to index: q in [-Q, Q] -> i in [0, 2Q]
    inv_idx = inventory + Q
    
    # Clip to valid range
    inv_idx = np.clip(inv_idx, 0, 2 * Q)
    
    # Get depths
    bid_depths = bid_table[n, inv_idx]
    ask_depths = ask_table[n, inv_idx]
    
    # Ensure values are valid: >= -3/k, not -inf, not NaN
    k = params['k']
    min_depth = -3.0 / k
    
    # Replace any values below min_depth with min_depth
    bid_depths = np.maximum(bid_depths, min_depth)
    ask_depths = np.maximum(ask_depths, min_depth)
    
    # Replace NaN with a default value
    bid_depths = np.nan_to_num(bid_depths, nan=1.0/k, posinf=np.inf, neginf=min_depth)
    ask_depths = np.nan_to_num(ask_depths, nan=1.0/k, posinf=np.inf, neginf=min_depth)
    
    return bid_depths, ask_depths
