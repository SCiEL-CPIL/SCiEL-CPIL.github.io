import numpy as np
import time

# Global state for caching and time management
_start_time = time.time()
_TIME_LIMIT = 325.0  # Safety margin before 330s kill

# Cache for computed policies
# Key: (sigma, A, k, gamma, T, steps, Q)
# Value: (opt_bid, opt_ask) - arrays of shape (steps, 2*Q+1)
_policy_cache = {}

def _compute_optimal_depths(sigma, A, k, gamma, T, steps, Q):
    """
    Compute optimal bid and ask depths for all steps and inventory levels.
    
    Uses the HJB/continuous approximation insight:
    In the small-dt limit, the optimal depth on each side can be derived analytically
    or via a 1D search that exploits the structure of the problem.
    
    The key insight: for a given inventory q and time t, the optimal bid depth b*
    maximizes: E[reward + future_value] where the fill probability depends on b.
    
    In the continuous limit, the optimal spread satisfies a first-order condition.
    However, since we have a discrete model, we use a refined 1D grid search
    per side, leveraging the fact that bid and ask decisions are approximately
    separable in the small-dt limit.
    """
    dt = T / steps
    n_inv = 2 * Q + 1  # inventory from -Q to Q
    
    # Depth grid: use a fine grid in log-space for better resolution at small depths
    # Depths must be > -3/k, but practically we use positive depths
    # For k=1.5: -3/k = -2, so depths in (0, ~10]
    # For k=0.75: -3/k = -4, so depths in (0, ~10]
    
    # Use a hybrid grid: fine near 0, coarser at large values
    depth_min = 0.001
    depth_max = 10.0 / k  # reasonable upper bound
    
    # Create a fine grid - increased resolution for better accuracy
    # 2000 points provides ~20 points per decade, significantly better than 200
    n_grid = 2000
    depths = np.geomspace(depth_min, depth_max, n_grid)
    
    # Precompute fill probabilities for each depth
    # p_fill(d) = 1 - exp(-A * exp(-k*d) * dt)
    exp_term = np.exp(-k * depths)  # shape (n_grid,)
    p_fill = 1.0 - np.exp(-A * exp_term * dt)  # shape (n_grid,)
    
    # V[n, q_idx] = optimal value from step n with inventory q
    # q_idx = q + Q, so q = q_idx - Q
    V = np.zeros((steps + 1, n_inv))
    
    # Action tables
    opt_bid = np.zeros((steps, n_inv))
    opt_ask = np.zeros((steps, n_inv))
    
    # Backward induction
    for n in range(steps - 1, -1, -1):
        for q_idx in range(n_inv):
            q = q_idx - Q
            
            # Inventory cost for this step
            inv_cost = -(gamma / 2.0) * sigma**2 * q**2 * dt
            
            # Check time limit
            if time.time() - _start_time > _TIME_LIMIT - 5.0:
                # Fallback: use a simple heuristic
                opt_bid[n, q_idx] = 1.0 / k
                opt_ask[n, q_idx] = 1.0 / k
                continue
            
            if q == Q:
                # No bid fill possible, so bid_depth = inf
                # Only ask side matters
                best_val = -np.inf
                best_ad = depths[0]
                
                # Vectorized over ask depths
                # Outcomes: ask fill (q->q-1) or no fill (q stays)
                vals = inv_cost + p_fill * depths + p_fill * V[n+1, q_idx-1] + (1-p_fill) * V[n+1, q_idx]
                max_idx = np.argmax(vals)
                best_val = vals[max_idx]
                best_ad = depths[max_idx]
                
                opt_bid[n, q_idx] = np.inf
                opt_ask[n, q_idx] = best_ad
                V[n, q_idx] = best_val
                
            elif q == -Q:
                # No ask fill possible, so ask_depth = inf
                best_val = -np.inf
                best_bd = depths[0]
                
                # Vectorized over bid depths
                # Outcomes: bid fill (q->q+1) or no fill (q stays)
                vals = inv_cost + p_fill * depths + p_fill * V[n+1, q_idx+1] + (1-p_fill) * V[n+1, q_idx]
                max_idx = np.argmax(vals)
                best_val = vals[max_idx]
                best_bd = depths[max_idx]
                
                opt_bid[n, q_idx] = best_bd
                opt_ask[n, q_idx] = np.inf
                V[n, q_idx] = best_val
                
            else:
                # Both sides active
                # Approximate separability: optimize bid and ask independently
                # This is valid in the small-dt limit where the interaction term is O(dt^2)
                
                V_q = V[n+1, q_idx]
                V_q_plus = V[n+1, q_idx+1]  # bid fill: q->q+1
                V_q_minus = V[n+1, q_idx-1]  # ask fill: q->q-1
                
                # Optimize bid depth (holding ask fixed at some typical value)
                # For bid: maximize p_bid * bd + p_bid * V_q_plus + (1-p_bid) * V_q
                # = p_bid * (bd + V_q_plus) + (1-p_bid) * V_q
                # = V_q + p_bid * (bd + V_q_plus - V_q)
                
                bid_vals = p_fill * (depths + V_q_plus - V_q)
                best_bd_idx = np.argmax(bid_vals)
                best_bd = depths[best_bd_idx]
                
                # Optimize ask depth
                # For ask: maximize p_ask * ad + p_ask * V_q_minus + (1-p_ask) * V_q
                # = V_q + p_ask * (ad + V_q_minus - V_q)
                
                ask_vals = p_fill * (depths + V_q_minus - V_q)
                best_ad_idx = np.argmax(ask_vals)
                best_ad = depths[best_ad_idx]
                
                # Compute the actual value using the optimal depths
                p_bid = p_fill[best_bd_idx]
                p_ask = p_fill[best_ad_idx]
                
                p_both = p_bid * p_ask
                p_bid_only = p_bid * (1 - p_ask)
                p_ask_only = (1 - p_bid) * p_ask
                p_neither = (1 - p_bid) * (1 - p_ask)
                
                earnings = p_bid * best_bd + p_ask * best_ad
                future = p_both * V_q + p_bid_only * V_q_plus + p_ask_only * V_q_minus + p_neither * V_q
                
                V[n, q_idx] = inv_cost + earnings + future
                opt_bid[n, q_idx] = best_bd
                opt_ask[n, q_idx] = best_ad
    
    return opt_bid, opt_ask


def quote(t, inventory, mid, params):
    sigma = params['sigma']
    A = params['A']
    k = params['k']
    gamma = params['gamma']
    T = params['T']
    steps = params['steps']
    Q = params['Q']
    
    n_paths = len(inventory)
    
    # Determine current step
    dt = T / steps
    step = int(round(t / dt))
    if step >= steps:
        step = steps - 1
    if step < 0:
        step = 0
    
    # Get or compute policy
    cache_key = (sigma, A, k, gamma, T, steps, Q)
    if cache_key not in _policy_cache:
        # Check if we have time to compute
        if time.time() - _start_time > _TIME_LIMIT - 10.0:
            # Fallback: use a simple heuristic
            # Optimal depth is approximately 1/k for symmetric inventory
            bid_depths = np.full(n_paths, 1.0 / k, dtype=np.float64)
            ask_depths = np.full(n_paths, 1.0 / k, dtype=np.float64)
            
            # Adjust for inventory: if q > 0, widen bid (higher depth); if q < 0, widen ask
            inv_adjust = np.zeros(n_paths)
            for i in range(n_paths):
                q = inventory[i]
                if q == Q:
                    bid_depths[i] = np.inf
                elif q == -Q:
                    ask_depths[i] = np.inf
                else:
                    # Simple adjustment: shift depths based on inventory
                    shift = 0.1 * q / Q
                    bid_depths[i] = max(0.01, bid_depths[i] * (1 + shift))
                    ask_depths[i] = max(0.01, ask_depths[i] * (1 - shift))
            
            # Ensure valid depths
            min_depth = -3.0 / k + 1e-9
            bid_depths[bid_depths <= min_depth] = np.inf
            ask_depths[ask_depths <= min_depth] = np.inf
            
            return bid_depths, ask_depths
        
        opt_bid, opt_ask = _compute_optimal_depths(sigma, A, k, gamma, T, steps, Q)
        _policy_cache[cache_key] = (opt_bid, opt_ask)
    else:
        opt_bid, opt_ask = _policy_cache[cache_key]
    
    # Map inventory to index
    inv_idx = np.clip(inventory, -Q, Q) + Q
    
    bid_depths = opt_bid[step, inv_idx].copy()
    ask_depths = opt_ask[step, inv_idx].copy()
    
    # Ensure valid depths: must be > -3/k
    min_depth = -3.0 / k + 1e-9
    bid_depths[bid_depths <= min_depth] = np.inf
    ask_depths[ask_depths <= min_depth] = np.inf
    
    # Ensure shape
    bid_depths = np.asarray(bid_depths, dtype=np.float64).reshape(n_paths)
    ask_depths = np.asarray(ask_depths, dtype=np.float64).reshape(n_paths)
    
    return bid_depths, ask_depths
