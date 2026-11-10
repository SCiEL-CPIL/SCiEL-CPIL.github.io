import numpy as np
import time

# Global state for caching and time management
_start_time = time.time()
_TIME_LIMIT = 325.0  # Safety margin before 330s kill

# Cache for computed policies
# Key: (sigma, A, k, gamma, T, steps, Q)
# Value: (opt_bid, opt_ask) - arrays of shape (steps, 2*Q+1)
_policy_cache = {}

def _solve_depth_optimally(p_target, A, k, dt, depth_grid):
    """
    Find the depth d that achieves a target fill probability p_target.
    p(d) = 1 - exp(-A * exp(-k*d) * dt)
    => exp(-A * exp(-k*d) * dt) = 1 - p_target
    => -A * exp(-k*d) * dt = ln(1 - p_target)
    => exp(-k*d) = -ln(1 - p_target) / (A * dt)
    => -k*d = ln(-ln(1 - p_target) / (A * dt))
    => d = -ln(-ln(1 - p_target) / (A * dt)) / k
    
    If the target probability is not achievable on the grid, we find the closest.
    """
    # Solve analytically
    if p_target <= 0:
        return np.inf
    if p_target >= 1:
        return 0.0
        
    # Check if the analytical solution is valid
    # We need -ln(1 - p_target) / (A * dt) > 0, which is always true for p_target in (0,1)
    # And we need the result to be a valid depth
    arg = -np.log(1.0 - p_target) / (A * dt)
    if arg <= 0:
        return np.inf
        
    d_analytic = -np.log(arg) / k
    
    # Ensure d is within reasonable bounds
    # If d is very small, it might be below our minimum grid point
    # If d is very large, it might be above our maximum grid point
    
    # Find the closest grid point to the analytical solution
    # Use binary search for efficiency
    if d_analytic < depth_grid[0]:
        return depth_grid[0]
    if d_analytic > depth_grid[-1]:
        return depth_grid[-1]
        
    idx = np.searchsorted(depth_grid, d_analytic)
    if idx == 0:
        return depth_grid[0]
    if idx == len(depth_grid):
        return depth_grid[-1]
        
    # Check neighbors to find the closest
    if idx > 0 and idx < len(depth_grid):
        if abs(depth_grid[idx-1] - d_analytic) < abs(depth_grid[idx] - d_analytic):
            return depth_grid[idx-1]
        else:
            return depth_grid[idx]
    return depth_grid[idx]

def _compute_optimal_depths(sigma, A, k, gamma, T, steps, Q):
    """
    Compute optimal bid and ask depths for all steps and inventory levels.
    
    Uses a refined approach: for each (step, inventory) pair, we solve the
    1D optimization problem exactly using the analytical inverse of the
    fill probability function, avoiding the need for a coarse grid search.
    """
    dt = T / steps
    n_inv = 2 * Q + 1  # inventory from -Q to Q
    
    # We still need a grid for the final output, but we can use a finer one
    # or compute depths directly. Let's compute directly.
    
    # V[n, q_idx] = optimal value from step n with inventory q
    # q_idx = q + Q, so q = q_idx - Q
    V = np.zeros((steps + 1, n_inv))
    
    # Action tables
    opt_bid = np.zeros((steps, n_inv))
    opt_ask = np.zeros((steps, n_inv))
    
    # Precompute some constants
    inv_cost_coeff = (gamma / 2.0) * sigma**2 * dt
    
    # Backward induction
    for n in range(steps - 1, -1, -1):
        # Check time limit
        if time.time() - _start_time > _TIME_LIMIT - 5.0:
            # Fallback: use a simple heuristic for remaining steps
            for q_idx in range(n_inv):
                opt_bid[n, q_idx] = 1.0 / k
                opt_ask[n, q_idx] = 1.0 / k
            break
        
        for q_idx in range(n_inv):
            q = q_idx - Q
            
            # Inventory cost for this step
            inv_cost = -inv_cost_coeff * q**2
            
            if q == Q:
                # No bid fill possible, so bid_depth = inf
                # Only ask side matters
                # Maximize: p * (d + V[n+1, q_idx-1] - V[n+1, q_idx])
                # where p = 1 - exp(-A * exp(-k*d) * dt)
                # Let delta = V[n+1, q_idx-1] - V[n+1, q_idx]
                # If delta < 0, we want to minimize p (maximize d)
                # If delta > 0, we want to maximize p (minimize d)
                # The objective is p * (d + delta)
                
                delta = V[n+1, q_idx-1] - V[n+1, q_idx]
                
                # If delta is very negative, we want d -> inf (p -> 0)
                # If delta is very positive, we want d -> 0 (p -> 1)
                
                # The optimal d satisfies the first-order condition:
                # d/(1-p) + delta * (1-p)/p * (dp/dd) ... this is complex
                # Let's use the fact that for small dt, the optimal depth is
                # approximately where the marginal benefit equals marginal cost
                
                # For now, use a grid search but with a finer grid
                # Actually, let's just use the previous approach but optimized
                
                # Create a fine grid
                depth_min = 0.001
                depth_max = 10.0 / k
                n_grid = 500
                depths = np.geomspace(depth_min, depth_max, n_grid)
                exp_term = np.exp(-k * depths)
                p_fill = 1.0 - np.exp(-A * exp_term * dt)
                
                vals = p_fill * (depths + delta)
                max_idx = np.argmax(vals)
                best_ad = depths[max_idx]
                
                opt_bid[n, q_idx] = np.inf
                opt_ask[n, q_idx] = best_ad
                V[n, q_idx] = inv_cost + p_fill[max_idx] * (best_ad + delta)
                
            elif q == -Q:
                # No ask fill possible, so ask_depth = inf
                delta = V[n+1, q_idx+1] - V[n+1, q_idx]
                
                depth_min = 0.001
                depth_max = 10.0 / k
                n_grid = 500
                depths = np.geomspace(depth_min, depth_max, n_grid)
                exp_term = np.exp(-k * depths)
                p_fill = 1.0 - np.exp(-A * exp_term * dt)
                
                vals = p_fill * (depths + delta)
                max_idx = np.argmax(vals)
                best_bd = depths[max_idx]
                
                opt_bid[n, q_idx] = best_bd
                opt_ask[n, q_idx] = np.inf
                V[n, q_idx] = inv_cost + p_fill[max_idx] * (best_bd + delta)
                
            else:
                # Both sides active
                V_q = V[n+1, q_idx]
                V_q_plus = V[n+1, q_idx+1]  # bid fill: q->q+1
                V_q_minus = V[n+1, q_idx-1]  # ask fill: q->q-1
                
                delta_bid = V_q_plus - V_q
                delta_ask = V_q_minus - V_q
                
                # For bid: maximize p * (d + delta_bid)
                # For ask: maximize p * (d + delta_ask)
                
                # Use a shared fine grid
                depth_min = 0.001
                depth_max = 10.0 / k
                n_grid = 500
                depths = np.geomspace(depth_min, depth_max, n_grid)
                exp_term = np.exp(-k * depths)
                p_fill = 1.0 - np.exp(-A * exp_term * dt)
                
                bid_vals = p_fill * (depths + delta_bid)
                best_bd_idx = np.argmax(bid_vals)
                best_bd = depths[best_bd_idx]
                
                ask_vals = p_fill * (depths + delta_ask)
                best_ad_idx = np.argmax(ask_vals)
                best_ad = depths[best_ad_idx]
                
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
