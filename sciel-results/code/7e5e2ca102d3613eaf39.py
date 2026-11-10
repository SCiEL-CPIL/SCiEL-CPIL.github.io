"""Inventory-aware market making with dynamic programming."""
import numpy as np
import time

# Cache for pre-computed optimal depths per regime
_cache = {}

def _precompute_policy(sigma, A, k, gamma, T, steps, Q):
    """
    Pre-compute optimal bid and ask depths for each (step, inventory) pair.
    
    Returns:
        bid_depths: array of shape (steps, 2*Q+1) 
        ask_depths: array of shape (steps, 2*Q+1)
    """
    dt = T / steps
    n_states = 2 * Q + 1  # inventory from -Q to Q
    
    # We'll work backwards from step steps-1 to step 0
    # V[n, q] = optimal expected value from step n with inventory q
    # At each step, we choose bid_depth d_b >= 0 and ask_depth d_a >= 0
    # 
    # Fill probabilities:
    #   p_b = 1 - exp(-A * exp(-k * d_b) * dt)  if q < Q, else 0
    #   p_a = 1 - exp(-A * exp(-k * d_a) * dt)  if q > -Q, else 0
    #
    # If bid fills: gain d_b, inventory becomes q+1
    # If ask fills: gain d_a, inventory becomes q-1
    # Both can fill independently
    #
    # Cost at this step: (gamma/2) * sigma^2 * q^2 * dt
    #
    # V[n, q] = -cost + max over (d_b, d_a) of:
    #   (1-p_b)(1-p_a) * V[n+1, q]
    #   + p_b(1-p_a) * (d_b + V[n+1, q+1])
    #   + (1-p_b)p_a * (d_a + V[n+1, q-1])
    #   + p_b*p_a * (d_b + d_a + V[n+1, q])
    #
    # This is complex. Let's use a grid search over depths.
    
    # Grid of candidate depths
    # The optimal depth is likely in range [0, 5/k] or so
    # Let's use a fine grid
    max_depth = 5.0 / k
    n_grid = 200
    depth_grid = np.linspace(0, max_depth, n_grid)
    
    # V[n, q] for n in [0, steps], q in [-Q, Q]
    # V[steps, q] = 0 (no terminal charge)
    V = np.zeros((steps + 1, n_states))
    bid_opt = np.zeros((steps, n_states))
    ask_opt = np.zeros((steps, n_states))
    
    # Pre-compute fill probabilities for each depth
    # p_b(d) = 1 - exp(-A * exp(-k * d) * dt)
    p_grid = 1.0 - np.exp(-A * np.exp(-k * depth_grid) * dt)
    
    # Work backwards
    for n in range(steps - 1, -1, -1):
        cost = (gamma / 2.0) * sigma**2 * dt  # will multiply by q^2 later
        
        for q_idx in range(n_states):
            q = q_idx - Q  # actual inventory
            
            # Cannot bid if q == Q, cannot ask if q == -Q
            can_bid = (q < Q)
            can_ask = (q > -Q)
            
            best_val = -np.inf
            best_db = 0.0
            best_da = 0.0
            
            # If we can't bid, set p_b = 0; if we can't ask, set p_a = 0
            # We need to search over (d_b, d_a) but constrained
            
            if can_bid and can_ask:
                # Search over both depths
                # This is a 2D grid search - expensive but doable with vectorization
                # For each (i_b, i_a), compute the value
                # p_b = p_grid[i_b], p_a = p_grid[i_a]
                # d_b = depth_grid[i_b], d_a = depth_grid[i_a]
                
                # V[n+1, q], V[n+1, q+1], V[n+1, q-1]
                V0 = V[n+1, q_idx]
                V1 = V[n+1, q_idx + 1]  # q+1
                Vm1 = V[n+1, q_idx - 1]  # q-1
                
                # Vectorized computation
                # For each i_b in range(n_grid), i_a in range(n_grid):
                # val = (1-p_b)(1-p_a)*V0 + p_b(1-p_a)*(d_b+V1) + (1-p_b)p_a*(d_a+Vm1) + p_b*p_a*(d_b+d_a+V0)
                
                # This is a 200x200 grid = 40000 evaluations per (n,q)
                # Total: 1000 * 61 * 40000 = 2.44 billion - too slow!
                
                # Need a smarter approach. Let's separate the optimization.
                # For fixed d_b, optimize over d_a, then optimize over d_b.
                
                # Actually, let's use a coarser grid first
                n_grid_coarse = 50
                depth_grid_c = np.linspace(0, max_depth, n_grid_coarse)
                p_grid_c = 1.0 - np.exp(-A * np.exp(-k * depth_grid_c) * dt)
                
                best_val_c = -np.inf
                best_db_c = 0.0
                best_da_c = 0.0
                
                for i_b in range(n_grid_coarse):
                    pb = p_grid_c[i_b]
                    db = depth_grid_c[i_b]
                    for i_a in range(n_grid_coarse):
                        pa = p_grid_c[i_a]
                        da = depth_grid_c[i_a]
                        
                        val = ((1-pb)*(1-pa)*V0 + 
                               pb*(1-pa)*(db+V1) + 
                               (1-pb)*pa*(da+Vm1) + 
                               pb*pa*(db+da+V0))
                        
                        if val > best_val_c:
                            best_val_c = val
                            best_db_c = db
                            best_da_c = da
                
                # Refine around the best with a finer grid
                if best_val_c > -np.inf:
                    # Local refinement
                    margin = max_depth / n_grid_coarse * 2
                    db_lo = max(0, best_db_c - margin)
                    db_hi = min(max_depth, best_db_c + margin)
                    da_lo = max(0, best_da_c - margin)
                    da_hi = min(max_depth, best_da_c + margin)
                    
                    n_refine = 20
                    db_grid_r = np.linspace(db_lo, db_hi, n_refine)
                    da_grid_r = np.linspace(da_lo, da_hi, n_refine)
                    pb_grid_r = 1.0 - np.exp(-A * np.exp(-k * db_grid_r) * dt)
                    pa_grid_r = 1.0 - np.exp(-A * np.exp(-k * da_grid_r) * dt)
                    
                    for i_b in range(n_refine):
                        pb = pb_grid_r[i_b]
                        db = db_grid_r[i_b]
                        for i_a in range(n_refine):
                            pa = pa_grid_r[i_a]
                            da = da_grid_r[i_a]
                            
                            val = ((1-pb)*(1-pa)*V0 + 
                                   pb*(1-pa)*(db+V1) + 
                                   (1-pb)*pa*(da+Vm1) + 
                                   pb*pa*(db+da+V0))
                            
                            if val > best_val_c:
                                best_val_c = val
                                best_db_c = db
                                best_da_c = da
                
                best_val = best_val_c
                best_db = best_db_c
                best_da = best_da_c
                
            elif can_bid:
                # Only bid side active
                V0 = V[n+1, q_idx]
                V1 = V[n+1, q_idx + 1]
                
                best_val_b = -np.inf
                best_db_b = 0.0
                for i_b in range(n_grid):
                    pb = p_grid[i_b]
                    db = depth_grid[i_b]
                    val = (1-pb)*V0 + pb*(db+V1)
                    if val > best_val_b:
                        best_val_b = val
                        best_db_b = db
                best_val = best_val_b
                best_db = best_db_b
                best_da = 0.0
            elif can_ask:
                # Only ask side active
                V0 = V[n+1, q_idx]
                Vm1 = V[n+1, q_idx - 1]
                
                best_val_a = -np.inf
                best_da_a = 0.0
                for i_a in range(n_grid):
                    pa = p_grid[i_a]
                    da = depth_grid[i_a]
                    val = (1-pa)*V0 + pa*(da+Vm1)
                    if val > best_val_a:
                        best_val_a = val
                        best_da_a = da
                best_val = best_val_a
                best_db = 0.0
                best_da = best_da_a
            else:
                # Neither side active (q == Q and q == -Q, impossible unless Q==0)
                best_val = V[n+1, q_idx]
                best_db = 0.0
                best_da = 0.0
            
            # Add the cost
            V[n, q_idx] = best_val - cost * q**2
            bid_opt[n, q_idx] = best_db
            ask_opt[n, q_idx] = best_da
    
    return bid_opt, ask_opt


def quote(t, inventory, mid, params):
    """
    Inventory-aware market making policy using pre-computed optimal depths.
    """
    sigma = params['sigma']
    A = params['A']
    k = params['k']
    gamma = params['gamma']
    T = params['T']
    steps = params['steps']
    Q = params['Q']
    
    # Check deadline
    if not hasattr(quote, '_start_time'):
        quote._start_time = time.time()
    elapsed = time.time() - quote._start_time
    if elapsed > 280:
        # Return a safe default
        depth = np.full(len(inventory), 1.0 / k)
        return depth, depth.copy()
    
    # Get or pre-compute the policy
    cache_key = (sigma, A, k, gamma, T, steps, Q)
    if cache_key not in _cache:
        bid_opt, ask_opt = _precompute_policy(sigma, A, k, gamma, T, steps, Q)
        _cache[cache_key] = (bid_opt, ask_opt)
    else:
        bid_opt, ask_opt = _cache[cache_key]
    
    # Determine current step
    dt = T / steps
    n = int(round(t / dt))
    n = max(0, min(n, steps - 1))
    
    # Map inventory to index
    inv = np.asarray(inventory, dtype=np.int32)
    inv_idx = inv + Q  # 0 to 2*Q
    
    # Clip to valid range
    inv_idx = np.clip(inv_idx, 0, 2 * Q)
    
    # Get optimal depths
    bid_depths = bid_opt[n, inv_idx]
    ask_depths = ask_opt[n, inv_idx]
    
    # Ensure valid values
    bid_depths = np.clip(bid_depths, 0, np.inf)
    ask_depths = np.clip(ask_depths, 0, np.inf)
    
    # Handle boundary cases: no bid at q=Q, no ask at q=-Q
    bid_depths[inv == Q] = np.inf
    ask_depths[inv == -Q] = np.inf
    
    return bid_depths, ask_depths
