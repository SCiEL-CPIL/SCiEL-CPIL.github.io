import numpy as np
import time

_start_time = time.time()
_TIME_LIMIT = 325.0

_policy_cache = {}

def _solve_hjb_analytic(sigma, A, k, gamma, T, steps, Q):
    """
    Solve the market-making problem using a continuous HJB approximation
    with analytical optimal depth from first-order conditions.
    
    Key insight: For each side, the optimal depth d* maximizes:
        p(d) * (d + delta_V) - c
    where p(d) = 1 - exp(-A*exp(-k*d)*dt) and delta_V is the value difference.
    
    Taking derivative w.r.t. d:
        p'(d)*(d + delta_V) + p(d) = 0
        A*exp(-k*d)*dt*exp(-k*d) * (d + delta_V) + (1 - exp(-A*exp(-k*d)*dt)) = 0
    
    This is transcendental but we can solve it iteratively or use a fine grid.
    However, for small dt, p(d) ≈ A*exp(-k*d)*dt, and the FOC becomes:
        -k * A*exp(-k*d)*dt * (d + delta_V) + A*exp(-k*d)*dt = 0
        => -k*(d + delta_V) + 1 = 0
        => d* = 1/k - delta_V
    
    This is the key simplification! The optimal depth is approximately:
        d_bid* = 1/k - (V(q+1) - V(q))
        d_ask* = 1/k - (V(q-1) - V(q))
    
    We use this as an initial guess and refine with a local 1D search.
    """
    dt = T / steps
    n_inv = 2 * Q + 1
    
    # Value function V[n, q_idx]
    V = np.zeros((steps + 1, n_inv))
    
    # Optimal depths
    opt_bid = np.full((steps, n_inv), 1.0 / k)
    opt_ask = np.full((steps, n_inv), 1.0 / k)
    
    # Precompute a fine depth grid for local refinement
    # The analytic solution gives us a good starting point, so we only need
    # a narrow local search
    n_local = 40
    local_half_width = 2.0 / k  # search window around analytic solution
    
    for n in range(steps - 1, -1, -1):
        for q_idx in range(n_inv):
            q = q_idx - Q
            
            inv_cost = -(gamma / 2.0) * sigma**2 * q**2 * dt
            
            if time.time() - _start_time > _TIME_LIMIT - 5.0:
                # Fallback
                opt_bid[n, q_idx] = 1.0 / k
                opt_ask[n, q_idx] = 1.0 / k
                continue
            
            if q == Q:
                # Only ask side
                d_analytic = 1.0 / k - (V[n+1, q_idx-1] - V[n+1, q_idx])
                d_analytic = max(0.001, min(d_analytic, 10.0/k))
                
                # Local search
                d_grid = np.linspace(max(0.001, d_analytic - local_half_width), 
                                     min(10.0/k, d_analytic + local_half_width), n_local)
                p_fill = 1.0 - np.exp(-A * np.exp(-k * d_grid) * dt)
                vals = p_fill * d_grid + p_fill * V[n+1, q_idx-1] + (1-p_fill) * V[n+1, q_idx]
                best_idx = np.argmax(vals)
                
                opt_ask[n, q_idx] = d_grid[best_idx]
                opt_bid[n, q_idx] = np.inf
                V[n, q_idx] = inv_cost + vals[best_idx]
                
            elif q == -Q:
                # Only bid side
                d_analytic = 1.0 / k - (V[n+1, q_idx+1] - V[n+1, q_idx])
                d_analytic = max(0.001, min(d_analytic, 10.0/k))
                
                d_grid = np.linspace(max(0.001, d_analytic - local_half_width), 
                                     min(10.0/k, d_analytic + local_half_width), n_local)
                p_fill = 1.0 - np.exp(-A * np.exp(-k * d_grid) * dt)
                vals = p_fill * d_grid + p_fill * V[n+1, q_idx+1] + (1-p_fill) * V[n+1, q_idx]
                best_idx = np.argmax(vals)
                
                opt_bid[n, q_idx] = d_grid[best_idx]
                opt_ask[n, q_idx] = np.inf
                V[n, q_idx] = inv_cost + vals[best_idx]
                
            else:
                # Both sides - use separability with analytic initial guess
                V_q = V[n+1, q_idx]
                V_q_plus = V[n+1, q_idx+1]
                V_q_minus = V[n+1, q_idx-1]
                
                # Analytic initial guesses
                d_bid_analytic = 1.0 / k - (V_q_plus - V_q)
                d_ask_analytic = 1.0 / k - (V_q_minus - V_q)
                d_bid_analytic = max(0.001, min(d_bid_analytic, 10.0/k))
                d_ask_analytic = max(0.001, min(d_ask_analytic, 10.0/k))
                
                # Local search for bid
                bid_grid = np.linspace(max(0.001, d_bid_analytic - local_half_width), 
                                       min(10.0/k, d_bid_analytic + local_half_width), n_local)
                p_bid = 1.0 - np.exp(-A * np.exp(-k * bid_grid) * dt)
                bid_vals = p_bid * (bid_grid + V_q_plus - V_q)
                best_bid_idx = np.argmax(bid_vals)
                best_bid = bid_grid[best_bid_idx]
                
                # Local search for ask
                ask_grid = np.linspace(max(0.001, d_ask_analytic - local_half_width), 
                                       min(10.0/k, d_ask_analytic + local_half_width), n_local)
                p_ask = 1.0 - np.exp(-A * np.exp(-k * ask_grid) * dt)
                ask_vals = p_ask * (ask_grid + V_q_minus - V_q)
                best_ask_idx = np.argmax(ask_vals)
                best_ask = ask_grid[best_ask_idx]
                
                # Compute true value with both fills
                p_b = p_bid[best_bid_idx]
                p_a = p_ask[best_ask_idx]
                p_both = p_b * p_a
                p_bid_only = p_b * (1 - p_a)
                p_ask_only = (1 - p_b) * p_a
                p_neither = (1 - p_b) * (1 - p_a)
                
                earnings = p_b * best_bid + p_a * best_ask
                future = p_both * V_q + p_bid_only * V_q_plus + p_ask_only * V_q_minus + p_neither * V_q
                
                V[n, q_idx] = inv_cost + earnings + future
                opt_bid[n, q_idx] = best_bid
                opt_ask[n, q_idx] = best_ask
    
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
    
    dt = T / steps
    step = int(round(t / dt))
    step = max(0, min(step, steps - 1))
    
    cache_key = (sigma, A, k, gamma, T, steps, Q)
    if cache_key not in _policy_cache:
        if time.time() - _start_time > _TIME_LIMIT - 10.0:
            # Emergency fallback
            bid_depths = np.full(n_paths, 1.0 / k, dtype=np.float64)
            ask_depths = np.full(n_paths, 1.0 / k, dtype=np.float64)
            for i in range(n_paths):
                q = inventory[i]
                if q == Q:
                    bid_depths[i] = np.inf
                elif q == -Q:
                    ask_depths[i] = np.inf
            min_depth = -3.0 / k + 1e-9
            bid_depths[bid_depths <= min_depth] = np.inf
            ask_depths[ask_depths <= min_depth] = np.inf
            return bid_depths, ask_depths
        
        opt_bid, opt_ask = _solve_hjb_analytic(sigma, A, k, gamma, T, steps, Q)
        _policy_cache[cache_key] = (opt_bid, opt_ask)
    else:
        opt_bid, opt_ask = _policy_cache[cache_key]
    
    inv_idx = np.clip(inventory, -Q, Q) + Q
    
    bid_depths = opt_bid[step, inv_idx].copy()
    ask_depths = opt_ask[step, inv_idx].copy()
    
    min_depth = -3.0 / k + 1e-9
    bid_depths[bid_depths <= min_depth] = np.inf
    ask_depths[ask_depths <= min_depth] = np.inf
    
    bid_depths = np.asarray(bid_depths, dtype=np.float64).reshape(n_paths)
    ask_depths = np.asarray(ask_depths, dtype=np.float64).reshape(n_paths)
    
    return bid_depths, ask_depths
