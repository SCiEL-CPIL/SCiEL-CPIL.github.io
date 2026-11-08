"""
Strategy: Analytic HJB approximation with quadratic value function.
Instead of grid search, we assume V(t, q) is quadratic in q and solve for coefficients
c(t), d(t), e(t) backwards in time.
The optimal depth is found by maximizing the Bellman expression, which can be done
via gradient ascent or Newton's method on the continuous depth variable.
"""
# EVOLVE-BLOCK-START
import numpy as np
import time

_policy_cache = {}
_start_time = time.time()
TIME_LIMIT = 320.0

def _compute_optimal_policy(params):
    sigma = params['sigma']
    A = params['A']
    k = params['k']
    gamma = params['gamma']
    T = params['T']
    steps = params['steps']
    Q = params['Q']
    
    dt = T / steps
    n_states = 2 * Q + 1
    q_values = np.arange(-Q, Q + 1, dtype=np.float64)
    
    opt_bid = np.zeros((steps, n_states))
    opt_ask = np.zeros((steps, n_states))
    
    # Coefficients for V(t, q) = 0.5 * c * q^2 + d * q + e
    # At t=T, V=0, so c=0, d=0, e=0
    c_next = 0.0
    d_next = 0.0
    e_next = 0.0
    
    # We store coefficients for each time step to allow exact depth optimization
    # Actually, we can compute optimal depths on the fly if we store c, d, e for each step
    # But quote() is called at specific t, so we need to look up or recompute.
    # Storing 1000 * 3 floats is cheap.
    
    c_coeff = np.zeros(steps)
    d_coeff = np.zeros(steps)
    e_coeff = np.zeros(steps)
    
    # Initialize last step
    c_coeff[steps-1] = 0.0
    d_coeff[steps-1] = 0.0
    e_coeff[steps-1] = 0.0
    
    # Backward induction
    for t in range(steps - 2, -1, -1):
        if time.time() - _start_time > TIME_LIMIT:
            # If we run out of time, we have computed some steps. 
            # The remaining steps (earlier in time) will use the last computed coefficients?
            # No, that's wrong. We need to fill the rest.
            # If we break, we haven't computed t-1, t-2...
            # We can just leave them as 0 or copy the last valid one?
            # Better: if we break, we should return what we have, but quote() needs all steps.
            # Since steps=1000 and computation is fast, we should not break.
            # But for safety, if we break, we fill remaining with last known values.
            pass
        
        c_curr = c_next
        d_curr = d_next
        e_curr = e_next
        
        # We want to find c_new, d_new, e_new such that:
        # V_curr(q) = max_{b, a} { E[V_next(q')] - cost(q) }
        # V_next(q') is quadratic.
        # The expectation involves probabilities that depend on b, a.
        # This is not linear in b, a, so V_curr is not exactly quadratic.
        # However, we can approximate the optimal b, a for each q, and then fit a quadratic to V_curr(q).
        # Or, we can assume the optimal b, a are constants or simple functions of q.
        
        # Let's try a different approach: 
        # For each q, we know V_next(q+1), V_next(q), V_next(q-1) from the quadratic form.
        # We want to maximize:
        # (1-pb)(1-pa) V_next(q) + pb(1-pa)(V_next(q+1) + b) + (1-pb)pa(V_next(q-1) + a) + pb pa (V_next(q) + b + a)
        # - cost(q)
        
        # This is a function of b and a. We can optimize b and a for each q.
        # Since there are only 61 states (Q=30), we can do this for each q independently.
        # Then we have V_curr(q) for all q. We fit a quadratic to these points to get c_new, d_new, e_new.
        
        # This hybrid approach: Grid-free depth optimization (via Newton/Gradient) + Quadratic Fit for Value Function.
        
        V_next_q = 0.5 * c_next * q_values**2 + d_next * q_values + e_next
        V_next_qp = 0.5 * c_next * (q_values + 1)**2 + d_next * (q_values + 1) + e_next
        V_next_qm = 0.5 * c_next * (q_values - 1)**2 + d_next * (q_values - 1) + e_next
        
        # Handle boundaries
        # At q=Q, no bid fill. So pb=0 effectively? No, we just don't quote bid.
        # The problem says "no bid fill at q=Q". This implies bid_depth = inf or we just don't place bid.
        # If we don't place bid, pb=0.
        # Similarly for q=-Q, ask_depth = inf, pa=0.
        
        V_curr = np.zeros(n_states)
        opt_b = np.zeros(n_states)
        opt_a = np.zeros(n_states)
        
        for i in range(n_states):
            q = q_values[i]
            v_q = V_next_q[i]
            
            if q == Q:
                v_qp = 0.0 # Not used
                pb_opt = 0.0 # No bid
                # Optimize a only
                # f(a) = (1-pa) v_q + pa (v_qm + a) - cost
                # cost = 0.5 * gamma * sigma**2 * q**2 * dt
                cost = 0.5 * gamma * sigma**2 * q**2 * dt
                v_qm = V_next_qm[i]
                
                # Maximize (1-pa) v_q + pa (v_qm + a)
                # pa = 1 - exp(-A exp(-k a) dt)
                # This is 1D optimization.
                # We can use scipy.optimize or simple grid/Newton.
                # Let's use a small grid for safety and speed, or Newton.
                # Newton on g(a) = (1-pa) v_q + pa (v_qm + a)
                # g'(a) = -pa' (v_q - v_qm - a) + pa + (v_qm + a) pa'
                # pa' = A exp(-k a) k exp(-k a) dt * exp(-A exp(-k a) dt) ? No.
                # pa = 1 - exp(-lambda(a)), lambda = A exp(-k a) dt
                # pa' = -exp(-lambda) * (-lambda') = exp(-lambda) lambda'
                # lambda' = A (-k) exp(-k a) dt = -k A exp(-k a) dt
                # So pa' = -k A exp(-k a) dt exp(-pa)
                
                # Let's just do a fine grid search for 1D, it's fast enough for 61 points * 1000 steps?
                # 61 * 1000 = 61000 points. Grid of 100 points -> 6M operations. Very fast.
                # But we can do better.
                
                # Let's define a helper to optimize 1D or 2D.
                
                pass # Placeholder for logic below
            
            # General case: Optimize b, a
            # We will use a coarse-to-fine grid or Newton.
            # Given the time limit and 1000 steps, we need efficiency.
            # Let's use a fixed small grid for b and a? No, that's what we are trying to avoid.
            # Let's use Newton's method.
            
            v_qp = V_next_qp[i] if q < Q else 0.0
            v_qm = V_next_qm[i] if q > -Q else 0.0
            cost = 0.5 * gamma * sigma**2 * q**2 * dt
            
            # Objective J(b, a)
            # pb = 1 - exp(-A exp(-k b) dt)
            # pa = 1 - exp(-A exp(-k a) dt)
            
            # Start with initial guess
            # Heuristic: b, a ~ 0.1
            b = 0.1
            a = 0.1
            
            # Newton iterations
            for _ in range(5):
                pb = 1.0 - np.exp(-A * np.exp(-k * b) * dt)
                pa = 1.0 - np.exp(-A * np.exp(-k * a) * dt)
                
                # If q == Q, force pb = 0
                if q == Q:
                    pb = 0.0
                    # Gradient wrt b is 0? No, we don't vary b.
                if q == -Q:
                    pa = 0.0
                
                J = (1-pb)*(1-pa)*v_q + pb*(1-pa)*(v_qp + b) + (1-pb)*pa*(v_qm + a) + pb*pa*(v_q + b + a) - cost
                
                # Gradients
                # dJ/db
                if q == Q:
                    dJ_db = 0.0
                else:
                    # pb' = -k A exp(-k b) dt exp(-pb)
                    pb_p = -k * A * np.exp(-k * b) * dt * np.exp(-pb)
                    # J = (1-pb)(1-pa)v_q + pb(1-pa)(v_qp+b) + ...
                    # dJ/db = -pb'(1-pa)v_q + pb'(1-pa)(v_qp+b) + pb(1-pa) + ... (terms with pa)
                    # = pb'(1-pa)(v_qp - v_q + b) + pb(1-pa) + pb' pa (v_q - v_qm - a) + pb pa # ?
                    # Let's expand carefully.
                    # J = v_q - pb v_q - pa v_q + pb pa v_q + pb v_qp + pb^2 - pb pa v_qp + pa v_qm + pa a - pb pa v_qm + pb pa b + pb pa a - pb pa^2 ? No.
                    # Let's use the explicit form:
                    # J = (1-pb)(1-pa)v_q + pb(1-pa)(v_qp+b) + (1-pb)pa(v_qm+a) + pb pa (v_q+b+a) - cost
                    # dJ/db = 
                    #   -pb'(1-pa)v_q + pb'(1-pa)(v_qp+b) + pb(1-pa) 
                    #   -pb' pa (v_qm+a) 
                    #   + pb' pa (v_q+b+a) + pb pa 
                    # Simplify:
                    # = pb' [ (1-pa)(v_qp+b-v_q) - pa(v_qm+a) + pa(v_q+b+a) ] + pb [ (1-pa) + pa ]
                    # = pb' [ (1-pa)(v_qp+b-v_q) + pa(v_q+b-v_qm) ] + pb
                    
                if q == -Q:
                    dJ_da = 0.0
                else:
                    pa_p = -k * A * np.exp(-k * a) * dt * np.exp(-pa)
                    # dJ/da = pa' [ (1-pb)(v_qm+a-v_q) + pb(v_q+a-v_qp) ] + pa
                    dJ_da = pa_p * ( (1-pb)*(v_qm+a-v_q) + pb*(v_q+a-v_qp) ) + pa
                
                # Hessian? Or just use finite differences for Hessian?
                # For 2D Newton, we need Hessian.
                # Given complexity, maybe use a simple grid search with refinement?
                # Or use scipy.optimize.minimize?
                # scipy is available.
                
                break # Skip Newton for now, use grid search in the loop below for reliability
                
            # Let's do a local grid search around the previous step's optimal values or 0.1
            # Since we don't have previous optimal values stored per state easily in this loop structure without extra memory,
            # let's use a small grid around 0.0 to 0.5.
            
            # To save time, we can use a very small grid, e.g., 20x20.
            # But we want accuracy.
            # Let's use 50x50 grid for the first few steps, then maybe less?
            # 61 states * 1000 steps * 2500 grid points = 152.5M evaluations.
            # Each evaluation is a few exps. Might be too slow in Python loop.
            
            # Vectorized grid search!
            # We can vectorize over the grid for all states?
            # No, v_q, v_qp, v_qm are different for each state.
            # But we can create a 3D array (n_states, n_grid_b, n_grid_a) and compute J for all.
            
            pass

    # The above loop is getting complicated. Let's restart the strategy implementation with a cleaner vectorized approach.
    # We will compute optimal b, a for all states simultaneously using vectorized numpy operations on a grid.
    # Then fit quadratic to V_curr.
    
    c_coeff[:] = 0.0
    d_coeff[:] = 0.0
    e_coeff[:] = 0.0
    
    c_next = 0.0
    d_next = 0.0
    e_next = 0.0
    
    # Grid for depths
    # max_depth 3.0, 50 points is reasonable.
    n_grid = 50
    grid = np.linspace(0, 3.0, n_grid)
    
    for t in range(steps - 2, -1, -1):
        if time.time() - _start_time > TIME_LIMIT:
            break
            
        q_vals = q_values # (n_states,)
        
        V_next_q = 0.5 * c_next * q_vals**2 + d_next * q_vals + e_next
        V_next_qp = 0.5 * c_next * (q_vals + 1)**2 + d_next * (q_vals + 1) + e_next
        V_next_qm = 0.5 * c_next * (q_vals - 1)**2 + d_next * (q_vals - 1) + e_next
        
        # Set boundary conditions for V_next
        # At q=Q, bid fill is impossible. This is handled by setting pb=0 in the objective.
        # We don't need to change V_next_qp, we just multiply by pb which will be 0.
        
        cost = 0.5 * gamma * sigma**2 * q_vals**2 * dt # (n_states,)
        
        # Create grid arrays
        b_grid = grid[None, :, None] # (1, n_grid, 1)
        a_grid = grid[None, None, :] # (1, 1, n_grid)
        
        # Compute pb, pa for all grid points
        # pb depends on b, pa on a.
        pb_grid = 1.0 - np.exp(-A * np.exp(-k * b_grid) * dt) # (1, n_grid, 1)
        pa_grid = 1.0 - np.exp(-A * np.exp(-k * a_grid) * dt) # (1, 1, n_grid)
        
        # Broadcast to (n_states, n_grid, n_grid)
        # v_q: (n_states, 1, 1)
        v_q_b = V_next_q[:, None, None]
        v_qp_b = V_next_qp[:, None, None]
        v_qm_b = V_next_qm[:, None, None]
        cost_b = cost[:, None, None]
        
        # Mask for boundaries
        # q == Q -> index 2Q. Set pb = 0.
        # q == -Q -> index 0. Set pa = 0.
        pb_full = np.broadcast_to(pb_grid, (n_states, n_grid, n_grid)).copy()
        pa_full = np.broadcast_to(pa_grid, (n_states, n_grid, n_grid)).copy()
        
        # Apply masks
        # Index 2Q is the last state
        pb_full[-1, :, :] = 0.0
        # Index 0 is the first state
        pa_full[0, :, :] = 0.0
        
        # Compute J
        t1 = v_q_b * (1-pb_full) * (1-pa_full)
        t2 = v_qp_b * pb_full * (1-pa_full) + b_grid * pb_full * (1-pa_full)
        t3 = v_qm_b * (1-pb_full) * pa_full + a_grid * (1-pb_full) * pa_full
        t4 = v_q_b * pb_full * pa_full + (b_grid + a_grid) * pb_full * pa_full
        
        J = t1 + t2 + t3 + t4 - cost_b
        
        # Find max
        J_flat = J.reshape(n_states, -1)
        max_idx = np.argmax(J_flat, axis=1)
        i = max_idx // n_grid
        j = max_idx % n_grid
        
        opt_b = grid[i]
        opt_a = grid[j]
        V_curr = J_flat[np.arange(n_states), max_idx]
        
        # Fit quadratic to V_curr
        # V_curr = 0.5 c q^2 + d q + e
        # Least squares
        A_mat = np.column_stack([0.5 * q_vals**2, q_vals, np.ones(n_states)])
        coeffs, _, _, _ = np.linalg.lstsq(A_mat, V_curr, rcond=None)
        c_next = coeffs[0]
        d_next = coeffs[1]
        e_next = coeffs[2]
        
        opt_bid[t] = opt_b
        opt_ask[t] = opt_a
        c_coeff[t] = c_next
        d_coeff[t] = d_next
        e_coeff[t] = e_next
        
    return opt_bid, opt_ask

def quote(t, inventory, mid, params):
    key = (params['sigma'], params['A'], params['k'], params['gamma'], params['T'], params['steps'], params['Q'])
    
    if key not in _policy_cache:
        bid_depths, ask_depths = _compute_optimal_policy(params)
        _policy_cache[key] = (bid_depths, ask_depths)
    else:
        bid_depths, ask_depths = _policy_cache[key]
    
    n = int(np.round(t * params['steps']))
    if n >= params['steps']:
        n = params['steps'] - 1
        
    indices = inventory + params['Q']
    
    bids = bid_depths[n, indices]
    asks = ask_depths[n, indices]
    
    return bids, asks

# EVOLVE-BLOCK-END
