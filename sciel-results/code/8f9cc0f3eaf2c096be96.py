import numpy as np
import time

def angles(n, edges, p):
    """
    Design graph-dependent two-layer QAOA angles.
    
    Strategy:
    1. Precompute the cut values for all 2^n basis states.
    2. Since p=2 is fixed and small, and n <= 10, we can afford a dense grid search 
       over the parameter space (gamma1, beta1, gamma2, beta2) in [0, pi]^4.
       The parameter space is periodic and symmetric, so we can restrict to [0, pi].
       For each parameter set, we compute the expected cut value exactly.
    3. To speed up the evaluation of the expectation value for a given parameter set,
       we use vectorized numpy operations. The state evolution is:
       |psi> = U_B(beta2) U_C(gamma2) U_B(beta1) U_C(gamma1) |+>^n
       where U_C(g) = exp(-i g C) is diagonal and U_B(b) = exp(-i b sum X_i) is a 
       tensor product of single-qubit rotations.
       
       Instead of simulating the state vector step-by-step, we can precompute 
       the action of the unitaries or use a highly optimized loop. 
       However, a full grid search over [0, pi]^4 with fine resolution might be 
       too slow if done naively. 
       
       Alternative: Use a coarse grid to identify promising regions, then refine.
       Or, exploit the fact that for p=2, the function is smooth and we can use 
       a multi-start local optimization with a very fast gradient-free method 
       like Nelder-Mead, but with a much better initial guess from a coarse grid.
       
       Let's stick to the directive: "tensor network contraction" was the prompt's 
       suggested *different* strategy, but the directive says "make a structural 
       change". The current program uses L-BFGS with parameter-shift gradients. 
       A structural change is to use a **global optimization** method like 
       Differential Evolution or a **dense grid search** followed by local refinement.
       
       Given the 120s limit and n<=10, a dense grid search over 16x16x16x16 = 65536 
       points is feasible if each evaluation is fast (~1ms). 65536 * 1ms = 65s. 
       This is risky. Let's use a coarser grid (8x8x8x8 = 4096 points) to find 
       the best region, then run a few local optimizations from the best grid points.
       
       Actually, the current program's score is 0.9643. The inspirations suggest 
       that precomputing unitaries or using global search helps. 
       
       Let's implement a **coarse grid search** to find the global optimum candidate, 
       then **Nelder-Mead** refinement from the top few grid points. This is 
       structurally different from the current L-BFGS + random restarts.
    """
    start_time = time.time()
    deadline = start_time + 115  # 5s safety margin

    # 1. Precompute cut values
    max_states = 1 << n
    cut_vals = np.zeros(max_states, dtype=np.float64)
    for s in range(max_states):
        c = 0
        for u, v in edges:
            if ((s >> u) ^ (s >> v)) & 1:
                c += 1
        cut_vals[s] = c
        
    opt_cut = np.max(cut_vals)
    if opt_cut == 0:
        return np.zeros((p, 2))

    # Precompute the diagonal of U_C for various gammas? No, gamma is continuous.
    # But we can precompute the "base" state |+>^n and the structure of U_B.
    
    # Fast simulation function for a single parameter set
    def evaluate_params(g1, b1, g2, b2):
        # State starts as |+>^n
        # We can represent the state as a 1D array of size 2^n
        # Initial state: all 1/sqrt(2^n)
        # Apply U_C(g1): multiply by exp(-i g1 cut_vals)
        # Apply U_B(b1): tensor product of Rx(b1)
        # Apply U_C(g2): multiply by exp(-i g2 cut_vals)
        # Apply U_B(b2): tensor product of Rx(b2)
        
        # To speed this up, we can precompute the rotation matrices for U_B
        # U_B(b) = (cos(b/2) I - i sin(b/2) X)^{⊗n}
        # This is a dense matrix of size 2^n x 2^n, which is too big for n=10 (1024x1024 is ok, 2048x2048 is ok, but applying it is O(4^n))
        
        # Better: Apply the rotations qubit by qubit using reshape and moveaxis, 
        # but do it in a tight loop with minimal overhead.
        
        state = np.ones(max_states, dtype=np.complex128)
        state /= np.sqrt(max_states)
        
        # U_C(g1)
        state *= np.exp(-1j * g1 * cut_vals)
        
        # U_B(b1)
        c_b = np.cos(b1)
        s_b = -1j * np.sin(b1)
        t = state.reshape([2] * n)
        for i in range(n):
            t = np.moveaxis(t, i, 0)
            t0 = t[0]
            t1 = t[1]
            new_t0 = c_b * t0 + s_b * t1
            new_t1 = s_b * t0 + c_b * t1
            t_new = np.empty_like(t)
            t_new[0] = new_t0
            t_new[1] = new_t1
            t = t_new
            t = np.moveaxis(t, 0, i)
        state = t.reshape(max_states)
        
        # U_C(g2)
        state *= np.exp(-1j * g2 * cut_vals)
        
        # U_B(b2)
        c_b = np.cos(b2)
        s_b = -1j * np.sin(b2)
        t = state.reshape([2] * n)
        for i in range(n):
            t = np.moveaxis(t, i, 0)
            t0 = t[0]
            t1 = t[1]
            new_t0 = c_b * t0 + s_b * t1
            new_t1 = s_b * t0 + c_b * t1
            t_new = np.empty_like(t)
            t_new[0] = new_t0
            t_new[1] = new_t1
            t = t_new
            t = np.moveaxis(t, 0, i)
        state = t.reshape(max_states)
        
        probs = np.abs(state) ** 2
        return np.sum(probs * cut_vals) / opt_cut

    # Coarse Grid Search
    # Parameters are in [0, pi]
    # Let's use 10 points per dimension -> 10^4 = 10,000 evaluations.
    # Each evaluation is ~1ms for n=10. 10s total. Feasible.
    
    grid_size = 10
    grid_vals = np.linspace(0, np.pi, grid_size)
    
    best_score = -1
    best_params = None
    
    # Iterate over grid
    for g1 in grid_vals:
        for b1 in grid_vals:
            for g2 in grid_vals:
                for b2 in grid_vals:
                    if time.time() > deadline - 20:
                        break
                    score = evaluate_params(g1, b1, g2, b2)
                    if score > best_score:
                        best_score = score
                        best_params = np.array([g1, b1, g2, b2])
                if time.time() > deadline - 20:
                    break
            if time.time() > deadline - 20:
                break
        if time.time() > deadline - 20:
            break

    # Refine with Nelder-Mead from the best grid point
    if best_params is not None and time.time() < deadline - 5:
        from scipy.optimize import minimize
        
        def obj(params):
            g1, b1, g2, b2 = params
            # Keep in [0, pi] by wrapping or just letting it be, but QAOA is periodic
            # Actually, the function is periodic with period 2pi, but symmetric in [0, pi]
            # Let's just evaluate directly.
            return -evaluate_params(g1, b1, g2, b2)
        
        try:
            res = minimize(obj, best_params, method='Nelder-Mead', 
                           options={'maxiter': 200, 'xatol': 1e-6, 'fatol': 1e-6})
            if -res.fun > best_score:
                best_score = -res.fun
                best_params = res.x
        except Exception:
            pass

    # Format output: (p, 2) array, rows (gamma, beta)
    # best_params is [g1, b1, g2, b2]
    g1, b1, g2, b2 = best_params
    return np.array([[g1, b1], [g2, b2]])
