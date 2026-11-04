import numpy as np
from scipy.optimize import minimize, differential_evolution
import time

N = 600
DELTA = (1.0 / 2.0) / N  # interval width = 1/1200

def _compute_score(h):
    """
    Compute the normalized maximum of the autoconvolution.
    Score = max(f*f) / (integral f)^2
    """
    h = np.asarray(h, dtype=float)
    if np.any(h < -1e-9):
        return 1e10
    h = np.maximum(h, 0)
    
    integral = np.sum(h) * DELTA
    if integral <= 1e-10:
        return 1e10
        
    # For non-negative step functions, the autoconvolution maximum occurs at t=0
    # (f*f)(0) = sum(h^2) * DELTA
    max_ac = np.sum(h**2) * DELTA
    
    return max_ac / (integral ** 2)

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin for 150s limit
    
    # Strategy: 
    # 1. Use a parameterized superposition of rectangular pulses (disjoint or overlapping) 
    #    to represent the step function. This allows for sparse, high-contrast structures 
    #    which can minimize peak autoconvolution relative to total energy.
    # 2. Optimize the parameters (positions, widths, heights) using Differential Evolution 
    #    for global exploration.
    # 3. Polish the result using L-BFGS-B on the direct 600-dimensional height vector.
    
    # Precompute bin centers
    x = np.linspace(-0.25, 0.25, N, endpoint=False) + 0.5 * DELTA
    
    # Helper to convert rectangle parameters to step heights
    def rects_to_h(params, n_rects):
        """
        params: [mu1, w1, a1, mu2, w2, a2, ..., base]
        h(x) = base + sum_i a_i * 1_{|x - mu_i| < w_i/2}
        """
        h = np.full(N, params[-1], dtype=float) # base
        for i in range(n_rects):
            mu = params[3*i]
            w = abs(params[3*i + 1]) + 1e-6
            a = params[3*i + 2]
            
            # Calculate bin indices
            left = mu - w/2
            right = mu + w/2
            
            # Find indices where x is in [left, right]
            mask = (x >= left) & (x <= right)
            h[mask] += a
            
        return h
    
    # Objective for rectangle parameterization
    def objective_rects(params, n_rects):
        h = rects_to_h(params, n_rects)
        # Penalty for negative heights
        neg = h[h < 0]
        if len(neg) > 0:
            penalty = 100.0 * np.sum(neg**2)
        else:
            penalty = 0.0
        return _compute_score(h) + penalty
    
    # Start with uniform
    best_h = np.ones(N)
    best_score = _compute_score(best_h)
    
    rng = np.random.default_rng(42)
    
    # Phase 1: Global search with Differential Evolution on Rectangle parameters
    # Try different numbers of rectangles
    for n_rects in [3, 5, 8]:
        if time.time() > deadline - 30.0:
            break
            
        n_params = 3 * n_rects + 1 # base
        bounds = []
        for _ in range(n_rects):
            bounds.append((-0.25, 0.25)) # mu
            bounds.append((0.01, 0.25))  # width
            bounds.append((-1.0, 5.0))   # amplitude
        bounds.append((0.0, 2.0))        # base
        
        # Differential Evolution
        try:
            # Use 'differential_evolution' from scipy.optimize
            # maxiter needs to be small to fit in time budget, or use 'tol'
            # DE is stochastic, so we might need multiple runs or a longer run.
            # Given time constraints, let's try a moderate number of generations.
            res = differential_evolution(
                objective_rects, bounds, args=(n_rects,),
                seed=42, maxiter=50, tol=1e-5, 
                popsize=15, 
                init='latinhypercube',
                polish=False # We will polish later
            )
            
            if res.fun < best_score:
                # Reconstruct h
                h_new = rects_to_h(res.x, n_rects)
                h_new = np.maximum(h_new, 0)
                score_new = _compute_score(h_new)
                if score_new < best_score:
                    best_score = score_new
                    best_h = h_new
        except Exception:
            pass

    # Phase 2: Local Polish with L-BFGS-B on direct heights
    # Start from the best h found so far
    if time.time() < deadline - 10.0:
        h_start = best_h.copy()
        
        def objective_direct(h_vec):
            if np.any(h_vec < 0):
                return 1e10
            return _compute_score(h_vec)
        
        try:
            # L-BFGS-B on 600 variables
            # Finite difference gradient will be expensive, but 600 * ~1ms = 0.6s per gradient
            # 20 iterations -> ~12s. Feasible.
            res_direct = minimize(
                objective_direct, h_start, method='L-BFGS-B',
                bounds=[(0, None)] * N,
                options={'maxiter': 50, 'ftol': 1e-8, 'gtol': 1e-5}
            )
            if res_direct.fun < best_score:
                best_score = res_direct.fun
                best_h = res_direct.x
        except Exception:
            pass

    # Final check
    best_h = np.maximum(best_h, 0)
    if np.sum(best_h) < 1e-10:
        best_h = np.ones(N)
        
    return list(best_h)
