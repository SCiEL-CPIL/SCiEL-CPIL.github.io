"""Black-box continuous optimizer for BBOB-style f14 (Different powers)."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

class Optimizer:
    """
    Optimizer using a hybrid strategy for BBOB f14:
    1. Fast gradient-based descent (L-BFGS-B) from multiple diverse starts for early progress.
    2. CMA-ES for global robustness and fine-tuning if budget remains.
    
    Key improvements over pure L-BFGS-B:
    - Better diversity in initial points to escape local minima.
    - CMA-ES handles the non-quadratic nature of f14 (different powers) better.
    - Adaptive switching: if L-BFGS-B stagnates, switch to CMA-ES.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        t_start = time.time()
        deadline = t_start + 290.0
        dim = self.dim
        budget = self.budget
        calls = 0
        best_x = np.zeros(dim)
        best_f = np.inf

        def update_best(x, f):
            nonlocal best_x, best_f
            if f < best_f:
                best_x = x.copy()
                best_f = f

        def time_left():
            return deadline - time.time() > 2.0

        def wrapped_func(x):
            nonlocal calls
            if calls >= budget:
                raise ValueError("Budget Exceeded")
            if time.time() > deadline:
                raise ValueError("Time Exceeded")
            calls += 1
            x_clipped = np.clip(x, -5.0, 5.0)
            return func(x_clipped)

        # Generate diverse starting points
        def halton_sequence(n, dims):
            points = []
            primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29]
            for i in range(1, n + 1):
                point = []
                for d in range(dims):
                    f = primes[d]
                    result = 0.0
                    denom = 1
                    ii = i
                    while True:
                        denom *= f
                        result += (ii % f) / denom
                        ii //= f
                        if ii == 0:
                            break
                    point.append(result)
                points.append(point)
            return np.array(points)

        def map_to_box(points):
            return -5.0 + 10.0 * points

        # Phase 1: Fast L-BFGS-B from multiple starts
        n_halton = 8
        halton_points = halton_sequence(n_halton, dim)
        start_points = map_to_box(halton_points)
        
        rng = np.random.default_rng(42)
        random_points = rng.uniform(-5, 5, size=(4, dim))
        start_points = np.vstack([start_points, np.zeros(dim), random_points])
        
        # Shuffle
        indices = rng.permutation(len(start_points))
        start_points = start_points[indices]

        bounds = [(-5.0, 5.0)] * dim
        top_candidates = []

        # Run L-BFGS-B from each start
        for x0 in start_points:
            if calls >= budget or not time_left():
                break
            
            try:
                res = minimize(
                    wrapped_func, 
                    x0, 
                    method='L-BFGS-B', 
                    bounds=bounds,
                    options={'maxiter': 100, 'gtol': 1e-10, 'ftol': 1e-12, 'maxfun': 60}
                )
                
                if res.fun < best_f:
                    update_best(res.x, res.fun)
                
                top_candidates.append((res.fun, res.x.copy()))
                
            except (ValueError, Exception) as e:
                if "Budget Exceeded" in str(e) or "Time Exceeded" in str(e):
                    break
                else:
                    pass

        # Phase 2: CMA-ES refinement if budget remains
        if top_candidates and calls < budget - 50 and time_left():
            top_candidates.sort(key=lambda item: item[0])
            best_cand_f, best_cand_x = top_candidates[0]
            
            # Use CMA-ES for global optimization around the best candidate
            try:
                from cma import CMA
                import cma
                
                # Initialize CMA-ES around the best candidate
                sigma0 = 1.0
                opts = {
                    'maxiter': 200,
                    'verbose': -9,
                    'bounds': [(-5.0, 5.0)] * dim,
                    'seed': 42,
                    'maxfevals': budget - calls
                }
                
                es = cma.CMAEvolutionStrategy(best_cand_x, sigma0, opts)
                
                while not es.stop() and calls < budget and time_left():
                    solutions = es.ask()
                    for sol in solutions:
                        if calls >= budget or not time_left():
                            break
                        try:
                            f = wrapped_func(sol)
                            if f < best_f:
                                update_best(sol, f)
                        except:
                            break
                    es.tell(solutions, [wrapped_func(s) for s in solutions])
                
            except ImportError:
                # Fallback to Nelder-Mead if CMA not available
                try:
                    res = minimize(
                        wrapped_func, 
                        best_cand_x, 
                        method='Nelder-Mead', 
                        bounds=bounds,
                        options={'maxiter': 100, 'xatol': 1e-8, 'fatol': 1e-10, 'maxfev': 100}
                    )
                    
                    if res.fun < best_f:
                        update_best(res.x, res.fun)
                        
                except:
                    pass
            except Exception:
                pass

        # Phase 3: Final L-BFGS-B polish from best candidate
        if calls < budget - 10 and time_left():
            try:
                res = minimize(
                    wrapped_func, 
                    best_x, 
                    method='L-BFGS-B', 
                    bounds=bounds,
                    options={'maxiter': 50, 'gtol': 1e-14, 'ftol': 1e-16, 'maxfun': 30}
                )
                
                if res.fun < best_f:
                    update_best(res.x, res.fun)
                    
            except:
                pass

        return best_x, best_f

# EVOLVE-BLOCK-END
