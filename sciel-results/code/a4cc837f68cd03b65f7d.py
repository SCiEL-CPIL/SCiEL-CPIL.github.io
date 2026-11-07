import numpy as np
import time
import itertools
from scipy.optimize import minimize

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.start_time = None
        self.call_count = 0

    def __call__(self, func):
        self.start_time = time.time()
        self.call_count = 0
        best_x = None
        best_f = float('inf')
        second_best_x = None
        second_best_f = float('inf')

        def safe_func(x):
            return func(np.asarray(x))

        # Phase 1: Coarse-to-Fine Grid Search
        levels = [-4.0, 0.0, 4.0]
        grid_points = list(itertools.product(levels, repeat=self.dim))
        
        origin = np.zeros(self.dim)
        grid_points.sort(key=lambda p: np.linalg.norm(np.array(p) - origin))

        grid_results = []
        for p in grid_points:
            if time.time() - self.start_time > 320 or self.call_count >= self.budget:
                break
            
            x = np.array(p)
            try:
                fx = safe_func(x)
                self.call_count += 1
                grid_results.append((fx, x))
                if fx < best_f:
                    best_f = fx
                    best_x = x
            except Exception:
                break

        if not grid_results:
            return best_x, best_f

        # Sort grid results to find top candidates
        grid_results.sort(key=lambda t: t[0])
        
        # Phase 2: Local Optimization (Nelder-Mead)
        # Run from the best point
        if best_x is not None and self.budget - self.call_count > 10:
            try:
                def wrapper(x):
                    if time.time() - self.start_time > 320:
                        raise TimeoutError("Time limit")
                    if self.call_count >= self.budget:
                        raise ValueError("Budget limit")
                    self.call_count += 1
                    return func(np.asarray(x))

                res = minimize(
                    wrapper, 
                    best_x, 
                    method='Nelder-Mead', 
                    options={'maxiter': 200, 'xatol': 1e-6, 'fatol': 1e-8}
                )
                
                if res.fun < best_f:
                    best_f = res.fun
                    best_x = res.x

            except Exception:
                pass

        # Phase 3: Try a second candidate if budget allows
        # Find the best grid point that is far from the current best_x
        if self.budget - self.call_count > 20 and best_x is not None:
            # Find candidate in grid_results that is not the best one and has good score
            # We want a point that is far from best_x to explore a different basin
            candidates = [item for item in grid_results if not np.allclose(item[1], best_x, atol=1e-5)]
            if candidates:
                # Pick the one with the lowest function value among those far away
                # Or just the second best overall?
                # Let's pick the one with the best function value that is at least some distance away
                # To keep it simple, let's just pick the second best grid point
                if len(grid_results) > 1:
                    second_fx, second_x = grid_results[1]
                    # Check distance
                    dist = np.linalg.norm(second_x - best_x)
                    if dist > 1.0: # If it's reasonably far
                        try:
                            def wrapper2(x):
                                if time.time() - self.start_time > 320:
                                    raise TimeoutError("Time limit")
                                if self.call_count >= self.budget:
                                    raise ValueError("Budget limit")
                                self.call_count += 1
                                return func(np.asarray(x))

                            res2 = minimize(
                                wrapper2, 
                                second_x, 
                                method='Nelder-Mead', 
                                options={'maxiter': 100, 'xatol': 1e-6, 'fatol': 1e-8}
                            )
                            
                            if res2.fun < best_f:
                                best_f = res2.fun
                                best_x = res2.x
                        except Exception:
                            pass

        return best_x, best_f
