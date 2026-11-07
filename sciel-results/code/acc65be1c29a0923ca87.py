import numpy as np
import time
from scipy.optimize import minimize

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
                dim = self.dim
                budget = self.budget
                calls = 0
                best_x = None
                best_f = float('inf')
                start_time = time.time()
                deadline = start_time + 290  # Safety margin

                def eval_func(x):
                    nonlocal calls, best_x, best_f
                    calls += 1
                    # Clip to bounds just in case, though SciPy handles bounds for L-BFGS-B
                    # But NM might step out? NM doesn't use bounds directly in the same way, 
                    # but we can clip in the wrapper to be safe.
                    x_clipped = np.clip(x, -5, 5)
                    fx = func(x_clipped)
                    if fx < best_f:
                        best_f = fx
                        best_x = x_clipped.copy()
                    return fx

                # Phase 1: Multi-start L-BFGS-B
                # 10 starts, 20 calls each = 200 calls
                n_starts = 10
                calls_per_start = 20
                rng = np.random.default_rng(42)
                
                # Generate start points
                # Mix of random and structured
                starts = []
                # Center
                starts.append(np.zeros(dim))
                # Random
                for _ in range(n_starts - 1):
                    starts.append(rng.uniform(-4, 4, dim))

                for x0 in starts:
                    if calls >= budget - 50: # Keep 50 for phase 2
                        break
                    if time.time() > deadline:
                        break
                    
                    # L-BFGS-B is good for smooth functions
                    try:
                        res = minimize(eval_func, x0, method='L-BFGS-B', 
                                       bounds=[(-5, 5)] * dim,
                                       options={'maxiter': 100, 'maxfun': calls_per_start})
                    except Exception:
                        pass

                # Phase 2: Polish with Nelder-Mead from best found
                if best_x is not None:
                    remaining = budget - calls
                    if remaining > 10:
                        if time.time() < deadline:
                            try:
                                res = minimize(eval_func, best_x, method='Nelder-Mead',
                                               options={'maxiter': 1000, 'maxfun': remaining, 'xatol': 1e-8, 'fatol': 1e-8})
                            except Exception:
                                pass

                return best_x, best_f
