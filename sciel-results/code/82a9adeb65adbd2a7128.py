import numpy as np
from scipy.optimize import minimize
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.calls = 0
        self.best_x = None
        self.best_f = np.inf
        self.deadline = time.time() + 280  # Safety margin before 300s limit

    def __call__(self, func):
        # Wrap func to count calls and check deadline
        def wrapped_func(x):
            if time.time() > self.deadline:
                raise TimeoutError()
            if self.calls >= self.budget:
                raise ValueError("Budget exceeded")
            self.calls += 1
            x = np.clip(x, -5, 5)
            val = func(x)
            if val < self.best_f:
                self.best_f = val
                self.best_x = x.copy()
            return val

        # Generate diverse starting points
        rng = np.random.default_rng(42)
        starts = []
        # 1. Origin
        starts.append(np.zeros(self.dim))
        # 2-6. Random points in a slightly smaller box to avoid boundary issues initially
        for _ in range(5):
            starts.append(rng.uniform(-4, 4, self.dim))
        
        bounds = [(-5.0, 5.0)] * self.dim

        for start in starts:
            if self.calls >= self.budget or time.time() > self.deadline:
                break
            
            # 1. L-BFGS-B for fast convergence
            try:
                res_lbfgs = minimize(
                    wrapped_func, 
                    start, 
                    method='L-BFGS-B', 
                    bounds=bounds, 
                    options={'maxiter': 50, 'gtol': 1e-5}
                )
            except (TimeoutError, ValueError):
                break
            except Exception:
                continue

            x_candidate = res_lbfgs.x
            f_candidate = self.best_f # wrapped_func updates this
            
            # 2. Nelder-Mead for polishing (derivative-free, good for final steps)
            if self.calls < self.budget - 20 and time.time() < self.deadline:
                try:
                    res_nm = minimize(
                        wrapped_func, 
                        x_candidate, 
                        method='Nelder-Mead', 
                        options={'maxiter': 20, 'xatol': 1e-6, 'fatol': 1e-6}
                    )
                except (TimeoutError, ValueError):
                    break
                except Exception:
                    pass

        # If no valid result was found (e.g. all exceptions), return a safe point
        if self.best_x is None:
             self.best_x = np.zeros(self.dim)
             self.best_f = wrapped_func(self.best_x)

        return self.best_x, self.best_f
