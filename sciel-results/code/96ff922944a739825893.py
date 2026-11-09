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
        self.deadline = time.time() + 285  # Safety margin before 300s limit

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        
        # Wrap func to count calls and check deadline
        def wrapped_func(x):
            if time.time() > self.deadline:
                raise TimeoutError()
            if self.calls >= budget:
                raise ValueError("Budget exceeded")
            self.calls += 1
            x = np.asarray(x, dtype=float)
            x = np.clip(x, -5, 5)
            val = func(x)
            if np.isnan(val) or np.isinf(val):
                val = 1e10
            if val < self.best_f:
                self.best_f = val
                self.best_x = x.copy()
            return val

        # Generate diverse starting points
        rng = np.random.default_rng(42)
        starts = []
        # 1. Origin
        starts.append(np.zeros(dim))
        # 2-7. Random points in a slightly smaller box
        for _ in range(6):
            starts.append(rng.uniform(-4.5, 4.5, dim))

        bounds = [(-5.0, 5.0)] * dim

        # Phase 1: L-BFGS-B from multiple starts
        for start in starts:
            if self.calls >= budget or time.time() > self.deadline:
                break
            
            try:
                res_lbfgs = minimize(
                    wrapped_func, 
                    start, 
                    method='L-BFGS-B', 
                    bounds=bounds, 
                    options={'maxiter': 100, 'gtol': 1e-8, 'ftol': 1e-12}
                )
                # Update best if this run found something better
                if res_lbfgs.fun < self.best_f:
                    self.best_f = res_lbfgs.fun
                    self.best_x = res_lbfgs.x.copy()
            except (TimeoutError, ValueError):
                break
            except Exception:
                continue

        # Phase 2: Nelder-Mead polishing from best point found so far
        if self.calls < budget - 10 and time.time() < self.deadline and self.best_x is not None:
            try:
                res_nm = minimize(
                    wrapped_func, 
                    self.best_x, 
                    method='Nelder-Mead', 
                    options={'maxiter': 50, 'xatol': 1e-8, 'fatol': 1e-12, 'adaptive': True}
                )
                if res_nm.fun < self.best_f:
                    self.best_f = res_nm.fun
                    self.best_x = res_nm.x.copy()
            except (TimeoutError, ValueError):
                pass
            except Exception:
                pass

        # Phase 3: If we still have budget, do a few more L-BFGS-B runs from perturbed best
        if self.calls < budget - 10 and time.time() < self.deadline and self.best_x is not None:
            for _ in range(3):
                if self.calls >= budget or time.time() > self.deadline:
                    break
                perturbation = rng.normal(0, 0.5, dim)
                new_start = np.clip(self.best_x + perturbation, -5, 5)
                try:
                    res_lbfgs = minimize(
                        wrapped_func, 
                        new_start, 
                        method='L-BFGS-B', 
                        bounds=bounds, 
                        options={'maxiter': 50, 'gtol': 1e-8, 'ftol': 1e-12}
                    )
                    if res_lbfgs.fun < self.best_f:
                        self.best_f = res_lbfgs.fun
                        self.best_x = res_lbfgs.x.copy()
                except (TimeoutError, ValueError):
                    break
                except Exception:
                    continue

        # If no valid result was found, evaluate origin
        if self.best_x is None:
             self.best_x = np.zeros(dim)
             try:
                 self.best_f = wrapped_func(self.best_x)
             except:
                 self.best_f = 1e10

        return self.best_x, self.best_f
