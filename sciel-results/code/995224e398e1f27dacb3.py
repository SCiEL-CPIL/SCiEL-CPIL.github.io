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
        self.deadline = time.time() + 290  # Safety margin

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        
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

        rng = np.random.default_rng(42)
        
        # Phase 1: Multistart Nelder-Mead
        starts = [np.zeros(dim)]
        for _ in range(9):
            starts.append(rng.uniform(-4, 4, dim))
            
        best_candidate = None
        best_candidate_val = np.inf
        
        for start in starts:
            if self.calls >= budget - 10 or time.time() > self.deadline:
                break
            try:
                res = minimize(wrapped_func, start, method='Nelder-Mead', 
                               options={'maxiter': 30, 'xatol': 1e-6, 'fatol': 1e-8, 'adaptive': True})
                if res.fun < best_candidate_val:
                    best_candidate_val = res.fun
                    best_candidate = res.x
            except (TimeoutError, ValueError):
                break
            except Exception:
                continue
                
        # Phase 2: Powell Polish
        if best_candidate is not None and self.calls < budget - 5 and time.time() < self.deadline:
            try:
                res_powell = minimize(wrapped_func, best_candidate, method='Powell', 
                                      options={'maxiter': 100, 'xtol': 1e-8, 'ftol': 1e-10})
                # Powell might not update best_f in wrapper if it hits budget limit mid-iteration?
                # Wrapper updates best_f on every call.
            except (TimeoutError, ValueError):
                pass
            except Exception:
                pass

        # Fallback
        if self.best_x is None:
             self.best_x = np.zeros(dim)
             try: self.best_f = wrapped_func(self.best_x)
             except: self.best_f = 1e10

        return self.best_x, self.best_f
