import numpy as np
import time
import scipy.optimize

class Optimizer:
    """Hybrid Powell + Nelder-Mead Optimizer for Rosenbrock-like functions.
    
    Uses Powell's method for its valley-alignment capability on Rosenbrock,
    followed by Nelder-Mead for local polishing. Multiple starts with diverse
    initializations to handle the hidden shift/rotation/offset.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.best_x = None
        self.best_f = np.inf
        self.calls_made = 0
        self.start_time = 0.0
        self.deadline = 0.0

    def __call__(self, func):
        self.start_time = time.time()
        self.deadline = self.start_time + 285.0

        def wrapped_func(x):
            if self.calls_made >= self.budget:
                raise StopIteration("Budget exhausted")
            if time.time() > self.deadline:
                raise StopIteration("Time limit exceeded")
            x = np.clip(x, -5, 5)
            self.calls_made += 1
            val = func(x)
            if val < self.best_f:
                self.best_f = val
                self.best_x = x.copy()
            return val

        rng = np.random.default_rng(42)
        
        # Diverse starting points covering the likely region
        starts = []
        starts.append(np.ones(self.dim))
        starts.append(np.zeros(self.dim))
        starts.append(np.full(self.dim, 2.0))
        starts.append(np.full(self.dim, -1.0))
        starts.append(np.full(self.dim, 3.0))
        starts.append(np.full(self.dim, -2.0))
        starts.append(np.full(self.dim, 4.0))
        
        # Random in likely region [-2, 4]
        for _ in range(5):
            starts.append(rng.uniform(-2, 4, self.dim))
            
        # Random in full box
        for _ in range(3):
            starts.append(rng.uniform(-5, 5, self.dim))

        # Remove duplicates
        unique_starts = []
        for s in starts:
            is_dup = False
            for u in unique_starts:
                if np.allclose(s, u, atol=1e-5):
                    is_dup = True
                    break
            if not is_dup:
                unique_starts.append(s)

        # Phase 1: Powell for valley alignment, Phase 2: Nelder-Mead for polishing
        for x0 in unique_starts:
            if self.calls_made >= self.budget:
                break
            if time.time() > self.deadline:
                break
            
            x0 = np.clip(x0, -5, 5)
            remaining_budget = self.budget - self.calls_made
            if remaining_budget < 30:
                break

            # Phase 1: Powell
            try:
                res_powell = scipy.optimize.minimize(
                    wrapped_func, 
                    x0, 
                    method='Powell',
                    options={
                        'maxiter': 500,
                        'ftol': 1e-6,
                        'xtol': 1e-6,
                    }
                )
                
                if res_powell.fun < self.best_f:
                    self.best_f = res_powell.fun
                    self.best_x = np.clip(res_powell.x, -5, 5)
                    
            except StopIteration:
                break
            except Exception:
                pass

            if self.calls_made >= self.budget:
                break
            if time.time() > self.deadline:
                break
            
            remaining_budget = self.budget - self.calls_made
            if remaining_budget < 15:
                break

            # Phase 2: Nelder-Mead polish from Powell's result
            x_start = self.best_x.copy() if self.best_x is not None else x0
            x_start = np.clip(x_start, -5, 5)
            
            try:
                delta = 0.3
                simplex = [x_start.copy()]
                for i in range(self.dim):
                    point = x_start.copy()
                    point[i] += delta
                    simplex.append(point)
                
                res_nm = scipy.optimize.minimize(
                    wrapped_func, 
                    x_start, 
                    method='Nelder-Mead',
                    options={
                        'initial_simplex': simplex,
                        'xatol': 1e-5,
                        'fatol': 1e-5,
                        'maxiter': 500,
                        'adaptive': True
                    }
                )
                
                if res_nm.fun < self.best_f:
                    self.best_f = res_nm.fun
                    self.best_x = np.clip(res_nm.x, -5, 5)
                    
            except StopIteration:
                break
            except Exception:
                pass

        # Fallback if no calls were made
        if self.best_x is None:
            x0 = np.ones(self.dim)
            self.best_x = x0
            self.best_f = func(np.clip(x0, -5, 5))

        return self.best_x, self.best_f
