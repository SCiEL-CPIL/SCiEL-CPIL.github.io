import numpy as np
import time
import scipy.optimize as opt

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280  # 20s margin
        calls = 0
        best_x = None
        best_f = np.inf

        def safe_func(x):
            nonlocal calls
            if time.time() > deadline:
                raise TimeoutError
            if calls >= self.budget:
                raise StopIteration
            calls += 1
            # Ensure x is 1D array of length dim
            x = np.asarray(x).flatten()
            if len(x) != self.dim:
                # Fallback if shape is wrong, though Nelder-Mead should maintain shape
                x = x[:self.dim]
            # Clip to bounds
            x_clipped = np.clip(x, -5, 5)
            val = func(x_clipped)
            return val

        # Phase 1: Global exploration with multiple cheap local optimizations
        n_starts = 10
        max_iter_small = 5
        
        # Use a fixed seed for reproducibility in the "black box" context if needed, 
        # but random is fine. Let's use np.random.default_rng
        rng = np.random.default_rng(42)
        
        start_points = []
        for _ in range(n_starts):
            x0 = rng.uniform(-5, 5, self.dim)
            start_points.append(x0)
            
        # Also include the origin and corners? Maybe just random is enough.
        # Let's add a few structured points
        start_points.append(np.zeros(self.dim))
        start_points.append(np.ones(self.dim) * 4)
        start_points.append(np.ones(self.dim) * -4)
        
        # Limit to n_starts if we added too many
        start_points = start_points[:n_starts]

        for x0 in start_points:
            if calls >= self.budget or time.time() > deadline:
                break
            try:
                res = opt.minimize(
                    safe_func, 
                    x0, 
                    method='Nelder-Mead', 
                    options={'maxiter': max_iter_small, 'xatol': 1e-4, 'fatol': 1e-4}
                )
                if res.fun < best_f:
                    best_f = res.fun
                    best_x = res.x
            except (TimeoutError, StopIteration):
                break
            except Exception:
                pass

        # Phase 2: Local exploitation from the best point found
        if best_x is not None and calls < self.budget and time.time() < deadline:
            try:
                # Use remaining budget
                remaining_budget = self.budget - calls
                # Estimate maxiter for Nelder-Mead: each iter is dim+1 calls
                est_maxiter = max(1, remaining_budget // (self.dim + 1))
                
                res = opt.minimize(
                    safe_func, 
                    best_x, 
                    method='Nelder-Mead', 
                    options={'maxiter': est_maxiter, 'xatol': 1e-8, 'fatol': 1e-8}
                )
                if res.fun < best_f:
                    best_f = res.fun
                    best_x = res.x
            except (TimeoutError, StopIteration):
                pass
            except Exception:
                pass

        # If we haven't made any calls (exception before first call), return something
        if best_x is None:
            best_x = np.zeros(self.dim)
            best_f = func(best_x)
            
        return best_x, best_f
