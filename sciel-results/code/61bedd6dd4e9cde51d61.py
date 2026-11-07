import numpy as np
import time
from scipy.optimize import minimize

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0
        d = self.dim
        budget = self.budget
        
        # Track calls strictly
        calls = [0]
        
        def evaluate(x):
            if calls[0] >= budget:
                return None
            calls[0] += 1
            val = func(x)
            return val

        rng = np.random.default_rng()
        
        # Phase 1: Random sampling to get a good starting point
        n_random = min(50, budget)
        best_x = None
        best_f = float('inf')
        
        for _ in range(n_random):
            if time.time() > deadline or calls[0] >= budget:
                break
            x = rng.uniform(-5, 5, d)
            f = evaluate(x)
            if f is not None and f < best_f:
                best_f = f
                best_x = x.copy()
                
        if best_x is None:
            best_x = np.zeros(d)
            f = evaluate(best_x)
            if f is not None:
                best_f = f
            else:
                best_f = 0.0

        # Phase 2: L-BFGS-B with bounds
        # L-BFGS-B is efficient for smooth, bound-constrained problems and handles
        # anisotropic landscapes well by approximating the Hessian.
        
        def bounded_func(x):
            x_clipped = np.clip(x, -5.0, 5.0)
            f_val = evaluate(x_clipped)
            if f_val is None:
                return 1e10
            return f_val

        bounds = [(-5.0, 5.0) for _ in range(d)]
        
        try:
            res = minimize(
                bounded_func,
                best_x,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 1000, 'maxfev': budget - calls[0]}
            )
            if res.x is not None:
                best_x = np.clip(res.x, -5.0, 5.0)
                best_f = bounded_func(best_x)
                if best_f > 1e9:
                    best_f = float('inf')
        except Exception:
            pass

        # Phase 3: If budget remains, try Powell for robustness
        if calls[0] < budget and time.time() < deadline:
            try:
                res2 = minimize(
                    bounded_func,
                    best_x,
                    method='Powell',
                    options={'maxiter': 5000, 'maxfev': budget - calls[0]}
                )
                if res2.x is not None:
                    best_x = np.clip(res2.x, -5.0, 5.0)
                    best_f = bounded_func(best_x)
                    if best_f > 1e9:
                        best_f = float('inf')
            except Exception:
                pass

        return best_x, best_f
