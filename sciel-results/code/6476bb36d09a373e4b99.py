import numpy as np
from scipy.optimize import minimize
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.call_count = 0
        self.start_time = time.time()
        self.time_limit = 300.0  # Safety margin for 330s limit

    def __call__(self, func):
        self.call_count = 0
        self.start_time = time.time()
        
        def wrapped_func(x):
            self.call_count += 1
            if time.time() - self.start_time > self.time_limit:
                raise TimeoutError
            x = np.clip(x, -5.0, 5.0)
            return func(x)

        n = self.dim
        budget = self.budget
        bounds = [(-5.0, 5.0) for _ in range(n)]
        
        # Phase 1: Latin Hypercube Sampling for global exploration
        # Use a fixed number of initial points, e.g., 20
        n_init = 20
        rng = np.random.default_rng(42)
        
        # Generate Latin Hypercube samples
        # For each dimension, divide [0, 1] into n_init intervals and pick one point in each
        X_lh = np.zeros((n_init, n))
        for i in range(n_init):
            for j in range(n):
                X_lh[i, j] = rng.uniform(i / n_init, (i + 1) / n_init)
        # Map to [-5, 5]
        X_lh = X_lh * 10.0 - 5.0
        
        best_x = None
        best_f = float('inf')
        candidates = []
        
        # Evaluate initial points
        for x in X_lh:
            if self.call_count >= budget:
                break
            f = wrapped_func(x)
            candidates.append((f, x.copy()))
            if f < best_f:
                best_f = f
                best_x = x.copy()
        
        # Sort candidates by fitness
        candidates.sort(key=lambda t: t[0])
        
        # Phase 2: Local Optimization on top candidates
        # Use L-BFGS-B on the top 3 candidates
        remaining = budget - self.call_count
        calls_per_opt = max(10, remaining // 3)
        
        for i, (f0, x0) in enumerate(candidates[:3]):
            if self.call_count >= budget:
                break
            if time.time() - self.start_time > self.time_limit:
                break
                
            try:
                res = minimize(wrapped_func, x0, method='L-BFGS-B', bounds=bounds,
                               options={'maxfun': calls_per_opt, 'maxiter': 50})
                if res.fun < best_f:
                    best_f = res.fun
                    best_x = res.x.copy()
            except Exception:
                pass
        
        # Phase 3: Final Polish on the best found
        if self.call_count < budget - 10 and best_x is not None:
            try:
                res = minimize(wrapped_func, best_x, method='L-BFGS-B', bounds=bounds,
                               options={'maxfun': budget - self.call_count - 5, 'maxiter': 100})
                if res.fun < best_f:
                    best_f = res.fun
                    best_x = res.x.copy()
            except Exception:
                pass

        return best_x, best_f
