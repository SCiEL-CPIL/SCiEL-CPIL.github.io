import numpy as np
import time

class Optimizer:
    """Optimizer for BBOB-style functions using Random Search + L-BFGS-B."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280  # Safety margin
        
        n = self.dim
        calls = 0
        best_f = float('inf')
        best_x = np.zeros(n)
        
        def eval_point(x):
            nonlocal calls, best_f, best_x
            if calls >= self.budget or time.time() > deadline:
                return None
            val = func(x)
            calls += 1
            if val < best_f:
                best_f = val
                best_x = x.copy()
            return val

        # Phase 1: Random Search with Center Bias
        # We use 50% of the budget for random search to find a good starting point.
        # We bias the search towards the center (0,0,...,0) because the minimizer is in [-4,4]^5.
        num_random = self.budget // 2
        
        for i in range(num_random):
            if time.time() > deadline:
                break
            
            # Sample from a Gaussian centered at 0 with std 2.0
            # This covers [-4, 4] well (2 sigma = 4)
            x = np.random.randn(n) * 2.0
            x = np.clip(x, -5, 5)
            
            eval_point(x)
            
            # Occasionally sample from a wider distribution to ensure coverage
            if i % 10 == 0:
                x = np.random.uniform(-5, 5, n)
                eval_point(x)

        # Phase 2: L-BFGS-B Optimization
        # Use the remaining budget for gradient-based optimization starting from the best point.
        # L-BFGS-B is efficient for smooth functions like Sphere.
        
        if calls < self.budget and best_f < float('inf'):
            # Wrap func to track calls and enforce deadline
            def wrapped_func(x):
                nonlocal calls
                if calls >= self.budget or time.time() > deadline:
                    return best_f  # Return current best to stop optimization
                val = func(x)
                calls += 1
                if val < best_f:
                    best_f = val
                    best_x = x.copy()
                return val
            
            # Use scipy.optimize.minimize
            from scipy.optimize import minimize
            
            # Try multiple starts? No, we only have one best point.
            # But we can try to restart if it fails or gets stuck.
            # Let's just run it once with a tight maxiter.
            # maxiter is not strictly function calls, but we monitor calls.
            
            try:
                res = minimize(
                    wrapped_func, 
                    best_x, 
                    method='L-BFGS-B', 
                    bounds=[(-5, 5)] * n,
                    options={'maxiter': 100, 'ftol': 1e-12, 'gtol': 1e-8}
                )
            except Exception:
                pass

        return best_x, best_f
