import numpy as np
import time
from scipy.optimize import minimize
import warnings
warnings.filterwarnings("ignore")

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        # Safety margin: 330s limit, leave 10s for overhead
        deadline = start_time + 320.0
        
        def time_ok():
            return time.time() < deadline
        
        rng = np.random.default_rng(42)
        
        best_x = np.zeros(self.dim)
        best_fx = np.inf
        call_count = 0

        # Tracked function to count calls and update best
        def tracked_func(x):
            nonlocal call_count, best_x, best_fx
            x = np.asarray(x, dtype=float)
            # Clip to box to ensure validity, though func also clips
            x = np.clip(x, -5.0, 5.0)
            fx = float(func(x))
            call_count += 1
            if fx < best_fx:
                best_fx = fx
                best_x = x.copy()
            return fx

        bounds = [(-5.0, 5.0)] * self.dim
        
        # Phase 1: Global Exploration
        # We need to find the basin of attraction. 
        # The minimizer is in [-4, 4]^5. 
        # Rosenbrock has a known structure, but it is rotated.
        # Origin is a good guess, but we need diverse samples.
        
        initial_points = []
        # 1. Origin
        initial_points.append(np.zeros(self.dim))
        
        # 2. Random points in the likely region
        # 12 random points is cheap and provides diversity
        n_random = 12
        for _ in range(n_random):
            initial_points.append(rng.uniform(-4.0, 4.0, self.dim))
            
        # Evaluate initial points
        for x0 in initial_points:
            if call_count >= self.budget - 20 or not time_ok():
                break
            tracked_func(x0)
        
        # Phase 2: Intensive Local Optimization with L-BFGS-B
        # L-BFGS-B is highly effective for smooth, bounded problems like Rosenbrock.
        # It uses gradient information (finite differences) to converge quickly.
        
        if time_ok() and call_count < self.budget - 10:
            x_start = best_x.copy()
            remaining_budget = self.budget - call_count - 10 # Reserve 10 for polish/overshoot safety
            
            if remaining_budget > 10:
                try:
                    res = minimize(
                        tracked_func,
                        x_start,
                        method='L-BFGS-B',
                        bounds=bounds,
                        options={
                            'maxiter': 1000,
                            'maxfun': remaining_budget,
                            'ftol': 1e-15,
                            'gtol': 1e-12,
                            'maxls': 50
                        }
                    )
                except Exception:
                    pass
        
        # Phase 3: Polish with Nelder-Mead
        # If L-BFGS-B stopped early or we have budget left, NM can refine the solution.
        # NM is robust and doesn't rely on gradient estimates which can be noisy near the optimum.
        
        if time_ok() and call_count < self.budget - 5:
            remaining = self.budget - call_count
            if remaining > 10:
                try:
                    res = minimize(
                        tracked_func,
                        best_x,
                        method='Nelder-Mead',
                        options={
                            'maxiter': 100,
                            'maxfev': remaining,
                            'adaptive': True,
                            'xatol': 1e-10,
                            'fatol': 1e-12
                        }
                    )
                except Exception:
                    pass
        
        return best_x, best_fx
