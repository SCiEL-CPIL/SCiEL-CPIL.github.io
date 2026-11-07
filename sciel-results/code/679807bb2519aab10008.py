import numpy as np
import time
from scipy.optimize import minimize

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 275.0
        dim = self.dim
        budget = self.budget
        rng = np.random.default_rng(42)

        best_x = np.zeros(dim)
        best_f = np.inf
        total_calls = 0

        def time_ok():
            return time.time() < deadline

        def eval_func(x):
            nonlocal total_calls, best_f, best_x
            if total_calls >= budget or not time_ok():
                return np.inf
            val = func(x)
            total_calls += 1
            if val < best_f:
                best_f = val
                best_x = np.array(x)
            return val

        # Multi-start Nelder-Mead strategy
        # Partition budget into 10 independent runs of ~50 evaluations each
        # Each starts from a distinct deterministic point in [-4, 4]^5
        
        # Generate 10 distinct starting points using a grid + random perturbation
        # Use a Sobol-like or deterministic spread
        n_starts = 10
        calls_per_start = (budget - 10) // n_starts  # reserve 10 for final polish
        calls_per_start = max(calls_per_start, 20)
        
        # Create diverse starting points
        start_points = []
        
        # Use a structured grid for coverage
        # For 5 dimensions, use a combination of center, corners, and random points
        # Point 0: center
        start_points.append(np.zeros(dim))
        
        # Points 1-4: axis-aligned points
        for i in range(dim):
            pt = np.zeros(dim)
            pt[i] = 3.0
            start_points.append(pt.copy())
        
        # Points 5-9: random points in [-4, 4]^5
        for i in range(5):
            pt = rng.uniform(-4, 4, size=dim)
            start_points.append(pt)
        
        # Ensure we have exactly n_starts points
        while len(start_points) < n_starts:
            pt = rng.uniform(-4, 4, size=dim)
            start_points.append(pt)
        
        start_points = start_points[:n_starts]
        
        # Initial evaluation at first start point
        if total_calls < budget and time_ok():
            eval_func(start_points[0])
        
        # Run Nelder-Mead from each start point
        for idx, start_pt in enumerate(start_points):
            if total_calls >= budget or not time_ok():
                break
            
            # Skip if we already evaluated this point
            if idx == 0:
                continue
            
            # Evaluate the starting point
            if total_calls < budget and time_ok():
                eval_func(start_pt)
            
            # Run Nelder-Mead from this starting point
            if total_calls < budget and time_ok():
                try:
                    # Limit the number of iterations based on remaining budget
                    max_iter = min(50, budget - total_calls - 1)
                    if max_iter < 10:
                        max_iter = 10
                    
                    res = minimize(eval_func, start_pt, method='Nelder-Mead',
                                   options={'maxiter': max_iter, 
                                            'xatol': 1e-6, 
                                            'fatol': 1e-6,
                                            'adaptive': True})
                except Exception:
                    pass
        
        # Final polish with a few more Nelder-Mead iterations from best point
        if total_calls < budget and time_ok():
            try:
                remaining = budget - total_calls
                if remaining > 10:
                    res = minimize(eval_func, best_x, method='Nelder-Mead',
                                   options={'maxiter': min(30, remaining - 1),
                                            'xatol': 1e-7,
                                            'fatol': 1e-7,
                                            'adaptive': True})
            except Exception:
                pass
        
        return best_x, best_f
