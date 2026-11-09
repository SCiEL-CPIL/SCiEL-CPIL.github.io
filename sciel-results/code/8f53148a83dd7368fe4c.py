"""Surrogate-assisted derivative-free optimizer for BBOB f11 (Discus)."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


class Optimizer:
    """
    Multi-stage derivative-free optimizer:
    1. Random sampling for initial exploration.
    2. Nelder-Mead local refinements from diverse starting points.
    3. (1+1)-ES polish with adaptive step size.
    
    Uses a time guard to stay within the 300s wall-clock limit.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 285.0  # 15s safety margin

        rng = np.random.default_rng(42)
        calls_used = 0
        best_x = None
        best_f = np.inf
        
        # Store top points from random phase for diverse NM seeds
        top_points = []

        def check_time():
            return (time.time() - start_time) > (deadline - 5.0)

        def try_update(x, f):
            nonlocal best_x, best_f
            if f < best_f:
                best_x = x.copy()
                best_f = f

        # Phase 1: Random sampling for broad exploration
        n_random = min(30, self.budget)
        for _ in range(n_random):
            if calls_used >= self.budget or check_time():
                break
            x = rng.uniform(-5, 5, self.dim)
            f = func(x)
            calls_used += 1
            try_update(x, f)
            top_points.append((f, x.copy()))

        if calls_used >= self.budget or check_time():
            return best_x, best_f

        # Sort top_points by fitness to get diverse good starting points
        top_points.sort(key=lambda t: t[0])
        
        # Phase 2: Nelder-Mead local refinements from diverse starting points
        # Use the best point and a few other promising points
        n_nm = min(5, max(1, (self.budget - calls_used) // 60))
        
        # Select starting points: best + top diverse points
        start_points = [best_x.copy()]
        for f_val, x_val in top_points[1:]:
            if len(start_points) >= n_nm:
                break
            # Only add if it's not too close to existing start points
            too_close = False
            for sp in start_points:
                if np.linalg.norm(x_val - sp) < 1.0:
                    too_close = True
                    break
            if not too_close:
                start_points.append(x_val)

        # If we didn't get enough diverse points, add some random perturbations
        while len(start_points) < n_nm and calls_used < self.budget and not check_time():
            perturbation = rng.normal(0, 2.0, self.dim)
            x_start = np.clip(best_x + perturbation, -5, 5)
            start_points.append(x_start)

        for x_start in start_points:
            if calls_used >= self.budget or check_time():
                break
            
            local_calls = [0]
            
            def nm_func(x):
                nonlocal calls_used, local_calls
                if local_calls[0] >= 50 or calls_used >= self.budget:
                    return best_f + 1e10  # Return large value to stop
                x_clipped = np.clip(x, -5, 5)
                f = func(x_clipped)
                calls_used += 1
                local_calls[0] += 1
                return f

            try:
                result = minimize(
                    nm_func,
                    x_start,
                    method='Nelder-Mead',
                    options={'maxiter': 200, 'xatol': 1e-5, 'fatol': 1e-7, 'adaptive': True}
                )
                if result.fun < best_f:
                    try_update(result.x, result.fun)
            except Exception:
                pass

        if calls_used >= self.budget or check_time():
            return best_x, best_f

        # Phase 3: (1+1)-ES polish with adaptive step size and restarts
        x = best_x.copy()
        fx = best_f
        sigma = 0.3
        
        while calls_used < self.budget and not check_time():
            y = np.clip(x + sigma * rng.standard_normal(self.dim), -5, 5)
            fy = func(y)
            calls_used += 1
            
            if fy < fx:
                x, fx = y, fy
                try_update(x, fx)
                sigma *= 1.15
            else:
                sigma *= 0.88
            
            if sigma < 1e-5:
                # Restart with larger sigma if we have budget
                if calls_used < self.budget - 10 and not check_time():
                    sigma = 0.5
                    # Small random jump to escape local minimum
                    x = np.clip(x + 0.3 * rng.standard_normal(self.dim), -5, 5)
                    fx = func(x)
                    calls_used += 1
                    try_update(x, fx)
                else:
                    break

        return best_x, best_f
# EVOLVE-BLOCK-END
