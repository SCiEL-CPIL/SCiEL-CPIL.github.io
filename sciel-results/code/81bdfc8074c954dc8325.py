"""Surrogate-assisted derivative-free optimizer for BBOB f11 (Discus)."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
from scipy.stats import norm


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
        deadline = start_time + 280.0  # 20s safety margin

        rng = np.random.default_rng(42)
        calls_used = 0
        best_x = np.zeros(self.dim)
        best_f = np.inf

        def check_time():
            return (time.time() - start_time) > (deadline - 5.0)

        def try_update(x, f):
            nonlocal best_x, best_f
            if f < best_f:
                best_x = x.copy()
                best_f = f

        # Phase 1: Random sampling for broad exploration
        # Use Halton sequence for better coverage than pure random
        def halton(index, base):
            f = 1.0
            r = 0.0
            while index > 0:
                f /= base
                r += f * (index % base)
                index //= base
            return r

        n_random = min(50, self.budget)
        for i in range(n_random):
            if calls_used >= self.budget or check_time():
                break
            x = np.array([halton(i + 1, base) for base in [2, 3, 5, 7, 11, 13, 17, 19, 23, 29][:self.dim]])
            # Map from [0,1] to [-5, 5]
            x = -5.0 + 10.0 * x
            f = func(x)
            calls_used += 1
            try_update(x, f)

        if calls_used >= self.budget or check_time():
            return best_x, best_f

        # Phase 2: Nelder-Mead local refinements
        # Start from best points found in Phase 1
        n_nm = min(15, (self.budget - calls_used) // 30)
        
        # Select diverse starting points
        # Use the best point and some random points near it, plus a few fully random
        start_points = [best_x.copy()]
        
        # Add points near the best
        for _ in range(min(n_nm - 1, 8)):
            if calls_used >= self.budget or check_time():
                break
            # Perturb the best point
            perturbation = rng.normal(0, 1.0, self.dim)
            x_start = np.clip(best_x + perturbation, -5, 5)
            start_points.append(x_start)
        
        # Add some fully random starts to escape local minima
        for _ in range(min(4, n_nm - len(start_points))):
            if calls_used >= self.budget or check_time():
                break
            x_start = rng.uniform(-5, 5, self.dim)
            start_points.append(x_start)

        for x_start in start_points:
            if calls_used >= self.budget or check_time():
                break
            
            # Track local calls
            local_calls = [0]
            
            def nm_func(x):
                nonlocal calls_used, local_calls
                if local_calls[0] >= 60 or calls_used >= self.budget:
                    return best_f + 1e10  # Return current best to stop
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
                    options={'maxiter': 150, 'xatol': 1e-5, 'fatol': 1e-7, 'adaptive': True}
                )
                if result.fun < best_f:
                    try_update(result.x, result.fun)
            except Exception:
                pass

        if calls_used >= self.budget or check_time():
            return best_x, best_f

        # Phase 3: (1+1)-ES polish
        x = best_x.copy()
        fx = best_f
        sigma = 0.3
        
        # Adaptive sigma with success rate tracking
        successes = 0
        total = 0
        window = 20
        
        while calls_used < self.budget and not check_time():
            y = np.clip(x + sigma * rng.standard_normal(self.dim), -5, 5)
            fy = func(y)
            calls_used += 1
            total += 1
            
            if fy < fx:
                x, fx = y, fy
                try_update(x, fx)
                successes += 1
                sigma *= 1.1
            else:
                sigma *= 0.95
            
            # Adapt sigma based on success rate
            if total % window == 0:
                success_rate = successes / total
                if success_rate > 0.3:
                    sigma *= 1.05
                elif success_rate < 0.1:
                    sigma *= 0.9
                successes = 0
                total = 0

            if sigma < 1e-7:
                break

        # Final high-precision polish with very small steps
        if not check_time() and calls_used < self.budget:
            sigma_final = max(sigma * 0.1, 1e-5)
            while calls_used < self.budget and not check_time():
                y = np.clip(x + sigma_final * rng.standard_normal(self.dim), -5, 5)
                fy = func(y)
                calls_used += 1
                if fy < fx:
                    x, fx = y, fy
                    try_update(x, fx)
                else:
                    sigma_final *= 0.8
                    if sigma_final < 1e-8:
                        break

        return best_x, best_f
# EVOLVE-BLOCK-END
