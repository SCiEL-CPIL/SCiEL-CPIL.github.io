"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


class Optimizer:
    """Derivative-free optimizer using Nelder-Mead with random restarts."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280  # safety margin before 300s limit
        rng = np.random.default_rng(42)
        
        # Track best solution
        best_x = None
        best_f = np.inf
        calls_used = 0
        max_calls = self.budget

        def safe_func(x):
            nonlocal calls_used
            if calls_used >= max_calls:
                return best_f
            calls_used += 1
            x_clipped = np.clip(x, -5.0, 5.0)
            val = func(x_clipped)
            if val < best_f:
                best_f = val
                best_x = x_clipped.copy()
            return val

        # Phase 1: Random search to find a good starting point
        n_random = min(30, max_calls // 10)
        for _ in range(n_random):
            if time.time() > deadline:
                break
            x = rng.uniform(-5, 5, self.dim)
            safe_func(x)

        # Phase 2: Nelder-Mead from best point, with multiple restarts
        remaining = max_calls - calls_used
        # Split remaining budget into several Nelder-Mead runs
        n_restarts = min(5, max(1, remaining // 50))
        budget_per_restart = max(20, (remaining - 10) // n_restarts)
        
        for r in range(n_restarts):
            if time.time() > deadline:
                break
            if calls_used >= max_calls:
                break
            
            # Start from current best or a new random point for later restarts
            if r == 0:
                x0 = best_x.copy()
            else:
                # Perturb the best point for diversity
                x0 = best_x + rng.normal(0, 1.0, self.dim)
                x0 = np.clip(x0, -5, 5)
            
            # Run Nelder-Mead
            try:
                result = minimize(
                    safe_func,
                    x0,
                    method='Nelder-Mead',
                    options={
                        'maxiter': budget_per_restart,
                        'xatol': 1e-6,
                        'fatol': 1e-8,
                        'adaptive': True,
                        'initial_simplex': np.array([x0 + rng.normal(0, 0.5, self.dim) for _ in range(self.dim + 1)])
                    }
                )
                if result.fun < best_f:
                    best_f = result.fun
                    best_x = np.clip(result.x, -5, 5)
            except Exception:
                pass

        # Phase 3: Fine-tune with small steps around the best point
        if time.time() < deadline and calls_used < max_calls:
            sigma = 0.1
            for _ in range(min(50, max_calls - calls_used)):
                if time.time() > deadline:
                    break
                if calls_used >= max_calls:
                    break
                y = np.clip(best_x + sigma * rng.standard_normal(self.dim), -5, 5)
                fy = safe_func(y)
                if fy < best_f:
                    best_f = fy
                    best_x = y.copy()
                    sigma *= 1.2
                else:
                    sigma *= 0.8

        return best_x, best_f
# EVOLVE-BLOCK-END
