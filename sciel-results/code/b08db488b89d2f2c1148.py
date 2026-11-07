"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import time


class Optimizer:
    """Derivative-free optimization using Nelder-Mead with multiple restarts."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280  # 280s safety margin within 300s limit
        rng = np.random.default_rng(42)

        best_x = None
        best_f = np.inf
        eval_count = 0

        def wrapped_func(x):
            nonlocal eval_count
            if time.time() > deadline:
                raise TimeoutError("Deadline exceeded")
            eval_count += 1
            x_clipped = np.clip(x, -5, 5)
            return func(x_clipped)

        # Phase 1: Scatter initial points to find good basins
        n_initial = min(50, self.budget // 10)
        for i in range(n_initial):
            if eval_count >= self.budget:
                break
            x = rng.uniform(-5, 5, self.dim)
            fx = wrapped_func(x)
            if fx < best_f:
                best_f = fx
                best_x = x.copy()

        # Phase 2: Nelder-Mead restarts from diverse starting points
        # We'll do multiple restarts, each with a budget of function evaluations
        n_restarts = 8
        evals_per_restart = max(30, (self.budget - n_initial) // n_restarts)

        for r in range(n_restarts):
            if eval_count >= self.budget:
                break
            if time.time() > deadline:
                break

            # Choose starting point: best found so far with some perturbation
            # or a random point
            if r == 0:
                x0 = best_x.copy()
            elif r % 2 == 0:
                x0 = best_x + rng.standard_normal(self.dim) * 1.0
            else:
                x0 = rng.uniform(-5, 5, self.dim)
            x0 = np.clip(x0, -5, 5)

            # Create a simplex for Nelder-Mead
            # Start with a small simplex around x0
            simplex = [x0.copy()]
            for i in range(self.dim):
                perturbation = x0.copy()
                perturbation[i] += 0.5
                perturbation[i] = np.clip(perturbation[i], -5, 5)
                simplex.append(perturbation)

            try:
                result = minimize(
                    wrapped_func,
                    x0,
                    method='Nelder-Mead',
                    options={
                        'maxiter': 200,
                        'maxfev': evals_per_restart,
                        'xatol': 1e-6,
                        'fatol': 1e-6,
                        'adaptive': True
                    }
                )
                if result.fun < best_f:
                    best_f = result.fun
                    best_x = result.x.copy()
            except (TimeoutError, Exception):
                # If timeout or any exception, just continue with next restart
                if time.time() > deadline:
                    break

        # Phase 3: Final local refinement around the best point
        if eval_count < self.budget and time.time() < deadline:
            remaining = self.budget - eval_count
            if remaining > 20:
                try:
                    result = minimize(
                        wrapped_func,
                        best_x,
                        method='Nelder-Mead',
                        options={
                            'maxiter': 100,
                            'maxfev': remaining,
                            'xatol': 1e-8,
                            'fatol': 1e-8,
                            'adaptive': True
                        }
                    )
                    if result.fun < best_f:
                        best_f = result.fun
                        best_x = result.x.copy()
                except (TimeoutError, Exception):
                    pass

        # Ensure we return valid values
        if best_x is None:
            best_x = np.zeros(self.dim)
            best_f = func(best_x)

        return best_x, best_f
# EVOLVE-BLOCK-END
