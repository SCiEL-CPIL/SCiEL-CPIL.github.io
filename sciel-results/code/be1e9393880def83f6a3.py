"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import differential_evolution, minimize


class Optimizer:
    """Hybrid DE + L-BFGS-B optimizer with time guard."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        start_time = time.time()
        deadline = start_time + 280  # safety margin before 300s kill

        # Track best
        best_x = None
        best_f = np.inf
        call_count = 0

        def tracked_func(x):
            nonlocal call_count, best_x, best_f
            if time.time() > deadline:
                raise TimeoutError()
            call_count += 1
            if call_count > budget:
                raise ValueError("Budget exceeded")
            fx = func(x)
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            return fx

        # Phase 1: Differential Evolution for global exploration
        # Use a moderate population and maxiter to stay within budget
        # DE in 5D with popsize=15 means 15*5=75 evals per generation
        # maxiter=5 means ~375 evals for DE
        try:
            de_result = differential_evolution(
                tracked_func,
                bounds=[(-5, 5)] * dim,
                popsize=15,
                maxiter=5,
                tol=1e-6,
                mutation=(0.3, 1.0),
                recombination=0.5,
                seed=42,
                polish=False,
                init='latinhypercube'
            )
            if de_result.fun < best_f:
                best_f = de_result.fun
                best_x = de_result.x.copy()
        except (TimeoutError, ValueError):
            pass

        # Phase 2: Local refinement with L-BFGS-B from multiple starting points
        # Start from the DE best and a few other promising points
        start_points = [best_x]
        
        # Add a few more starting points for diversity
        rng = np.random.default_rng(123)
        for _ in range(3):
            # Start near best_x with some perturbation
            perturbation = rng.normal(0, 1.0, dim)
            start_point = np.clip(best_x + perturbation, -5, 5)
            start_points.append(start_point)

        for sp in start_points:
            if time.time() > deadline:
                break
            if call_count >= budget:
                break
            try:
                local_result = minimize(
                    tracked_func,
                    sp,
                    method='L-BFGS-B',
                    bounds=[(-5, 5)] * dim,
                    options={'maxiter': 100, 'ftol': 1e-12}
                )
                if local_result.fun < best_f:
                    best_f = local_result.fun
                    best_x = local_result.x.copy()
            except (TimeoutError, ValueError):
                pass

        return best_x, best_f
# EVOLVE-BLOCK-END
