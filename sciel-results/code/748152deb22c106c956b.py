"""Sobol-based adaptive search for Rastrigin f15."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Sobol quasi-random global search with adaptive step scaling."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def _sobol_sequence(self, n, dim, seed=12345):
        """Generate a Sobol sequence in [0, 1)^dim."""
        from scipy.stats import qmc
        sampler = qmc.Sobol(d=dim, scramble=True, seed=seed)
        points = sampler.random(n)
        return points

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0  # safety margin before 300s kill

        dim = self.dim
        budget = self.budget

        # Phase 1: Global exploration with Sobol sequence
        # Use a good chunk of budget for global coverage
        n_global = min(200, budget // 2)

        # Generate Sobol points in [-5, 5]^dim
        sobol_pts = self._sobol_sequence(n_global, dim, seed=42)
        sobol_pts = sobol_pts * 10.0 - 5.0  # map to [-5, 5]

        best_x = None
        best_f = np.inf
        call_count = 0

        # Evaluate global points
        for i in range(n_global):
            if call_count >= budget:
                break
            if time.time() > deadline:
                break

            x = sobol_pts[i]
            fx = func(x)
            call_count += 1

            if fx < best_f:
                best_f = fx
                best_x = x.copy()

        # Phase 2: Adaptive local refinement around best point
        # Use remaining budget for local search
        remaining = budget - call_count
        if remaining <= 0 or time.time() > deadline:
            return best_x, best_f

        # Adaptive step size based on Rastrigin periodicity (period = 2)
        # Start with moderate step and adapt
        sigma = 1.5  # initial step size
        rng = np.random.default_rng(7)

        # Use a (1+1)-ES style local search with adaptive sigma
        x = best_x.copy()
        fx = best_f

        # Track success rate for sigma adaptation
        successes = 0
        window = 10
        success_history = []

        for i in range(remaining):
            if time.time() > deadline:
                break

            # Generate perturbation
            y = x + sigma * rng.standard_normal(dim)
            y = np.clip(y, -5, 5)

            fy = func(y)
            call_count += 1

            if fy < fx:
                x = y
                fx = fy
                successes += 1
                # Increase sigma slightly on success
                sigma *= 1.2
                if sigma > 4.0:
                    sigma = 4.0
            else:
                # Decrease sigma on failure
                sigma *= 0.85
                if sigma < 0.1:
                    sigma = 0.1

            # Adaptive: if stuck, try a larger jump
            success_history.append(1 if fy < fx else 0)
            if len(success_history) > window:
                success_history.pop(0)

            if len(success_history) == window:
                success_rate = np.mean(success_history)
                if success_rate < 0.1:
                    # Stagnant: try a random restart from global best with larger step
                    sigma = 2.0
                    # Occasionally jump to a new random point near best
                    if rng.random() < 0.3:
                        x = best_x + 1.0 * rng.standard_normal(dim)
                        x = np.clip(x, -5, 5)
                        fx = func(x)
                        call_count += 1
                        if fx < best_f:
                            best_f = fx
                            best_x = x.copy()
                        success_history = []

        # Final: ensure we return the overall best
        if best_x is None:
            best_x = np.zeros(dim)
            best_f = func(best_x)

        return best_x, best_f
# EVOLVE-BLOCK-END
