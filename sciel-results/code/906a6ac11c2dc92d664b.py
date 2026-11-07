"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import qmc


class Optimizer:
    """Sobol space-filling search followed by local refinement."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290  # safety margin before 300s limit

        n = self.dim
        budget = self.budget

        # Phase 1: Sobol sequence for broad exploration
        # Use a Sobol sequence with enough points to cover a good portion of budget
        sobol_points_count = min(256, budget // 2)  # power of 2 for Sobol

        # Generate Sobol sequence in [0,1]^n
        sampler = qmc.Sobol(d=n, scramble=True, seed=42)
        unit_points = sampler.random(sobol_points_count)

        # Map to [-5, 5]^n
        sobol_points = -5.0 + 10.0 * unit_points

        best_x = None
        best_f = np.inf

        # Evaluate Sobol points
        for i in range(sobol_points_count):
            if time.time() > deadline:
                break

            x = sobol_points[i]
            fx = func(x)

            if fx < best_f:
                best_f = fx
                best_x = x.copy()

        # Phase 2: Local refinement around the best point
        # Use a simple gradient-free local search with decreasing step size
        if best_x is not None and time.time() < deadline:
            # Local refinement using coordinate-wise search
            # Start with a moderate step and decrease
            step_sizes = [1.0, 0.5, 0.25, 0.125, 0.0625, 0.03125, 0.015625, 0.0078125]

            for step in step_sizes:
                if time.time() > deadline:
                    break

                improved = True
                while improved and time.time() < deadline:
                    improved = False

                    # Try moving in each coordinate direction (positive and negative)
                    for d in range(n):
                        for sign in [1.0, -1.0]:
                            if time.time() > deadline:
                                break

                            trial_x = best_x.copy()
                            trial_x[d] += sign * step
                            trial_x = np.clip(trial_x, -5, 5)

                            trial_f = func(trial_x)

                            if trial_f < best_f:
                                best_f = trial_f
                                best_x = trial_x
                                improved = True

        # Phase 3: If we have budget remaining, do a few random perturbations
        # around the best point with small steps
        if time.time() < deadline and best_x is not None:
            rng = np.random.default_rng(0)
            remaining_time = deadline - time.time()
            # Estimate how many more evaluations we can afford
            # Assume ~0.1ms per evaluation, so we can do many
            # But let's be conservative and just do a few
            for _ in range(min(50, int(remaining_time * 1000))):
                if time.time() > deadline:
                    break

                # Small random perturbation
                perturbation = 0.01 * rng.standard_normal(n)
                trial_x = np.clip(best_x + perturbation, -5, 5)
                trial_f = func(trial_x)

                if trial_f < best_f:
                    best_f = trial_f
                    best_x = trial_x

        return best_x, best_f
# EVOLVE-BLOCK-END
