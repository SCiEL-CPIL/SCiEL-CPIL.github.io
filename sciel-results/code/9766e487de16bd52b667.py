"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import qmc


class Optimizer:
    """Hybrid: Sobol exploration + multi-start (1+1)-ES refinement."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        deadline = time.time() + 280  # Safety margin for 300s limit

        # Phase 1: Sobol-based global exploration (deterministic, space-filling)
        # Generate a Sobol sequence for the first ~200 points
        n_explore = 200
        sobol = qmc.Sobol(d=dim, scramble=True, seed=42)
        # Generate 2^ceil(log2(n_explore)) points, take first n_explore
        power = max(1, int(np.ceil(np.log2(n_explore))))
        sobol_samples = sobol.random(2 ** power)
        sobol_samples = sobol_samples[:n_explore]
        # Map from [0,1]^dim to [-5,5]^dim
        sobol_samples = sobol_samples * 10.0 - 5.0

        best_x = None
        best_f = np.inf

        for i in range(n_explore):
            if time.time() > deadline:
                break
            x = sobol_samples[i]
            fx = func(x)
            if fx < best_f:
                best_f = fx
                best_x = x.copy()

        # Phase 2: Multi-start (1+1)-ES from diverse starting points
        # Use remaining budget for refinement
        remaining_budget = budget - n_explore

        # Generate diverse starting points: best found + random + zero
        rng = np.random.default_rng(123)
        start_points = [best_x.copy()]
        # Add zero vector
        start_points.append(np.zeros(dim))
        # Add a few random points
        for _ in range(4):
            start_points.append(rng.uniform(-5, 5, dim))

        # Allocate budget to each start point
        n_starts = len(start_points)
        budget_per_start = max(10, remaining_budget // n_starts)

        for start_idx, x0 in enumerate(start_points):
            if time.time() > deadline:
                break

            x = x0.copy()
            fx = func(x)
            if fx < best_f:
                best_f = fx
                best_x = x.copy()

            sigma = 1.0
            # Adaptive (1+1)-ES with 1/5 rule
            successes = 0
            window = 5
            for step in range(budget_per_start):
                if time.time() > deadline:
                    break
                y = np.clip(x + sigma * rng.standard_normal(dim), -5, 5)
                fy = func(y)
                if fy < fx:
                    x, fx = y, fy
                    if fx < best_f:
                        best_f = fx
                        best_x = x.copy()
                    successes += 1
                else:
                    pass

                # Every 5 steps, adapt sigma
                if (step + 1) % window == 0:
                    if successes > 0:
                        # Increase sigma if more than 1/5 succeeded
                        if successes / window > 0.2:
                            sigma *= 1.3
                        else:
                            sigma *= 0.7
                    else:
                        sigma *= 0.7
                    successes = 0

                # Clamp sigma to reasonable bounds
                sigma = np.clip(sigma, 0.01, 5.0)

        # Phase 3: Final polish with smaller sigma around best solution
        if time.time() < deadline - 10 and best_x is not None:
            x = best_x.copy()
            fx = func(x)
            sigma = 0.1
            for step in range(50):
                if time.time() > deadline:
                    break
                y = np.clip(x + sigma * rng.standard_normal(dim), -5, 5)
                fy = func(y)
                if fy < fx:
                    x, fx = y, fy
                    if fx < best_f:
                        best_f = fx
                        best_x = x.copy()
                sigma *= 0.95

        return best_x, best_f
# EVOLVE-BLOCK-END
