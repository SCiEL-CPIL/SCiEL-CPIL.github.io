"""Optimized program for BBOB f18 (Schaffers F7 ill-conditioned)."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Pattern search with adaptive step sizes and time guard."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0  # safety margin before 300s kill

        rng = np.random.default_rng(42)
        dim = self.dim
        budget = self.budget

        # Initial point
        x = rng.uniform(-4, 4, dim)
        fx = func(x)
        calls = 1

        # Build search directions: standard basis + diagonals
        directions = []
        for i in range(dim):
            e = np.zeros(dim)
            e[i] = 1.0
            directions.append(e)
            directions.append(-e)
        # Diagonal directions (all +1, all -1, and some random)
        directions.append(np.ones(dim) / np.sqrt(dim))
        directions.append(-np.ones(dim) / np.sqrt(dim))
        for _ in range(dim):
            d = rng.standard_normal(dim)
            d /= np.linalg.norm(d)
            directions.append(d)
            directions.append(-d)

        # Adaptive step sizes per direction
        step_sizes = np.full(len(directions), 2.0)

        best_x = x.copy()
        best_fx = fx

        while calls < budget:
            # Time guard
            if time.time() > deadline:
                break

            improved = False
            for di, d in enumerate(directions):
                if calls >= budget:
                    break
                if time.time() > deadline:
                    break

                step = step_sizes[di]
                # Try positive direction
                y = np.clip(x + step * d, -5, 5)
                fy = func(y)
                calls += 1

                if fy < fx:
                    x, fx = y, fy
                    step_sizes[di] *= 1.5
                    improved = True
                    if fx < best_fx:
                        best_fx, best_x = fx, x.copy()
                else:
                    # Try negative direction
                    y = np.clip(x - step * d, -5, 5)
                    fy = func(y)
                    calls += 1
                    if fy < fx:
                        x, fx = y, fy
                        step_sizes[di] *= 1.5
                        improved = True
                        if fx < best_fx:
                            best_fx, best_x = fx, x.copy()
                    else:
                        step_sizes[di] *= 0.5
                        # Clamp step size
                        step_sizes[di] = max(step_sizes[di], 1e-6)

            if not improved:
                # No improvement in full pass; perturb with random restart
                if calls < budget:
                    x = np.clip(x + 1.0 * rng.standard_normal(dim), -5, 5)
                    fx = func(x)
                    calls += 1
                    if fx < best_fx:
                        best_fx, best_x = fx, x.copy()
                    # Reset some step sizes
                    step_sizes[:] = 2.0

        return best_x, best_fx
# EVOLVE-BLOCK-END
