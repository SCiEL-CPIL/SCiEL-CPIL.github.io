"""Improved optimizer for BBOB f19 (Griewank-Rosenbrock) with adaptive CMA-like search."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Adaptive coordinate-wise search with momentum and deadline guard."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280  # safety margin before 300s limit

        rng = np.random.default_rng(42)
        dim = self.dim
        budget = self.budget

        # Initial point: center of the box
        x = np.zeros(dim)
        fx = func(x)
        best_x = x.copy()
        best_fx = fx

        # Adaptive step sizes per dimension
        sigma = np.full(dim, 1.0)
        # Momentum
        momentum = np.zeros(dim)
        beta = 0.8

        # Track per-dimension success for adaptive sigma
        successes = np.zeros(dim)
        total_tries = np.zeros(dim)

        call_count = 1

        for iteration in range(budget - 1):
            # Check deadline
            if time.time() > deadline:
                break

            # Generate candidate
            # Mix of large exploratory steps and small exploitative steps
            if iteration < 50:
                # Exploration phase: larger steps
                step_scale = 2.0
            elif iteration < 150:
                step_scale = 1.0
            else:
                # Exploitation phase: smaller steps
                step_scale = 0.3

            # Per-dimension adaptive sigma
            adaptive_sigma = sigma * step_scale
            noise = rng.standard_normal(dim)
            y = x + adaptive_sigma * noise
            y = np.clip(y, -5, 5)

            fy = func(y)
            call_count += 1

            if fy < fx:
                # Success: move toward y
                improvement = y - x
                momentum = beta * momentum + (1 - beta) * improvement
                x = y
                fx = fy

                # Increase sigma for dimensions that improved
                for d in range(dim):
                    if abs(improvement[d]) > 1e-10:
                        sigma[d] = min(sigma[d] * 1.2, 5.0)
                        successes[d] += 1

                # Update best
                if fy < best_fx:
                    best_x = x.copy()
                    best_fx = fy
            else:
                # Failure: shrink sigma
                for d in range(dim):
                    sigma[d] *= 0.95
                    total_tries[d] += 1

                # Occasionally try a coordinate-wise step
                if iteration % 10 == 0:
                    coord = rng.integers(dim)
                    step_dir = rng.choice([-1, 1])
                    y_coord = x.copy()
                    y_coord[coord] = x[coord] + step_dir * sigma[coord] * 0.5
                    y_coord = np.clip(y_coord, -5, 5)
                    fy_coord = func(y_coord)
                    call_count += 1
                    if fy_coord < fx:
                        x = y_coord
                        fx = fy_coord
                        if fy_coord < best_fx:
                            best_x = x.copy()
                            best_fx = fy_coord
                        sigma[coord] = min(sigma[coord] * 1.3, 5.0)
                    else:
                        sigma[coord] *= 0.9

            # Reset momentum if stuck
            if iteration > 100 and iteration % 50 == 0:
                # Try a random restart in a neighborhood
                if rng.random() < 0.1:
                    restart_x = x + rng.standard_normal(dim) * 0.5
                    restart_x = np.clip(restart_x, -5, 5)
                    fr = func(restart_x)
                    call_count += 1
                    if fr < fx:
                        x = restart_x
                        fx = fr
                        if fr < best_fx:
                            best_x = x.copy()
                            best_fx = fr

            # Final polish: if we've been stable, do fine-tuning
            if iteration > budget * 0.7:
                # Reduce step sizes further
                sigma *= 0.98

        return best_x, best_fx
# EVOLVE-BLOCK-END
