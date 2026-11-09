"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-restart (1+1)-ES with diverse initial points, adaptive sigma, and time guard."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0  # Safety margin before 330s kill
        rng = np.random.default_rng(42)

        best_x = None
        best_f = np.inf

        # Generate diverse initial points: origin, shifted, random
        init_points = [
            np.zeros(self.dim),
            np.ones(self.dim) * 2.0,
            np.ones(self.dim) * -2.0,
            np.ones(self.dim) * 4.0,
            np.ones(self.dim) * -4.0,
            rng.uniform(-5, 5, self.dim),
            rng.uniform(-5, 5, self.dim),
            rng.uniform(-5, 5, self.dim),
            rng.uniform(-5, 5, self.dim),
            rng.uniform(-5, 5, self.dim),
        ]

        # Shuffle to diversify which starts get more budget
        idx = np.arange(len(init_points))
        rng.shuffle(idx)

        total_budget = self.budget
        calls_made = 0

        for start_idx in idx:
            if calls_made >= total_budget:
                break
            if time.time() >= deadline:
                break

            x0 = np.clip(init_points[start_idx], -5, 5)
            fx = func(x0)
            calls_made += 1

            if fx < best_f:
                best_f = fx
                best_x = x0.copy()

            # If we're already very close to a good value, spend less here
            # Adaptive sigma with 1/5 success rule
            sigma = 1.5
            x = x0.copy()
            f = fx
            window = []
            window_size = 10

            while calls_made < total_budget:
                if time.time() >= deadline:
                    break

                y = np.clip(x + sigma * rng.standard_normal(self.dim), -5, 5)
                fy = func(y)
                calls_made += 1

                if fy < f:
                    x = y
                    f = fy
                    window.append(1)
                    if f < best_f:
                        best_f = f
                        best_x = x.copy()
                else:
                    window.append(0)

                if len(window) >= window_size:
                    success_rate = np.mean(window)
                    if success_rate > 0.5:
                        sigma *= 1.3
                    elif success_rate < 0.5:
                        sigma *= 0.7
                    window = window[-window_size // 2:] if window_size > 2 else []

                # Early stopping if sigma is very small and no progress
                if sigma < 1e-4:
                    break

        return best_x, best_f
# EVOLVE-BLOCK-END
