"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Deterministic coordinate-descent with adaptive step sizes and time guard."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280  # safety margin before 300s limit

        # Initialize with a few random points to get a good starting point
        rng = np.random.default_rng(42)
        x = np.zeros(self.dim)
        fx = func(x)
        best_x = x.copy()
        best_fx = fx

        # Try a few random starts to find a better initial point
        for i in range(5):
            if time.time() > deadline:
                break
            x_try = rng.uniform(-5, 5, self.dim)
            fx_try = func(x_try)
            if fx_try < best_fx:
                best_fx = fx_try
                best_x = x_try.copy()
                x = x_try.copy()
                fx = fx_try

        # Coordinate descent with adaptive step sizes
        # For Sharp Ridge, the function is separable, so coordinate descent works well
        step_sizes = np.ones(self.dim) * 1.0
        x = best_x.copy()
        fx = best_fx

        # Main coordinate descent loop
        eval_count = 6  # 1 initial + 5 random starts
        max_evals = self.budget

        while eval_count < max_evals:
            if time.time() > deadline:
                break

            improved = False
            for d in range(self.dim):
                if time.time() > deadline:
                    break
                if eval_count >= max_evals:
                    break

                # Try moving in positive direction
                x_new = x.copy()
                x_new[d] = np.clip(x_new[d] + step_sizes[d], -5, 5)
                fx_new = func(x_new)
                eval_count += 1

                if fx_new < fx:
                    x = x_new
                    fx = fx_new
                    improved = True
                    step_sizes[d] *= 1.5  # Grow step size
                else:
                    # Try moving in negative direction
                    x_new = x.copy()
                    x_new[d] = np.clip(x_new[d] - step_sizes[d], -5, 5)
                    fx_new = func(x_new)
                    eval_count += 1

                    if fx_new < fx:
                        x = x_new
                        fx = fx_new
                        improved = True
                        step_sizes[d] *= 1.5
                    else:
                        step_sizes[d] *= 0.5  # Shrink step size

                if fx < best_fx:
                    best_fx = fx
                    best_x = x.copy()

            # If no improvement in a full cycle, shrink all steps
            if not improved:
                step_sizes *= 0.7
                # If steps are too small, do a random restart
                if np.all(step_sizes < 1e-4):
                    x_try = best_x + rng.normal(0, 0.5, self.dim)
                    x_try = np.clip(x_try, -5, 5)
                    fx_try = func(x_try)
                    eval_count += 1
                    if fx_try < best_fx:
                        best_fx = fx_try
                        best_x = x_try.copy()
                        x = x_try.copy()
                        fx = fx_try
                    step_sizes = np.ones(self.dim) * 0.5

        return best_x, best_fx
# EVOLVE-BLOCK-END
