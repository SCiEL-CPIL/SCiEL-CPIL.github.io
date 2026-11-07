"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES-lite with per-dimension adaptive step sizes and periodic gradient polish."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 320.0  # safety margin before 330s kill

        rng = np.random.default_rng(42)
        dim = self.dim

        # Initialize
        x = rng.uniform(-5, 5, dim)
        fx = func(x)
        best_x = x.copy()
        best_fx = fx

        # Per-dimension step sizes
        sigma = np.full(dim, 2.0)
        # Track recent successes per dimension
        recent_successes = [0] * dim
        recent_trials = [0] * dim
        window = 10

        calls_made = 1
        total_budget = self.budget

        for iteration in range(total_budget - 1):
            # Time guard
            if time.time() > deadline:
                break

            # Periodic gradient polish every 50 iterations
            if iteration % 50 == 0 and iteration > 0 and calls_made < total_budget - 2:
                # Finite-difference gradient estimation
                eps = 1e-4
                grad = np.zeros(dim)
                for i in range(dim):
                    if time.time() > deadline:
                        break
                    xp = x.copy()
                    xm = x.copy()
                    xp[i] += eps
                    xm[i] -= eps
                    xp = np.clip(xp, -5, 5)
                    xm = np.clip(xm, -5, 5)
                    fp = func(xp)
                    fm = func(xm)
                    calls_made += 2
                    grad[i] = (fp - fm) / (2 * eps)

                # Gradient descent step
                step_size = min(0.5, np.linalg.norm(grad) * 0.1)
                if step_size > 1e-10:
                    x_new = np.clip(x - step_size * grad, -5, 5)
                    fx_new = func(x_new)
                    calls_made += 1
                    if fx_new < fx:
                        x, fx = x_new, fx_new
                        if fx < best_fx:
                            best_x, best_fx = x.copy(), fx
                    # Reduce sigma after polish
                    sigma *= 0.9

            if calls_made >= total_budget:
                break

            # CMA-ES-lite: sample with per-dimension sigma
            y = x + sigma * rng.standard_normal(dim)
            y = np.clip(y, -5, 5)
            fy = func(y)
            calls_made += 1

            if fy < fx:
                # Success: update x and increase sigma for dimensions that moved
                x = y
                fx = fy
                if fx < best_fx:
                    best_x, best_fx = x.copy(), fx

                # Track per-dimension success
                for i in range(dim):
                    recent_successes[i] += 1
                    recent_trials[i] += 1
                    if recent_trials[i] >= window:
                        rate = recent_successes[i] / recent_trials[i]
                        # Adapt sigma: increase if success rate > 0.5, decrease otherwise
                        if rate > 0.5:
                            sigma[i] *= 1.2
                        elif rate < 0.5:
                            sigma[i] *= 0.8
                        # Reset window
                        recent_successes[i] = 0
                        recent_trials[i] = 0

                # Cap sigma
                sigma = np.clip(sigma, 0.01, 10.0)
            else:
                # Failure: decrease sigma
                for i in range(dim):
                    recent_trials[i] += 1
                    if recent_trials[i] >= window:
                        rate = recent_successes[i] / recent_trials[i]
                        if rate < 0.3:
                            sigma[i] *= 0.7
                        elif rate < 0.5:
                            sigma[i] *= 0.9
                        recent_successes[i] = 0
                        recent_trials[i] = 0

                sigma = np.clip(sigma, 0.01, 10.0)

        return best_x, best_fx
# EVOLVE-BLOCK-END
