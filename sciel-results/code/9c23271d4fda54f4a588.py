"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES-lite with coordinate-wise adaptive step sizes and gradient polish."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280  # safety margin before 300s limit
        rng = np.random.default_rng(42)
        d = self.dim
        budget = self.budget

        # Initial point
        x = rng.uniform(-5, 5, d)
        fx = func(x)
        best_x = x.copy()
        best_fx = fx
        calls = 1

        # Coordinate-wise step sizes
        sigma = np.full(d, 1.0)
        min_sigma = 1e-6
        max_sigma = 10.0

        # Track recent successes per coordinate
        window = 10
        successes = np.zeros((d, window))
        success_ptr = 0

        # Phase control: alternate between exploration and gradient polish
        # Every 50 calls, do a gradient polish if stagnating
        calls_since_polish = 0
        last_improve_call = 0

        def try_step(x, fx, sigma, calls_remaining):
            """Attempt a step with coordinate-wise adaptive sizes."""
            nonlocal best_x, best_fx, calls
            direction = rng.standard_normal(d)
            y = np.clip(x + sigma * direction, -5, 5)
            fy = func(y)
            calls += 1

            if fy < fx:
                # Success: increase sigma for coordinates that moved in the right direction
                for i in range(d):
                    if direction[i] * (y[i] - x[i]) > 0:
                        successes[i, success_ptr % window] = 1
                    else:
                        successes[i, success_ptr % window] = 0
                x, fx = y, fy
                # Adapt sigma based on success rate per coordinate
                for i in range(d):
                    rate = successes[i].mean()
                    if rate > 0.5:
                        sigma[i] = min(sigma[i] * 1.3, max_sigma)
                    else:
                        sigma[i] = max(sigma[i] * 0.85, min_sigma)
                return x, fx, True
            else:
                # Failure: decrease sigma for coordinates that moved
                for i in range(d):
                    successes[i, success_ptr % window] = 0
                for i in range(d):
                    rate = successes[i].mean()
                    if rate < 0.5:
                        sigma[i] = max(sigma[i] * 0.9, min_sigma)
                return x, fx, False

        def gradient_polish(x, fx, budget_remaining):
            """Finite-difference gradient descent polish."""
            nonlocal best_x, best_fx, calls
            if budget_remaining < 10:
                return x, fx

            h = 1e-4
            # Estimate gradient
            grad = np.zeros(d)
            for i in range(d):
                if calls >= budget or time.time() > deadline:
                    break
                xp = x.copy()
                xp[i] += h
                xp = np.clip(xp, -5, 5)
                fp = func(xp)
                calls += 1

                xm = x.copy()
                xm[i] -= h
                xm = np.clip(xm, -5, 5)
                fm = func(xm)
                calls += 1

                grad[i] = (fp - fm) / (2 * h)

            if calls >= budget or time.time() > deadline:
                return x, fx

            # Do a few gradient steps
            lr = 0.1
            for _ in range(3):
                if calls >= budget or time.time() > deadline:
                    break
                # Line search: try shrinking step
                for scale in [lr, lr * 0.5, lr * 0.1]:
                    y = np.clip(x - scale * grad, -5, 5)
                    fy = func(y)
                    calls += 1
                    if fy < fx:
                        x, fx = y, fy
                        break
                else:
                    break

            return x, fx

        while calls < budget and time.time() < deadline:
            calls_since_polish += 1

            # Check for stagnation
            if fx < best_fx:
                best_x = x.copy()
                best_fx = fx
                last_improve_call = calls

            # Every 40 calls, if no improvement in last 30, do gradient polish
            if calls_since_polish >= 40 and (calls - last_improve_call) >= 30:
                x, fx = gradient_polish(x, fx, budget - calls)
                calls_since_polish = 0
                # Reset sigma to moderate values after polish
                sigma = np.clip(sigma, 0.01, 5.0)
                successes = np.zeros((d, window))
                success_ptr = 0
                continue

            x, fx, improved = try_step(x, fx, sigma, budget - calls)
            success_ptr += 1

        return best_x, best_fx
# EVOLVE-BLOCK-END
