"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """(1+1)-ES with adaptive sigma, coordinate descent polish, and deadline guard."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0  # safety margin before 300s limit

        rng = np.random.default_rng(42)
        dim = self.dim
        budget = self.budget

        # Start at center, which is often closer to the minimum in [-4,4]^5
        x = np.zeros(dim)
        fx = func(x)
        best_x = x.copy()
        best_fx = fx

        sigma = 1.0
        sigma_min = 1e-8
        sigma_max = 5.0

        # Adaptive sigma based on success rate (1/5 rule variant)
        # Track successes in a window
        window_size = 10
        successes = 0
        call_count = 1

        def coord_descent_polish(x, fx, max_iter=50):
            """Coordinate descent polish exploiting separable structure."""
            nonlocal best_x, best_fx
            for _ in range(max_iter):
                if time.time() > deadline - 10.0:
                    return x, fx
                improved = False
                for i in range(dim):
                    # Try small perturbations along each axis
                    for delta in [0.5, 0.1, 0.01, -0.5, -0.1, -0.01]:
                        y = x.copy()
                        y[i] = np.clip(y[i] + delta, -5, 5)
                        fy = func(y)
                        call_count_local = 1
                        if fy < fx:
                            x, fx = y, fy
                            improved = True
                            if fx < best_fx:
                                best_fx = fx
                                best_x = x.copy()
                            break
                if not improved:
                    break
            return x, fx

        for _ in range(budget - 1):
            # Deadline check
            if time.time() > deadline - 5.0:
                # Do a final polish if time permits
                if time.time() < deadline - 2.0:
                    x, fx = coord_descent_polish(x, fx, max_iter=20)
                break

            # Generate candidate
            y = np.clip(x + sigma * rng.standard_normal(dim), -5, 5)
            fy = func(y)
            call_count += 1

            if fy < fx:
                successes += 1
                x, fx = y, fy
                if fx < best_fx:
                    best_fx = fx
                    best_x = x.copy()
                # Adaptive sigma: increase on success
                sigma = min(sigma * 1.2, sigma_max)
            else:
                # Adaptive sigma: decrease on failure
                sigma = max(sigma * 0.85, sigma_min)

            # Periodic coordinate descent polish every 50 calls
            if call_count % 50 == 0 and call_count < budget - 20:
                x, fx = coord_descent_polish(x, fx, max_iter=30)
                if fx < best_fx:
                    best_fx = fx
                    best_x = x.copy()
                # Reset sigma based on improvement
                if fx < best_fx + 1e-6:
                    sigma = max(sigma * 0.5, sigma_min)

            # Window-based sigma adaptation
            if successes >= window_size:
                # High success rate: increase sigma more aggressively
                if successes / max(call_count - 1, 1) > 0.5:
                    sigma = min(sigma * 1.5, sigma_max)
                successes = 0

        # Final high-precision polish pass
        if time.time() < deadline - 2.0:
            x, fx = coord_descent_polish(x, fx, max_iter=50)
            if fx < best_fx:
                best_fx = fx
                best_x = x.copy()

        return best_x, best_fx
# EVOLVE-BLOCK-END
