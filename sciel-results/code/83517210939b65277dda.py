"""Rewritten optimizer for BBOB f12 (Bent cigar) with deadline guard."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-restart CMA-ES style optimizer with adaptive step sizes."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0  # safety margin before 300s limit
        rng = np.random.default_rng(42)
        dim = self.dim
        budget = self.budget

        best_x = None
        best_f = np.inf
        calls = 0

        def try_update(x, f):
            nonlocal best_x, best_f, calls
            if f < best_f:
                best_x = x.copy()
                best_f = f
            calls += 1

        # Phase 1: Broad exploration with multiple random starts and coarse steps
        # Use a few initial points to find promising basins
        n_initial = 5
        initial_sigma = 3.0

        for i in range(n_initial):
            if calls >= budget or time.time() > deadline:
                break
            x = rng.uniform(-5, 5, dim)
            fx = func(x)
            try_update(x, fx)

        # Phase 2: Adaptive (1+1)-ES with momentum-like behavior
        # Start from the best point found
        if best_x is None:
            best_x = np.zeros(dim)
            best_f = func(best_x)
            calls += 1

        x = best_x.copy()
        fx = best_f
        sigma = 2.0
        sigma_min = 1e-6
        sigma_max = 8.0
        success_rate = 0.5  # target success rate for step size adaptation
        c_sigma = 0.5       # adaptation speed
        chi_n = np.sqrt(dim) * (1 - 1/(4*dim) + 1/(21*dim**2))  # expected norm of standard normal

        # Track consecutive successes for adaptive step size
        successes = 0
        total_steps = 0

        # Phase 3: Main search loop
        while calls < budget and time.time() < deadline:
            # Check if we should stop early if sigma is very small and no improvement
            if sigma < sigma_min:
                break

            # Generate candidate
            direction = rng.standard_normal(dim)
            y = np.clip(x + sigma * direction, -5, 5)
            fy = func(y)
            calls += 1

            if fy < fx:
                # Success
                x = y
                fx = fy
                try_update(x, fx)
                successes += 1
                # Increase sigma based on success rate
                success_rate = 0.9 * success_rate + 0.1
            else:
                # Failure
                success_rate = 0.9 * success_rate + 0.0

            total_steps += 1

            # Adapt sigma using 1/5 rule variant with smoothing
            # Target: maintain ~20% success rate for optimal convergence
            target_success = 0.2
            if success_rate > target_success:
                sigma *= 1.05
            else:
                sigma *= 0.95

            sigma = np.clip(sigma, sigma_min, sigma_max)

            # Periodic restart if stuck (no improvement in last 50 steps)
            if total_steps % 50 == 0 and successes / max(total_steps, 1) < 0.05:
                # Try a restart from a random point near current best
                x = best_x.copy()
                fx = best_f
                sigma = max(sigma, 1.0)
                successes = 0
                total_steps = 0

        # Phase 4: Fine polishing if we have budget remaining
        if calls < budget and time.time() < deadline:
            sigma = 0.01
            for _ in range(min(50, budget - calls)):
                if time.time() > deadline:
                    break
                direction = rng.standard_normal(dim)
                y = np.clip(x + sigma * direction, -5, 5)
                fy = func(y)
                calls += 1
                if fy < fx:
                    x = y
                    fx = fy
                    try_update(x, fx)
                    sigma *= 1.2
                else:
                    sigma *= 0.8
                if sigma < 1e-8:
                    break

        return best_x, best_f
# EVOLVE-BLOCK-END
