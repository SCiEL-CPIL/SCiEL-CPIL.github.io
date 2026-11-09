"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Hybrid (1+1)-ES with coordinate descent exploitation and time guard."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280  # 20s safety margin before 300s limit

        rng = np.random.default_rng(0)
        dim = self.dim
        budget = self.budget

        # Phase 1: Global exploration with (1+1)-ES and 1/5 success rule
        x = rng.uniform(-5, 5, dim)
        fx = func(x)
        calls = 1

        successes = 0
        window = 10
        sigma = 2.0
        sigma_min = 1e-6
        sigma_max = 5.0

        # Run (1+1)-ES until we've used ~40% of budget or time runs low
        es_budget = int(budget * 0.4)
        while calls < es_budget:
            if time.time() >= deadline:
                break

            y = np.clip(x + sigma * rng.standard_normal(dim), -5, 5)
            fy = func(y)
            calls += 1

            if fy < fx:
                x, fx = y, fy
                successes += 1
                sigma = min(sigma * 1.05, sigma_max)
            else:
                sigma = max(sigma * 0.95, sigma_min)

            if calls % window == 0:
                if successes >= window / 5.0:
                    sigma = min(sigma * 1.2, sigma_max)
                else:
                    sigma = max(sigma * 0.8, sigma_min)
                successes = 0

        # Phase 2: Coordinate descent exploitation
        # Cyclic coordinate descent with adaptive step size
        if calls < budget and time.time() < deadline - 5:
            # Start with a moderate step size for coordinate descent
            step = max(sigma * 0.5, 0.5)
            step_min = 1e-8
            step_max = 3.0

            # Multiple passes of coordinate descent
            max_cd_passes = 20
            cd_pass = 0
            while calls < budget and cd_pass < max_cd_passes:
                if time.time() >= deadline:
                    break

                improved = False
                for d in range(dim):
                    if calls >= budget:
                        break
                    if time.time() >= deadline:
                        break

                    # Try positive and negative steps along axis d
                    for direction in [1, -1]:
                        if calls >= budget:
                            break

                        y = x.copy()
                        y[d] = np.clip(y[d] + direction * step, -5, 5)
                        fy = func(y)
                        calls += 1

                        if fy < fx:
                            x, fx = y, fy
                            improved = True
                            # Increase step slightly on improvement
                            step = min(step * 1.1, step_max)
                        else:
                            # Decrease step on failure
                            step = max(step * 0.85, step_min)

                cd_pass += 1

                # If no improvement in a full pass, reduce step significantly
                if not improved:
                    step = max(step * 0.5, step_min)
                    if step <= step_min * 1.01:
                        break

        # Phase 3: Final polish with very small random steps
        if calls < budget and time.time() < deadline - 2:
            sigma_polish = max(1e-4, 1e-6)
            while calls < budget:
                if time.time() >= deadline:
                    break
                y = np.clip(x + sigma_polish * rng.standard_normal(dim), -5, 5)
                fy = func(y)
                calls += 1
                if fy < fx:
                    x, fx = y, fy
                    sigma_polish = min(sigma_polish * 1.05, 1e-3)
                else:
                    sigma_polish = max(sigma_polish * 0.95, 1e-10)

        return x, fx
# EVOLVE-BLOCK-END
