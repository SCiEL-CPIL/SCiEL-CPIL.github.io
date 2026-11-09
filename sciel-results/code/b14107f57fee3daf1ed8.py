"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """
    Multi-start (1+1)-ES with 1/5 success rule for BBOB f18 (Schaffers F7).
    - Brief random sampling to identify promising regions.
    - Each local search uses the 1/5 success rule for principled sigma adaptation.
    - Periodic restarts from the global best with escalating step sizes.
    - Final polish phase spends remaining budget on tight local refinement.
    - Time guard to stay under the 330s wall-clock limit.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 320.0  # 10s safety margin

        rng = np.random.default_rng(42)
        dim = self.dim
        budget = self.budget

        # Phase 1: Random sampling to find a good initial point
        n_samples = min(30, budget // 15)
        best_x = None
        best_fx = float('inf')

        for i in range(n_samples):
            x = rng.uniform(-5, 5, dim)
            fx = func(x)
            if fx < best_fx:
                best_fx = fx
                best_x = x.copy()

        calls_used = n_samples

        # Phase 2: Multi-start (1+1)-ES with 1/5 success rule
        # Each "run" is a local (1+1)-ES with 1/5 rule, restarted when stalled
        best_x = best_x.copy()
        best_fx = best_fx

        # 1/5 success rule parameters
        sigma = 1.5
        successes = 0
        total_trials = 0
        window_size = 10  # evaluate rule every 10 trials
        stall_count = 0
        max_stall = 15
        restart_sigma = 2.5

        while calls_used < budget:
            # Time check
            if time.time() > deadline - 5.0:
                break

            # Generate candidate using 1/5 success rule adapted sigma
            y = best_x + sigma * rng.standard_normal(dim)
            y = np.clip(y, -5, 5)
            fy = func(y)
            calls_used += 1
            total_trials += 1

            if fy < best_fx:
                best_x = y.copy()
                best_fx = fy
                successes += 1
                stall_count = 0
            else:
                stall_count += 1

            # Apply 1/5 success rule every window_size trials
            if total_trials >= window_size:
                rate = successes / total_trials
                if rate > 0.2:
                    sigma *= 1.5
                elif rate < 0.2:
                    sigma *= (1.0 / 1.5)
                # Clamp sigma to reasonable range
                sigma = np.clip(sigma, 0.05, 10.0)
                successes = 0
                total_trials = 0

            # Restart if stalled: jump from global best with larger step
            if stall_count >= max_stall:
                # Reset sigma to a larger value for exploration
                sigma = max(restart_sigma, sigma * 2.0)
                restart_sigma *= 1.3
                stall_count = 0
                max_stall = min(max_stall + 5, 50)

        # Phase 3: Final polish with very small sigma
        if calls_used < budget and time.time() < deadline - 2.0:
            sigma = 0.1
            successes = 0
            total_trials = 0
            while calls_used < budget:
                if time.time() > deadline - 1.0:
                    break
                y = best_x + sigma * rng.standard_normal(dim)
                y = np.clip(y, -5, 5)
                fy = func(y)
                calls_used += 1
                total_trials += 1

                if fy < best_fx:
                    best_x = y.copy()
                    best_fx = fy
                    successes += 1

                if total_trials >= window_size:
                    rate = successes / total_trials
                    if rate > 0.2:
                        sigma *= 1.3
                    elif rate < 0.2:
                        sigma *= (1.0 / 1.3)
                    sigma = np.clip(sigma, 0.01, 0.5)
                    successes = 0
                    total_trials = 0

        return best_x, best_fx
# EVOLVE-BLOCK-END
