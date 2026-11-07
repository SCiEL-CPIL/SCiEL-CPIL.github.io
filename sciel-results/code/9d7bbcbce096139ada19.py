"""Optimized Weierstrass (f16) black-box optimizer."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-start (1+1)-ES with 1/5-rule adaptation, two-phase polish, and deadline guard."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        start_time = time.time()
        deadline = start_time + 285.0  # safety margin before 300s wall-clock
        rng = np.random.default_rng(42)

        # --- Multi-start initialization ---
        # Evaluate center and several diverse random points to find a good basin
        candidates = [np.zeros(dim)]
        for _ in range(6):
            candidates.append(rng.uniform(-4, 4, dim))

        best_x = None
        best_f = np.inf
        for c in candidates:
            c_clipped = np.clip(c, -5, 5)
            fc = func(c_clipped)
            if fc < best_f:
                best_f = fc
                best_x = c_clipped.copy()

        calls_made = len(candidates)

        # --- Phase 1: (1+1)-ES main search with 1/5-rule adaptation ---
        # Use 75% of budget for main search
        main_budget = int(budget * 0.75)
        sigma = 1.5
        window_size = 15
        successes = []

        for i in range(main_budget):
            if calls_made >= budget:
                break
            if time.time() >= deadline:
                break

            # Generate trial point
            z = rng.standard_normal(dim)
            trial_x = np.clip(best_x + sigma * z, -5, 5)
            trial_f = func(trial_x)
            calls_made += 1

            if trial_f < best_f:
                best_f = trial_f
                best_x = trial_x.copy()
                successes.append(1)
            else:
                successes.append(0)

            # Adapt sigma using 1/5-rule on sliding window
            if len(successes) >= window_size:
                recent_successes = successes[-window_size:]
                success_rate = sum(recent_successes) / window_size
                if success_rate > 0.3:
                    sigma *= 1.10
                elif success_rate < 0.15:
                    sigma *= 0.80
                else:
                    sigma *= 0.98
                # Clamp sigma
                sigma = max(1e-6, min(sigma, 10.0))

        # --- Phase 2: High-precision polish with smaller steps ---
        # Use remaining budget for fine-grained local search
        remaining = budget - calls_made
        if remaining > 0 and time.time() < deadline:
            sigma_polish = max(sigma * 0.05, 1e-4)
            polish_window = 5
            polish_successes = []

            for j in range(remaining):
                if calls_made >= budget:
                    break
                if time.time() >= deadline:
                    break

                z = rng.standard_normal(dim)
                trial_x = np.clip(best_x + sigma_polish * z, -5, 5)
                trial_f = func(trial_x)
                calls_made += 1

                if trial_f < best_f:
                    best_f = trial_f
                    best_x = trial_x.copy()
                    polish_successes.append(1)
                else:
                    polish_successes.append(0)

                # Adapt polish sigma more aggressively
                if len(polish_successes) >= polish_window:
                    rate = sum(polish_successes[-polish_window:]) / polish_window
                    if rate > 0.4:
                        sigma_polish *= 1.08
                    elif rate < 0.1:
                        sigma_polish *= 0.70
                    else:
                        sigma_polish *= 0.92
                    sigma_polish = max(1e-9, min(sigma_polish, 1.0))

        return best_x, best_f
# EVOLVE-BLOCK-END
