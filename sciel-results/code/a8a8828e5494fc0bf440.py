"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """(1+1)-ES with 1/5 success rule, multi-start, and time guard."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        start_time = time.time()
        deadline = start_time + 275  # Safety margin for 300s limit

        rng = np.random.default_rng(42)

        best_x = None
        best_f = np.inf

        def safe_func(x):
            try:
                v = func(x)
                if np.isfinite(v):
                    return v
            except Exception:
                pass
            return 1e10

        # Phase 1: Multi-start with diverse initializations (60% of budget)
        n_starts = 6
        phase1_budget = int(budget * 0.6)
        per_budget = max(15, phase1_budget // n_starts)

        for start_i in range(n_starts):
            if time.time() > deadline:
                break

            # Diverse starting strategies
            if start_i == 0:
                x = rng.uniform(-5, 5, dim)
            elif start_i == 1:
                x = np.zeros(dim)
            elif start_i == 2:
                x = rng.uniform(-3, 3, dim)
            elif start_i == 3:
                # Perturb the current best if available
                if best_x is not None:
                    x = np.clip(best_x + 2.0 * rng.standard_normal(dim), -5, 5)
                else:
                    x = rng.uniform(-5, 5, dim)
            elif start_i == 4:
                # Large random perturbation
                x = rng.uniform(-5, 5, dim)
            else:
                # Mix: half random, half perturbed best
                if best_x is not None:
                    x = 0.5 * rng.uniform(-5, 5, dim) + 0.5 * best_x
                    x = np.clip(x, -5, 5)
                else:
                    x = rng.uniform(-5, 5, dim)

            fx = safe_func(x)
            if fx < best_f:
                best_f = fx
                best_x = x.copy()

            sigma = 2.0
            success_count = 0
            window = 10

            for step in range(per_budget - 1):
                if time.time() > deadline:
                    break

                y = np.clip(x + sigma * rng.standard_normal(dim), -5, 5)
                fy = safe_func(y)

                if fy < fx:
                    x, fx = y, fy
                    sigma *= 1.25
                    success_count += 1
                    if fx < best_f:
                        best_f = fx
                        best_x = x.copy()
                else:
                    sigma *= 0.85

                # Adaptive sigma based on success rate over sliding window
                if step >= window:
                    # Rough success rate estimate
                    pass

                if sigma < 1e-10:
                    break
                if sigma > 10.0:
                    sigma = 10.0

        # Phase 2: Local refinement around best point (25% of budget)
        phase2_budget = int(budget * 0.25)
        if best_x is not None and time.time() < deadline:
            x = best_x.copy()
            fx = best_f
            sigma = 1.0

            for _ in range(phase2_budget):
                if time.time() > deadline:
                    break

                y = np.clip(x + sigma * rng.standard_normal(dim), -5, 5)
                fy = safe_func(y)

                if fy < fx:
                    x, fx = y, fy
                    sigma *= 1.15
                    if fx < best_f:
                        best_f = fx
                        best_x = x.copy()
                else:
                    sigma *= 0.92

                if sigma < 1e-12:
                    break

        # Phase 3: High-precision final polish (15% of budget)
        phase3_budget = budget - phase1_budget - phase2_budget
        if phase3_budget > 5 and best_x is not None and time.time() < deadline:
            x = best_x.copy()
            fx = best_f
            sigma = 0.3

            for _ in range(phase3_budget):
                if time.time() > deadline:
                    break

                y = np.clip(x + sigma * rng.standard_normal(dim), -5, 5)
                fy = safe_func(y)

                if fy < fx:
                    x, fx = y, fy
                    sigma *= 1.1
                    if fx < best_f:
                        best_f = fx
                        best_x = x.copy()
                else:
                    sigma *= 0.95

                if sigma < 1e-14:
                    break

        # Ensure we've made at least one call
        if best_x is None:
            x = np.zeros(dim)
            fx = safe_func(x)
            best_x = x.copy()
            best_f = fx

        return best_x, best_f
# EVOLVE-BLOCK-END
