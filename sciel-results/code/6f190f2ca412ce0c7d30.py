"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-start (1+1)-ES with 1/5 rule, CMA-like final polish, deadline guard."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 285.0  # safety margin before 300s limit
        dim = self.dim
        budget = self.budget

        global_x = None
        global_fx = np.inf
        calls_used = 0

        def check_deadline():
            return time.time() > deadline

        # --- Phase 1: Quick multi-start to find a good region (first ~20% of budget) ---
        n_starts = min(10, max(3, budget // 50))
        start_budget_each = max(5, budget // (n_starts * 4))
        rng_starts = np.random.default_rng(42)

        for s in range(n_starts):
            if calls_used >= budget or check_deadline():
                break

            if s == 0:
                x = np.zeros(dim)
            elif s == 1:
                x = np.full(dim, 0.5)
            elif s == 2:
                x = np.full(dim, -0.5)
            else:
                x = rng_starts.uniform(-4, 4, dim)

            fx = func(x)
            calls_used += 1
            if fx < global_fx:
                global_fx = fx
                global_x = x.copy()

            # Short local search from this start
            sigma = 1.5
            window = 8
            successes = 0
            total = 0
            no_improve = 0
            max_no_improve = 15

            for _ in range(start_budget_each):
                if calls_used >= budget or check_deadline():
                    break

                y = np.clip(x + sigma * rng_starts.standard_normal(dim), -5, 5)
                fy = func(y)
                calls_used += 1

                if fy < fx:
                    x, fx = y, fy
                    successes += 1
                    no_improve = 0
                    if fx < global_fx:
                        global_fx = fx
                        global_x = x.copy()
                else:
                    no_improve += 1

                total += 1
                if total >= window:
                    rate = successes / total
                    if rate > 0.2:
                        sigma *= 1.4
                    elif rate < 0.15:
                        sigma *= 0.7
                    successes = 0
                    total = 0
                sigma = np.clip(sigma, 0.05, 5.0)

                if no_improve >= max_no_improve:
                    break

        # --- Phase 2: Focused (1+1)-ES from global best with restarts ---
        remaining = budget - calls_used
        if remaining > 10 and not check_deadline() and global_x is not None:
            n_es_restarts = min(8, max(2, remaining // 40))
            es_budget_each = max(10, remaining // (n_es_restarts + 1))

            for r in range(n_es_restarts):
                if calls_used >= budget or check_deadline():
                    break

                rng_es = np.random.default_rng(1000 + r * 31)

                if r == 0:
                    x = global_x.copy()
                    fx = global_fx
                else:
                    noise_scale = 1.0 + 0.5 * r
                    x = np.clip(global_x + noise_scale * rng_es.standard_normal(dim), -5, 5)
                    fx = func(x)
                    calls_used += 1
                    if fx < global_fx:
                        global_fx = fx
                        global_x = x.copy()

                sigma = 0.8
                window = 12
                successes = 0
                total = 0
                no_improve = 0
                max_no_improve = 30 + 10 * r

                for _ in range(es_budget_each):
                    if calls_used >= budget or check_deadline():
                        break

                    y = np.clip(x + sigma * rng_es.standard_normal(dim), -5, 5)
                    fy = func(y)
                    calls_used += 1

                    if fy < fx:
                        x, fx = y, fy
                        successes += 1
                        no_improve = 0
                        if fx < global_fx:
                            global_fx = fx
                            global_x = x.copy()
                    else:
                        no_improve += 1

                    total += 1
                    if total >= window:
                        rate = successes / total
                        if rate > 0.2:
                            sigma *= 1.25
                        elif rate < 0.15:
                            sigma *= 0.75
                        successes = 0
                        total = 0
                    sigma = np.clip(sigma, 0.02, 5.0)

                    if no_improve >= max_no_improve:
                        break

        # --- Phase 3: Fine-grained local polish around global best ---
        remaining = budget - calls_used
        if remaining > 5 and not check_deadline() and global_x is not None:
            rng_polish = np.random.default_rng(99999)
            x = global_x.copy()
            fx = global_fx
            sigma = 0.3

            # Adaptive fine search
            window = 10
            successes = 0
            total = 0
            no_improve = 0
            max_no_improve = 40

            for _ in range(remaining):
                if calls_used >= budget or check_deadline():
                    break

                # Mix of small random steps and coordinate-wise nudges
                if rng_polish.random() < 0.7:
                    y = np.clip(x + sigma * rng_polish.standard_normal(dim), -5, 5)
                else:
                    # Coordinate-wise: pick 1-2 coords to nudge
                    y = x.copy()
                    n_coords = 1 if rng_polish.random() < 0.5 else 2
                    coords = rng_polish.choice(dim, size=n_coords, replace=False)
                    for c in coords:
                        y[c] = np.clip(y[c] + sigma * 0.5 * rng_polish.standard_normal(), -5, 5)

                fy = func(y)
                calls_used += 1

                if fy < fx:
                    x, fx = y, fy
                    successes += 1
                    no_improve = 0
                    if fx < global_fx:
                        global_fx = fx
                        global_x = x.copy()
                else:
                    no_improve += 1

                total += 1
                if total >= window:
                    rate = successes / total
                    if rate > 0.25:
                        sigma *= 1.3
                    elif rate < 0.1:
                        sigma *= 0.7
                    successes = 0
                    total = 0
                sigma = np.clip(sigma, 0.005, 2.0)

                if no_improve >= max_no_improve:
                    break

        return global_x, global_fx
# EVOLVE-BLOCK-END
