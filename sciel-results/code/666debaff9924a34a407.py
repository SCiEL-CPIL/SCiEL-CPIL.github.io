"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-restart local search with CMA-ES-like adaptation for Rastrigin f3."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290  # safety margin before 300s limit
        calls_made = 0
        dim = self.dim
        budget = self.budget

        def clip(x):
            return np.clip(x, -5.0, 5.0)

        def safe_call(x):
            nonlocal calls_made
            calls_made += 1
            return func(x)

        # Phase 1: Coarse random sampling to find good starting points
        rng = np.random.default_rng(42)
        n_initial = min(50, budget // 10)
        candidates = []
        for i in range(n_initial):
            if calls_made >= budget or time.time() > deadline:
                break
            x = rng.uniform(-5, 5, dim)
            fx = safe_call(x)
            candidates.append((fx, x))

        # Sort by fitness
        candidates.sort(key=lambda t: t[0])
        best_fx = candidates[0][0]
        best_x = candidates[0][1]

        # Phase 2: Multi-restart local search from top candidates
        n_restarts = min(8, len(candidates))
        # Take top n_restarts, but ensure diversity
        selected = []
        for fx, x in candidates[:n_restarts]:
            # Check diversity: only add if sufficiently different from already selected
            diverse = True
            for _, sx in selected:
                if np.linalg.norm(x - sx) < 1.0:
                    diverse = False
                    break
            if diverse:
                selected.append((fx, x))
            if len(selected) >= n_restarts:
                break

        # If not enough diverse, fill with next best
        if len(selected) < n_restarts:
            for fx, x in candidates:
                if (fx, x) not in selected:
                    selected.append((fx, x))
                    if len(selected) >= n_restarts:
                        break

        # Local search: (1+1)-ES with adaptive sigma
        for fx0, x0 in selected:
            if calls_made >= budget or time.time() > deadline:
                break

            x = x0.copy()
            fx = fx0
            sigma = 1.0
            no_improve = 0
            max_no_improve = 30

            while calls_made < budget and time.time() < deadline:
                # Generate candidate
                y = clip(x + sigma * rng.standard_normal(dim))
                fy = safe_call(y)

                if fy < fx:
                    x = y
                    fx = fy
                    sigma *= 1.1
                    no_improve = 0
                    if fx < best_fx:
                        best_fx = fx
                        best_x = x.copy()
                else:
                    sigma *= 0.9
                    no_improve += 1

                if no_improve >= max_no_improve:
                    break

                # If sigma becomes too small, stop this restart
                if sigma < 0.01:
                    break

        # Phase 3: Fine-tune around the best point found
        if calls_made < budget and time.time() < deadline:
            x = best_x.copy()
            fx = best_fx
            sigma = 0.1
            no_improve = 0

            while calls_made < budget and time.time() < deadline:
                y = clip(x + sigma * rng.standard_normal(dim))
                fy = safe_call(y)

                if fy < fx:
                    x = y
                    fx = fy
                    sigma *= 1.05
                    no_improve = 0
                else:
                    sigma *= 0.9
                    no_improve += 1

                if no_improve >= 50:
                    break
                if sigma < 0.001:
                    break

        return best_x, best_fx
# EVOLVE-BLOCK-END
