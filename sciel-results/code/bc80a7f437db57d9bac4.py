"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-start adaptive ES with hierarchical budget allocation and refinement."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280  # safety margin before 330s kill

        rng = np.random.default_rng(123)
        calls_used = [0]
        best_f = float('inf')
        best_x = None

        def safe_func(x):
            if time.time() > deadline:
                raise TimeoutError()
            if calls_used[0] >= self.budget:
                raise TimeoutError()
            calls_used[0] += 1
            return func(x)

        def update_best(x, f):
            nonlocal best_f, best_x
            if f < best_f:
                best_f = f
                best_x = x.copy()

        # Phase 1: Broad random sampling to find good basins
        n_initial = 20
        candidates = []
        for i in range(n_initial):
            if calls_used[0] >= self.budget or time.time() > deadline:
                break
            # Mix of broader and centered starts
            if i < n_initial // 2:
                x = rng.uniform(-4, 4, self.dim)
            else:
                x = rng.uniform(-2, 2, self.dim)
            try:
                f = safe_func(x)
            except TimeoutError:
                break
            candidates.append((f, x.copy()))
            update_best(x, f)

        if not candidates:
            x0 = np.zeros(self.dim)
            try:
                f0 = safe_func(x0)
            except TimeoutError:
                return np.zeros(self.dim), float('inf')
            candidates.append((f0, x0.copy()))
            update_best(x0, f0)

        candidates.sort(key=lambda t: t[0])

        # Phase 2: Adaptive ES from top candidates with hierarchical budget allocation
        n_local = min(8, len(candidates))
        remaining_budget = self.budget - calls_used[0]

        if remaining_budget <= 0:
            return best_x, best_f

        # Hierarchical allocation: top 3 get most, next 3 get medium, rest get minimum
        allocations = []
        for i in range(n_local):
            if i < 3:
                alloc = max(30, remaining_budget // 4)
            elif i < 6:
                alloc = max(15, remaining_budget // 8)
            else:
                alloc = max(5, remaining_budget // 16)
            allocations.append(alloc)

        # Adjust to fit within remaining budget
        total_alloc = sum(allocations)
        if total_alloc > remaining_budget:
            scale = remaining_budget / total_alloc
            allocations = [int(a * scale) for a in allocations]
            diff = remaining_budget - sum(allocations)
            allocations[0] += diff

        for i in range(n_local):
            if calls_used[0] >= self.budget or time.time() > deadline:
                break

            x = candidates[i][1].copy()
            f = candidates[i][0]
            sigma = max(0.3, 1.5 / (i + 1))  # larger sigma for better starts
            local_budget = allocations[i]
            no_improve = 0
            patience = 25

            for _ in range(local_budget):
                if calls_used[0] >= self.budget or time.time() > deadline:
                    break

                # Generate 3 candidates per iteration for better exploration
                best_offspring = None
                best_offspring_f = float('inf')

                for _ in range(3):
                    if calls_used[0] >= self.budget or time.time() > deadline:
                        break
                    y = x + sigma * rng.standard_normal(self.dim)
                    y = np.clip(y, -5, 5)
                    try:
                        fy = safe_func(y)
                    except TimeoutError:
                        break
                    if fy < best_offspring_f:
                        best_offspring_f = fy
                        best_offspring = y.copy()

                if best_offspring is None:
                    break

                if best_offspring_f < f:
                    x = best_offspring
                    f = best_offspring_f
                    update_best(x, f)
                    sigma = min(sigma * 1.2, 3.0)
                    no_improve = 0
                else:
                    sigma *= 0.85
                    no_improve += 1

                # Early termination if converged
                if no_improve >= patience or sigma < 1e-4:
                    break

        # Phase 3: Final refinement on the best point found
        if best_x is not None and calls_used[0] < self.budget and time.time() < deadline:
            remaining = self.budget - calls_used[0]
            if remaining > 10:
                x = best_x.copy()
                fx = best_f
                sigma = 0.15
                for _ in range(remaining):
                    if time.time() > deadline:
                        break
                    y = np.clip(x + sigma * rng.standard_normal(self.dim), -5, 5)
                    try:
                        fy = safe_func(y)
                    except TimeoutError:
                        break
                    if fy < fx:
                        x, fx = y, fy
                        sigma = min(sigma * 1.1, 0.5)
                    else:
                        sigma *= 0.95
                    update_best(x, fx)

        return best_x, best_f
# EVOLVE-BLOCK-END
