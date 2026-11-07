"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """(1+1)-ES with adaptive sigma, multi-start, and local refinement."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        rng = np.random.default_rng(42)
        dim = self.dim
        budget = self.budget
        deadline = time.time() + 280.0  # safety margin before 330s kill

        # Track best
        best_x = None
        best_f = np.inf
        calls = 0

        def eval_func(x):
            nonlocal calls, best_x, best_f
            if calls >= budget:
                return np.inf
            calls += 1
            try:
                val = float(func(x))
            except:
                return np.inf
            if np.isnan(val) or np.isinf(val):
                return np.inf
            if val < best_f:
                best_f = val
                best_x = x.copy()
            return val

        # Phase 1: Global Exploration with multiple random starts
        # Use about 30% of budget for exploration
        explore_budget = max(30, budget // 4)
        n_starts = min(10, explore_budget // 5)
        per_start_budget = explore_budget // n_starts

        candidates = []
        for i in range(n_starts):
            if calls >= explore_budget:
                break
            x = rng.uniform(-5, 5, dim)
            fx = eval_func(x)
            candidates.append((fx, x.copy()))

        # Sort by function value
        candidates.sort(key=lambda t: t[0])

        # Phase 2: Local refinement using (1+1)-ES on top candidates
        remaining_budget = budget - calls - 10  # reserve 10 calls for safety
        n_refine = min(3, len(candidates))
        per_refine_budget = max(10, remaining_budget // n_refine) if n_refine > 0 else 0

        for i in range(n_refine):
            if calls >= budget - 10 or time.time() > deadline:
                break
            x0 = candidates[i][1]
            fx0 = eval_func(x0)  # re-evaluate to ensure it's counted
            
            # (1+1)-ES refinement
            sigma = 1.0
            max_iter = per_refine_budget
            for _ in range(max_iter):
                if calls >= budget - 10 or time.time() > deadline:
                    break
                y = np.clip(x0 + sigma * rng.standard_normal(dim), -5, 5)
                fy = eval_func(y)
                if fy < fx0:
                    x0, fx0 = y, fy
                    sigma *= 1.3
                else:
                    sigma *= 0.85
                    if sigma < 1e-6:
                        break

        # Phase 3: Final local search around the best point found
        if best_x is not None and calls < budget - 5 and time.time() < deadline:
            x0 = best_x.copy()
            fx0 = eval_func(x0)
            sigma = 0.5
            for _ in range(min(50, budget - calls - 5)):
                if calls >= budget - 5 or time.time() > deadline:
                    break
                y = np.clip(x0 + sigma * rng.standard_normal(dim), -5, 5)
                fy = eval_func(y)
                if fy < fx0:
                    x0, fx0 = y, fy
                    sigma *= 1.2
                else:
                    sigma *= 0.8
                    if sigma < 1e-7:
                        break

        if best_x is None:
            best_x = np.zeros(dim)
            best_f = func(best_x)

        return best_x, best_f
# EVOLVE-BLOCK-END
