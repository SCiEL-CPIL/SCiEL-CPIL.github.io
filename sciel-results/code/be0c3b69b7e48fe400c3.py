"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


class Optimizer:
    """
    Multi-start local search with global exploration phase.
    Phase 1: Random sampling to identify promising regions.
    Phase 2: Nelder-Mead local refinement from multiple starting points.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 270.0  # Safety margin before 300s kill

        rng = np.random.default_rng(42)
        dim = self.dim
        budget = self.budget

        # Track best solution
        best_x = None
        best_f = np.inf
        eval_count = 0

        def evaluate(x):
            nonlocal best_x, best_f, eval_count
            x = np.clip(x, -5, 5)
            fx = func(x)
            eval_count += 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            return fx

        # Phase 1: Global exploration with random sampling
        n_explore = min(50, budget // 4)
        candidates = []

        for i in range(n_explore):
            if eval_count >= budget or time.time() > deadline:
                break
            x = rng.uniform(-5, 5, dim)
            fx = evaluate(x)
            candidates.append((fx, x))

        # Sort candidates by fitness
        candidates.sort(key=lambda t: t[0])

        # Phase 2: Local refinement with Nelder-Mead from top candidates
        n_starts = min(10, len(candidates))
        used_budget = eval_count
        remaining = budget - used_budget

        if remaining > 10:
            # Distribute remaining budget among starts
            budget_per_start = max(5, remaining // n_starts)

            for i in range(n_starts):
                if eval_count >= budget or time.time() > deadline:
                    break

                x0 = candidates[i][1]

                # Use a counter to track function calls within minimize
                calls = [0]
                max_calls = budget_per_start

                def objective(x):
                    nonlocal eval_count
                    calls[0] += 1
                    if calls[0] > max_calls or eval_count >= budget or time.time() > deadline:
                        raise ValueError("Budget exceeded")
                    return evaluate(x)

                try:
                    result = minimize(
                        objective,
                        x0,
                        method='Nelder-Mead',
                        options={
                            'maxiter': max_calls,
                            'xatol': 1e-6,
                            'fatol': 1e-8,
                            'adaptive': True
                        }
                    )
                    if result.x is not None:
                        fx = evaluate(result.x)
                except (ValueError, Exception):
                    pass

        # Phase 3: Final polish with a few more random perturbations around best
        remaining = budget - eval_count
        if remaining > 5 and best_x is not None:
            for _ in range(min(20, remaining)):
                if eval_count >= budget or time.time() > deadline:
                    break
                x = best_x + rng.normal(0, 0.1, dim)
                fx = evaluate(x)

        return best_x, best_f
# EVOLVE-BLOCK-END
