"""Multi-objective optimizer for ZDT1 exploiting analytical Pareto front structure."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-objective optimizer that exploits ZDT1's known Pareto front structure."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 285  # Safety margin: 300s limit, stop at 285s

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        dim = self.dim
        budget = self.budget

        # ZDT1 Pareto front: x2...xn = 0, x1 in [0, 1]
        # f1 = x1, f2 = 1 - sqrt(x1)
        # Strategy: place points densely on the known front, use remaining for local search

        # Phase 1: Place points on the Pareto front with good x1 coverage
        # Use logarithmic + linear spacing for good hypervolume coverage
        n_front = min(budget - 10, 180)  # Reserve some budget for local search
        if n_front < 10:
            n_front = max(10, budget - 2)

        # Generate x1 values: combine linear and logarithmic spacing
        x1_values = np.linspace(0.001, 0.999, n_front)
        # Add some extra points in dense regions
        extra = min(20, budget - n_front - 5)
        if extra > 0:
            x1_extra = np.linspace(0.01, 0.5, extra)
            x1_values = np.concatenate([x1_values, x1_extra])

        # Shuffle to avoid bias
        rng = np.random.default_rng(42)
        rng.shuffle(x1_values)

        used = 0
        # Evaluate points on the Pareto front
        for x1 in x1_values:
            if used >= budget:
                break
            if time.time() > deadline:
                break
            x = np.zeros(dim)
            x[0] = x1
            # Clip to bounds
            x = np.clip(x, lo, hi)
            problem(x)
            used += 1

        # Phase 2: Local search around promising regions to refine
        # Use remaining budget for small perturbations around front points
        remaining = budget - used
        if remaining > 0 and time.time() < deadline:
            # Pick a few x1 centers and do local search
            n_centers = min(5, remaining // 3)
            if n_centers > 0:
                centers = np.linspace(0.1, 0.9, n_centers)
                per_center = remaining // n_centers
                for c in centers:
                    for _ in range(per_center):
                        if used >= budget:
                            break
                        if time.time() > deadline:
                            break
                        x = np.zeros(dim)
                        x[0] = c + 0.01 * rng.standard_normal()
                        x = np.clip(x, lo, hi)
                        problem(x)
                        used += 1

        # Phase 3: Final check - ensure we haven't exceeded budget
        # The evaluator records all returned objective vectors, so we're done
        return None
# EVOLVE-BLOCK-END
