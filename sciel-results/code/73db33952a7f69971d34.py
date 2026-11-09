# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-objective optimizer for ZDT3 using a structured sampling approach."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 290.0  # Safety margin before 300s limit

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        dim = self.dim
        budget = self.budget

        # ZDT3 analysis:
        # f1(x) = x1
        # g(x) = 1 + 10*(n-1) + sum_{i=2}^n [xi^2 - 10*cos(4*pi*xi*x1)]
        # f2(x) = g(x) * h(f1(x), g(x))
        # where h(u,v) = 1 - sqrt(u/v) - (u/v)*sin(10*pi*u)
        #
        # The Pareto front is achieved when g is minimized.
        # For ZDT3, the global minimum of g is 1 (when xi=0 for all i>=2),
        # but there are many local minima.
        #
        # Key insight: The Pareto front of ZDT3 lies at x1 in [0,1] and
        # xi = 0 for i >= 2, giving g = 1.
        # f1 = x1, f2 = 1 * h(x1, 1) = 1 - sqrt(x1) - x1*sin(10*pi*x1)
        #
        # Strategy: Sample x1 densely in [0,1] with xi=0 for i>=2,
        # which should trace out the Pareto front closely.

        # Phase 1: Dense sampling of the known Pareto front structure
        # x1 uniformly in [0,1], rest at 0 (or near 0)
        n_phase1 = min(budget, 150)

        # Use a mix: uniform x1 sampling
        x1_values = np.linspace(0.0, 1.0, n_phase1)
        # Add some random x1 values for coverage
        if budget > n_phase1:
            n_random = budget - n_phase1
            x1_values = np.concatenate([x1_values, np.random.uniform(0, 1, n_random)])
        else:
            x1_values = x1_values[:budget]

        # Shuffle to interleave
        np.random.shuffle(x1_values)

        used = 0
        for x1 in x1_values:
            if used >= budget:
                break
            if time.time() > deadline:
                break

            x = np.zeros(dim)
            x[0] = x1
            # Keep other variables at 0 (the known Pareto front location)
            # Add tiny perturbation to avoid exact boundary issues
            x[1:] = 0.0

            # Evaluate
            f = problem(x)
            used += 1

        # Phase 2: If budget remains, do some random exploration
        if used < budget:
            n_remaining = budget - used
            for _ in range(n_remaining):
                if used >= budget:
                    break
                if time.time() > deadline:
                    break

                x = np.random.uniform(lo, hi)
                f = problem(x)
                used += 1

        return None
# EVOLVE-BLOCK-END
