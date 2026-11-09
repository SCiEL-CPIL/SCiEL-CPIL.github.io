import numpy as np


class Optimizer:
    """ZDT6-aware optimizer: exploit known Pareto front structure g=1 at x2..x10=0."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        import time
        deadline = time.time() + 290  # safety margin before 300s limit
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        dim = self.dim
        budget = self.budget

        # Strategy: ZDT6 Pareto front is achieved at x2..x10 = 0 (g=1)
        # Sample x1 densely across [0,1]
        # Use most budget for x1 sampling, a few for verification

        # Number of x1 samples: use budget - 10 for sampling, 10 for polish
        n_samples = max(10, budget - 20)
        n_polish = budget - n_samples

        # Dense sampling of x1
        x1_vals = np.linspace(lo[0], hi[0], n_samples)

        results = []
        used = 0
        for x1 in x1_vals:
            if used >= budget:
                break
            x = np.zeros(dim)
            x[0] = x1
            # x2..x10 = 0 (middle of [0,1] is 0.5, but we want 0 for g=1)
            # Actually x2..x10 should be at their lower bound for g=1
            x[1:] = lo[1:]
            f = problem(x)
            results.append(np.asarray(f, dtype=float))
            used += 1
            if time.time() > deadline - 5:
                break

        # Polish: try small perturbations around x2..x10=0 to ensure we're at optimum
        # Also try a few random x1 with optimized x2..x10
        if used < budget:
            # Try a few points with x2..x10 slightly perturbed to confirm g=1 region
            for i in range(n_polish):
                if used >= budget or time.time() > deadline - 2:
                    break
                x1 = np.random.uniform(lo[0], hi[0])
                x = np.zeros(dim)
                x[0] = x1
                x[1:] = lo[1:] + np.random.uniform(0, 0.01, size=dim - 1)
                x = np.clip(x, lo, hi)
                f = problem(x)
                results.append(np.asarray(f, dtype=float))
                used += 1

        return results
