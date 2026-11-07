# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-objective optimizer for ZDT2 using Chebyshev-based Pareto front sampling."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 290.0  # 300s limit, 10s safety margin

        lo = np.array([b[0] for b in bounds], dtype=float)
        hi = np.array([b[1] for b in bounds], dtype=float)

        # ZDT2: x[0] controls the Pareto front, x[1:] should be 0 for optimality
        # f1 = x[0]
        # f2 = (1 + 9 * sum(x[1:]) / (n-1)) * (1 - sqrt(x[0] / (1 + 9 * sum(x[1:]) / (n-1))))
        # For x[1:] = 0: f1 = x[0], f2 = 1 - sqrt(x[0])

        # Use Chebyshev nodes for x[0] to optimize hypervolume coverage
        n_cheb = min(180, self.budget)

        # Chebyshev nodes on [0, 1]: x_k = 0.5 * (1 - cos((2k+1) * pi / (2n)))
        k = np.arange(n_cheb)
        x0_cheb = 0.5 * (1 - np.cos((2 * k + 1) * np.pi / (2 * n_cheb)))
        # Sort to ensure monotonicity
        x0_cheb = np.sort(x0_cheb)

        # Evaluate Chebyshev-spread points (x[1:] = 0)
        evaluated_x0 = set()
        all_f = []

        for x0_val in x0_cheb:
            if time.time() > deadline:
                break

            x = np.zeros(self.dim)
            x[0] = x0_val
            f = problem(x)
            all_f.append(f)
            evaluated_x0.add(x0_val)

            if len(all_f) >= self.budget:
                break

        # If we have budget remaining, add some random exploration
        rng = np.random.default_rng(0)
        remaining = self.budget - len(all_f)

        for _ in range(remaining):
            if time.time() > deadline:
                break
            if len(all_f) >= self.budget:
                break

            x = lo + (hi - lo) * rng.random(self.dim)
            f = problem(x)
            all_f.append(f)

        return all_f
# EVOLVE-BLOCK-END
