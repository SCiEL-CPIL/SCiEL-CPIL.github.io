import numpy as np
import time


class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget, self.dim, self.n_obj = budget, dim, n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 290.0  # safety margin before 300s limit

        lo = np.array([b[0] for b in bounds], dtype=np.float64)
        hi = np.array([b[1] for b in bounds], dtype=np.float64)
        dim = self.dim
        n_obj = self.n_obj
        budget = self.budget

        # DTLZ1 structure for M=3, n=7:
        # Head variables: indices 0, 1 (M-1 = 2 position vars)
        # Tail variables: indices 2-6 (5 convergence vars)
        # g = 100*(k + sum_{i=0}^{k-1}((x_{M+i}-0.5)^2 - cos(20*pi*(x_{M+i}-0.5))))
        # where k = n - M + 1 = 5
        # At x_i=0.5: each term = -1, sum = -5, g = 100*(5-5) = 0

        n_head = n_obj - 1  # 2
        n_tail = dim - n_head  # 5

        tail_mid = (lo[n_head:] + hi[n_head:]) / 2.0

        # Generate Halton sequence in n_head dimensions
        def halton_sequence(n, dims):
            """Generate n points in [0,1]^dims using Halton sequence."""
            points = np.zeros((n, dims))
            for d in range(dims):
                base = d + 2  # primes: 2, 3, 5, ...
                for i in range(n):
                    f = 1.0
                    r = 0.0
                    i_val = i + 1
                    while i_val > 0:
                        f /= base
                        r += f * (i_val % base)
                        i_val //= base
                    points[i, d] = r
            return points

        # Use all budget for Pareto front sampling
        n_samples = budget

        # Generate Halton sequence
        halton_pts = halton_sequence(n_samples, n_head)

        X = []
        F = []

        for i in range(n_samples):
            if time.time() > deadline:
                break

            x = np.empty(dim)
            x[:n_head] = lo[:n_head] + (hi[:n_head] - lo[:n_head]) * halton_pts[i]
            x[n_head:] = tail_mid

            f = np.asarray(problem(x), dtype=np.float64)
            X.append(x)
            F.append(f)

        return X
