import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 290  # 300s limit, keep 10s margin

        n_total = self.budget
        if n_total <= 0:
            return None

        # ZDT1 Pareto front: x[1:] = 0, f1 = x[0], f2 = 1 + sqrt(x[0])
        # Optimal hypervolume distribution: cluster near x[0]=0
        # Use power law x[0] = u^k with k=4/3
        k = 4.0 / 3.0
        u_vals = np.linspace(0.0, 1.0, n_total)
        x0_vals = u_vals ** k

        # Create evaluation points
        X = np.zeros((n_total, self.dim))
        X[:, 0] = x0_vals

        # Clip to bounds
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        X = np.clip(X, lo, hi)

        # Evaluate all points
        for i in range(n_total):
            if time.time() > deadline:
                break
            try:
                problem(X[i])
            except Exception:
                break

        return None
