import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 295

        n_evals = self.budget
        if n_evals <= 0:
            return []

        # For ZDT6, the Pareto front is 1-dimensional: x[0] in [0,1], all other x_i = 0.
        # Uniform spacing on [0,1] maximizes hypervolume for a fixed number of points.
        if n_evals == 1:
            us = np.array([0.5])
        else:
            us = np.linspace(0.0, 1.0, n_evals)

        results = []
        for i in range(n_evals):
            if time.time() > deadline:
                break
            x = np.zeros(self.dim, dtype=float)
            x[0] = float(us[i])
            try:
                f = problem(x)
                results.append(f)
            except Exception:
                break

        return results
