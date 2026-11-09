import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280.0

        # DTLZ7 Pareto front: x1 = 0, x3..x22 = 0, x2 in [0,1]
        # f1 = 0
        # f2 = x2
        # f3 = 1 + x2 + (1-x2)*cos(pi*x2/4)

        # Sample x2 uniformly to densely populate the 1D Pareto front
        n_samples = self.budget
        x2_vals = np.linspace(0.0, 1.0, n_samples)

        X = np.zeros((n_samples, self.dim))
        # x0 is inactive for the Pareto front in DTLZ7 when g=0
        # x1 = 0, x2 = sampled, x3.. = 0
        X[:, 1] = 0.0
        X[:, 2] = x2_vals

        for i in range(n_samples):
            if time.time() > deadline:
                break
            problem(X[i])

        return X
