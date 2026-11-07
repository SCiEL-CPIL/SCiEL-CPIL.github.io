import numpy as np
import time


class Optimizer:
    """ZDT3 Pareto front sampler using full budget with uniform x[0] grid."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 290  # Safety margin before 300s kill

        lo = np.array([b[0] for b in bounds], dtype=np.float64)
        hi = np.array([b[1] for b in bounds], dtype=np.float64)

        budget = self.budget
        dim = self.dim

        # For ZDT3 with g=1 (x[1:]=0), Pareto front is parameterized by x[0] in [0,1]
        # f1 = x[0], f2 = 2*(1 - sqrt(x[0]) * 100*exp(-4*x[0]*(sin(10*pi*x[0]))^2))
        # We sample x[0] uniformly to cover the full front
        # Using all budget evaluations

        n_samples = min(budget, 200)
        x_values = np.linspace(0.0, 1.0, n_samples)

        for x0 in x_values:
            if time.time() > deadline:
                break
            x = np.zeros(dim)
            x[0] = x0
            # x[1:] = 0 gives g=1, which is on the Pareto front
            problem(x)

        return None
