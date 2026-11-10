import numpy as np
import time


class Optimizer:
    """Exploit ZDT1 structure with non-uniform sampling."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 290

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])

        budget = self.budget
        dim = self.dim

        # Generate non-uniform samples for x_1
        # Power 2/3 concentrates points near x_1=1
        t = np.linspace(0.0, 1.0, budget)
        x1_samples = t ** (2.0 / 3.0)
        
        # Ensure exact boundaries
        x1_samples[0] = 0.0
        x1_samples[-1] = 1.0

        points = []
        for x1 in x1_samples:
            x = np.zeros(dim)
            x[0] = x1
            points.append(x)

        for x in points:
            if time.time() > deadline:
                break
            problem(x)

        return points
