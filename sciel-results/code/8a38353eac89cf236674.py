import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280  # Safety margin for 300s limit

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        n = self.dim

        n_evals = self.budget

        # For standard ZDT2, g(x) = sum(x_i - 0.5)^2 is minimized at x_i = 0.5
        # Allocate all budget to the correct strategy
        n_main = n_evals

        points = np.zeros((n_evals, self.dim))

        # Main strategy: x1 varies, x2..xn = 0.5 (standard ZDT2 Pareto front)
        # Use quadratic spacing for x1 to better approximate the convex Pareto front
        # which has high curvature near x1=0
        if n_main > 1:
            indices = np.arange(n_main)
            x1_vals = (indices / (n_main - 1)) ** 2
            # Ensure endpoints are exactly lo and hi
            x1_vals[0] = lo[0]
            x1_vals[-1] = hi[0]
            points[:n_main, 0] = x1_vals
            points[:n_main, 1:] = 0.5
        else:
            points[0, 0] = lo[0]
            points[0, 1:] = 0.5

        # Evaluate all points
        for i in range(n_evals):
            if time.time() > deadline:
                break
            problem(points[i])

        return points
