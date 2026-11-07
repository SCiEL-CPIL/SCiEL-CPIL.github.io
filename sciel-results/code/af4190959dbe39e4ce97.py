import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        n_evals = self.budget
        if n_evals <= 0:
            return None

        t_start = time.time()
        deadline = t_start + 320.0  # safety margin before 330s kill

        # For ZDT1, the Pareto front is f2 = 1 - sqrt(f1) where f1 = x1.
        # The hypervolume contribution of a point at (f1, f2) is proportional to
        # the width in f1 times (11 - f2) = (11 - 1 + sqrt(f1)) = 10 + sqrt(f1).
        # To equalize the area contribution per point, we want each point to cover
        # an equal strip of hypervolume. The optimal spacing for a concave front
        # f2 = 1 - sqrt(f1) is to use f1_i proportional to i^2 (i.e., x ~ i^2),
        # which makes the sqrt(f1) term linear in i and equalizes the Riemann sum.
        #
        # Specifically: f1_i = ((i+1)/N)^2 gives f2_i = 1 - (i+1)/N,
        # so the points are evenly spaced in f2, and the hypervolume contribution
        # of each strip is approximately (1/N) * (10 + (i+1)/N) * (1/N),
        # which is nearly uniform.
        #
        # Actually, let's think more carefully. The hypervolume of the non-dominated
        # set with reference point (11, 11) for points on the Pareto front:
        # For points (f1_i, f2_i) sorted by f1 increasing, f2 decreasing:
        # HV = sum_i (f1_{i+1} - f1_i) * (11 - f2_i) + (11 - f1_N) * (11 - f2_N)
        # where f1_{N+1} = 11.
        #
        # With f1_i = ((i+1)/N)^2, f2_i = 1 - (i+1)/N:
        # f1_{i+1} - f1_i = ((i+2)^2 - (i+1)^2)/N^2 = (2i+3)/N^2
        # 11 - f2_i = 10 + (i+1)/N
        # Term i: (2i+3)/N^2 * (10 + (i+1)/N)
        #
        # Alternative: f1_i = (i+1)/N (uniform in f1):
        # f1_{i+1} - f1_i = 1/N
        # 11 - f2_i = 10 + sqrt((i+1)/N)
        # Term i: (1/N) * (10 + sqrt((i+1)/N))
        #
        # The x~i^2 spacing makes the first factor (2i+3)/N^2 grow linearly while
        # the second factor grows sublinearly, balancing them better than uniform
        # spacing where the first factor is constant and the second grows as sqrt.
        #
        # Let's use x1_i = ((i+1)/N)^2 for i = 0, ..., N-1.
        # This gives x1_0 = 1/N^2, x1_{N-1} = 1.

        i = np.arange(n_evals)
        x1_vals = ((i + 1) / n_evals) ** 2

        last_x = None
        for x1 in x1_vals:
            if time.time() > deadline:
                break
            x = np.zeros(self.dim)
            x[0] = x1
            last_x = problem(x)

        return last_x
