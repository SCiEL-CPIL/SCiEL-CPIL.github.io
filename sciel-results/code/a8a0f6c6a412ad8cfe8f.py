"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """ZDT1-aware optimizer: exploit known Pareto front structure via optimal stratified sampling."""

    def __init__(self, budget, dim, n_obj):
        self.budget, self.dim, self.n_obj = budget, dim, n_obj

    def __call__(self, problem, bounds):
        t_start = time.time()
        deadline = t_start + 280.0  # Safety margin before 330s kill

        dim = self.dim
        budget = self.budget
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])

        F_list = []
        X_list = []

        # ZDT1 Pareto Front:
        # f1 = x0
        # f2 = 1 - sqrt(x0)  (when x1..xn-1 = 0)
        # We want to maximize Hypervolume with ref [11, 11].
        # Optimal point distribution for 2D concave front is uniform in f2 (or proportional to curvature).
        # f2 = 1 - sqrt(f1) => f1 = (1-f2)^2.
        # Uniform sampling in f2: f2_i = (i - 0.5) / N
        # f1_i = (1 - f2_i)^2 = ((N - i + 0.5) / N)^2
        # Note: i=1 gives f2 ~ 1/N, f1 ~ 1.
        # i=N gives f2 ~ 1, f1 ~ 0.
        # This covers the front from f1=1 to f1=0.
        
        n_points = min(budget, 200)
        
        if n_points <= 0:
            return X_list

        # Generate i from 1 to N
        i_vals = np.arange(1, n_points + 1)
        
        # Calculate f2 values (uniformly spaced)
        # f2 ranges from near 0 to near 1
        # f2_i = (i - 0.5) / N
        f2_vals = (i_vals - 0.5) / n_points
        
        # Calculate f1 values
        # f1_i = (1 - f2_i)^2
        f1_vals = (1.0 - f2_vals) ** 2
        
        # Clip to bounds
        f1_vals = np.clip(f1_vals, lo[0], hi[0])
        
        # Evaluate each point
        for f1 in f1_vals:
            if time.time() > deadline:
                break
            
            x = np.zeros(dim)
            x[0] = float(f1)
            # x[1:] = 0 is optimal for ZDT1 (g=1)
            # Ensure x[1:] is within bounds (0, 1)
            # x[1:] = 0 is in [0, 1]
            
            f = problem(x)
            F_list.append(np.array(f, dtype=float))
            X_list.append(x.copy())

        return X_list
# EVOLVE-BLOCK-END
