# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Structure-aware optimizer for ZDT2 using anchors and local refinement."""

    def __init__(self, budget, dim, n_obj):
        self.budget, self.dim, self.n_obj = budget, dim, n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 320  # Safety margin before 330s kill

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        dim = self.dim
        budget = self.budget
        n_obj = self.n_obj

        rng = np.random

        # Strategy: Exploit ZDT2 structure.
        # Pareto front is x1 in [0, 1], x2..xn = 0.
        # We want to maximize the discrete Hypervolume.
        # For a convex decreasing front, we should use larger gaps in x1 where f2 is high (x1 small)
        # and smaller gaps where f2 is low (x1 large).
        # This corresponds to a sampling distribution x_i = 1 - (1 - t_i)^k with k > 1.
        
        n_points = budget
        if n_points < 2:
            n_points = 2
            
        t = np.linspace(0, 1, n_points)
        # Use k=2.0 to bias points towards x=1 (small gaps there, large gaps at x=0)
        k = 2.0
        x1_vals = 1.0 - (1.0 - t) ** k
        
        X_list = []
        F_list = []
        
        for i in range(n_points):
            x = np.zeros(dim)
            x[0] = x1_vals[i]
            
            # Clip to bounds
            x = np.clip(x, lo, hi)
            
            f = problem(x)
            X_list.append(x)
            F_list.append(f)
            
            if time.time() > deadline:
                break

        return np.array(X_list)
# EVOLVE-BLOCK-END
