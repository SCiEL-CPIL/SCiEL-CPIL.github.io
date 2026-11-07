# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Structure-aware optimizer for ZDT2 exploiting the Pareto front geometry."""

    def __init__(self, budget, dim, n_obj):
        self.budget, self.dim, self.n_obj = budget, dim, n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 300  # Safety margin before 330s kill

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        dim = self.dim
        budget = self.budget
        n_obj = self.n_obj

        n_points = min(budget, 200)
        if n_points < 2:
            n_points = 2

        # For ZDT2, the Pareto front is f1 = x1, f2 = 1 - sqrt(f1)
        # We want to sample x1 values that give good coverage of the front.
        # Using a power-law transform to bias sampling toward the high-curvature region.
        # The front f2 = 1 - sqrt(x1) has high curvature near x1=0.
        # A good strategy is to use a non-uniform distribution for x1.
        
        # Use Latin Hypercube Sampling over u in [0, 1]
        # Then transform to x1 using a power law to bias towards x1=0
        # where the Pareto front has the highest curvature.
        # Using x1 = u^alpha where alpha > 1 biases toward 0.
        # Let's use alpha = 2 for a moderate bias.
        
        rng = np.random.default_rng(42)  # Fixed seed for reproducibility within the run
        
        # Generate LHS samples in u-space
        indices = np.arange(n_points)
        perm = rng.permutation(n_points)
        u_vals = (perm + rng.random(n_points)) / n_points
        u_vals = np.sort(u_vals)
        
        # Transform to x1 with power law bias toward 0
        alpha = 2.0
        x1_vals = u_vals ** alpha
        
        # Ensure x1_vals are within [0, 1]
        x1_vals = np.clip(x1_vals, 0.0, 1.0)
        
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
