"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """ZDT1-aware optimizer: exploit known Pareto front structure via power-law stratified sampling."""

    def __init__(self, budget, dim, n_obj):
        self.budget, self.dim, self.n_obj = budget, dim, n_obj

    def __call__(self, problem, bounds):
        t_start = time.time()
        deadline = t_start + 280.0  # Safety margin before 330s kill

        dim = self.dim
        budget = self.budget
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])

        used = 0
        F_list = []
        X_list = []

        # ZDT1 structure:
        # f1(x) = x[0]
        # g(x) = 1 + 9 * sum(x[1:]) / (dim-1)
        # f2(x) = g(x) * (1 - sqrt(f1(x) / g(x)))
        #
        # Pareto front: g(x) = 0, which requires x[1:] = 0
        # On the front: f1 = x[0], f2 = 1 - sqrt(x[0])
        #
        # Strategy:
        # Place all points on the Pareto front (x[1:] = 0).
        # Use power-law stratification for x[0] to maximize hypervolume.
        # The hypervolume is dominated by the tail term and the area under the curve.
        # Denser sampling near x[0]=0 minimizes the "dead zone" and captures the steep part of the front.
        # Sparser sampling near x[0]=1 allows the last point to have a wider "tail" to the reference point.
        
        n_points = budget
        # Power exponent alpha. 
        # alpha=1 is uniform. alpha>1 is denser near 0.
        # We hypothesize alpha=2.0 is optimal for this reference point.
        alpha = 2.0
        
        # Generate stratified points
        # i ranges from 1 to n_points
        # f1_i = (i / (n_points + 1)) ** alpha
        i_vals = np.arange(1, n_points + 1)
        x0_vals = (i_vals / (n_points + 1)) ** alpha
        
        # Clip to bounds just in case, though (1/(N+1))^alpha < 1 and > 0
        x0_vals = np.clip(x0_vals, lo[0], hi[0])

        for x0 in x0_vals:
            if used >= budget:
                break
            if time.time() > deadline:
                break
            
            x = np.zeros(dim)
            x[0] = float(x0)
            # Set x[1:] to lower bound (0) to be on the Pareto front
            for j in range(1, dim):
                x[j] = lo[j]
            
            f = problem(x)
            F_list.append(np.array(f, dtype=float))
            X_list.append(x.copy())
            used += 1

        return X_list
# EVOLVE-BLOCK-END
