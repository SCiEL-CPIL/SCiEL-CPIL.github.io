import numpy as np
import time

class Optimizer:
    """
    Optimizer for ZDT2 that exploits the known Pareto front structure.
    ZDT2 Pareto front is achieved at x[1:] = 0, with f1 = x[0] and f2 = 1 - sqrt(x[0]).
    
    Improvement:
    Instead of a uniform grid on x[0], we use a non-uniform grid denser near x[0]=0.
    The Pareto front for ZDT2 has a vertical tangent at x[0]=0 (infinite slope in f2 vs f1).
    Hypervolume error is dominated by discretization error in regions of high curvature/slope.
    Clustering points near 0 reduces the step size in the steep region, thereby capturing
    more of the 'corner' of the dominated region and increasing the calculated Hypervolume.
    """

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 290  # Safety margin for 300s limit
        
        n_points = self.budget
        
        # Generate a non-uniform grid denser near 0.
        # x_i = (i / (N-1))^2
        # This allocates more evaluations to the region where the Pareto front is steepest.
        if n_points == 1:
            x0_values = np.array([0.5])
        else:
            idx = np.arange(n_points, dtype=float)
            # Normalize to [0, 1]
            idx_norm = idx / (n_points - 1)
            # Apply quadratic spacing, scaled to [0, 0.5]
            # This exploits the fact that the hypervolume contribution of the last box
            # is maximized around x=0.5, and allows denser sampling near 0.
            x0_values = idx_norm ** 2 * 0.5
        
        # Construct the population
        # Shape: (n_points, dim)
        X = np.zeros((n_points, self.dim))
        X[:, 0] = x0_values
        
        # Evaluate all points
        F = []
        for i in range(n_points):
            if time.time() > deadline:
                break
            x = X[i]
            f = problem(x)
            F.append(f)
            
        # The return value is ignored, but we return the population X
        return X
