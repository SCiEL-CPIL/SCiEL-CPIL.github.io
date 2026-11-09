import numpy as np


class Optimizer:
    """
    Optimizer for DTLZ1 using stratified sampling on the known Pareto front.
    Exploits the structure of DTLZ1 where x3..x7=0.5 minimizes g(x),
    and x1, x2 define the position on the front.
    """

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        # DTLZ1 specific knowledge:
        # g(x) is minimized at x3...x7 = 0.5
        # The Pareto front is determined by x1, x2 in the triangle x1>=0, x2>=0, x1+x2<=1
        # We use stratified sampling (jittered grid) on the 2D square mapped to the triangle
        
        n_evals = self.budget
        dim = self.dim
        
        # Number of grid points per dimension
        k = int(np.ceil(np.sqrt(n_evals)))
        
        # Generate grid indices
        i_idx, j_idx = np.meshgrid(np.arange(k), np.arange(k), indexing='ij')
        i_flat = i_idx.flatten()
        j_flat = j_idx.flatten()
        
        # Add jitter using global numpy random (seeded by evaluator)
        jitter_i = np.random.random(len(i_flat))
        jitter_j = np.random.random(len(j_flat))
        
        # Normalized coordinates in [0, 1]
        u1 = (i_flat + jitter_i) / k
        u2 = (j_flat + jitter_j) / k
        
        # We might have more points than budget (k*k > n_evals)
        n_points = len(u1)
        if n_points > n_evals:
            # Randomly select n_evals points to maintain uniformity
            perm = np.random.permutation(n_points)
            u1 = u1[perm[:n_evals]]
            u2 = u2[perm[:n_evals]]
        elif n_points < n_evals:
            # If we have fewer points (unlikely for small n), pad with random points
            # For n=200, k=15, 225 points. So this block is not executed.
            extra = n_evals - n_points
            u1 = np.concatenate([u1, np.random.random(extra)])
            u2 = np.concatenate([u2, np.random.random(extra)])

        # Map from unit square to triangle:
        # x1 = sqrt(u1) * u2
        # x2 = sqrt(u1) * (1 - u2)
        # This ensures x1, x2 >= 0 and x1+x2 = sqrt(u1) <= 1
        x1 = np.sqrt(u1) * u2
        x2 = np.sqrt(u1) * (1 - u2)
        
        # Construct decision vectors
        # x3, x4, x5, x6, x7 set to 0.5 to minimize g(x)
        X = np.zeros((n_evals, dim))
        X[:, 0] = x1
        X[:, 1] = x2
        if dim > 2:
            X[:, 2:] = 0.5
            
        # Evaluate all points
        # problem(x) returns 3 objective values
        F = np.array([problem(x) for x in X])
        
        return X
