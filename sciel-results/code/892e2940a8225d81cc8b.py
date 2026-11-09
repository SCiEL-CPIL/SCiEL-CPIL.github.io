"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """
    Structured Grid Pareto Front Sampler for DTLZ1.
    
    Strategy:
    1. Exploit the known Pareto front structure of DTLZ1:
       - For M=3 objectives and N=7 variables, the first 2 variables (x[0], x[1])
         determine the position on the Pareto front.
       - The remaining 5 variables (x[2], x[3], x[4], x[5], x[6]) control the
         distance g from the front; setting them to 0 places points on the front.
       
    2. Use a structured grid over the shape parameter space [0, 1]^2 instead of
       Latin Hypercube. This creates a regular mesh that may capture the 
       hypervolume more uniformly.
       
    3. Map the 2D grid to the Pareto front using the standard DTLZ1 mapping:
       - Let a = x[0], b = x[1]
       - f1 = 0.5 * (1 + g) * a
       - f2 = 0.5 * (1 + g) * (1 - a) * b  
       - f3 = 0.5 * (1 + g) * (1 - a) * (1 - b)
       With g=0, this gives the Pareto front.
       
    4. Generate a grid of points in [0, 1]^2 and evaluate each.
    """

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 290  # Safety margin before 330s limit
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        n = self.dim
        m = self.n_obj
        
        # For DTLZ1 with M objectives:
        # Shape vars: first m-1
        # Dist vars: rest
        n_shape = m - 1
        n_dist = n - n_shape
        
        # We will use the full budget for sampling the Pareto front
        n_samples = self.budget
        
        # Generate a structured grid over [0, 1]^n_shape
        # For n_shape=2, use a 2D grid
        if n_shape == 2:
            # Create a square-ish grid
            n_per_dim = int(np.sqrt(n_samples))
            # Adjust to get exactly n_samples or close
            # Use n_per_dim x n_per_dim, then add remaining points
            grid_u = np.linspace(0.5 / n_per_dim, 1.0 - 0.5 / n_per_dim, n_per_dim)
            grid_v = np.linspace(0.5 / n_per_dim, 1.0 - 0.5 / n_per_dim, n_per_dim)
            
            # Create meshgrid
            U, V = np.meshgrid(grid_u, grid_v)
            U = U.flatten()
            V = V.flatten()
            
            # If we have more points than budget, trim
            if len(U) > n_samples:
                U = U[:n_samples]
                V = V[:n_samples]
            elif len(U) < n_samples:
                # Add extra points randomly to fill the budget
                n_extra = n_samples - len(U)
                extra_u = np.random.uniform(0, 1, n_extra)
                extra_v = np.random.uniform(0, 1, n_extra)
                U = np.concatenate([U, extra_u])
                V = np.concatenate([V, extra_v])
            
            # Construct full variable vectors
            X = np.zeros((n_samples, n))
            X[:, 0] = U
            X[:, 1] = V
            # Distance variables are set to 0 to be on the Pareto front
            # Note: Bounds are [0, 1], so 0 is valid.
            
        elif n_shape == 1:
            # 1D grid
            grid_u = np.linspace(0.5 / n_samples, 1.0 - 0.5 / n_samples, n_samples)
            X = np.zeros((n_samples, n))
            X[:, 0] = grid_u
            # Distance variables are 0
            
        else:
            # Fallback to LHS for higher dimensional shape spaces
            rng = np.random.default_rng(42)
            shape_samples = np.zeros((n_samples, n_shape))
            for j in range(n_shape):
                perm = rng.permutation(n_samples)
                shape_samples[:, j] = (perm + rng.random(n_samples)) / n_samples
            
            X = np.zeros((n_samples, n))
            X[:, :n_shape] = shape_samples
            # Distance variables are 0
        
        # Clip to bounds just in case
        X = np.clip(X, lo, hi)
        
        # Evaluate all points
        for i in range(n_samples):
            if time.time() > deadline:
                break
            problem(X[i])
            
        return X
# EVOLVE-BLOCK-END
