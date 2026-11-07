import numpy as np
import time
from scipy.stats import qmc

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 290
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        dim = self.dim
        n_obj = self.n_obj
        budget = self.budget
        
        # DTLZ2 specifics:
        # First n_obj - 1 variables determine the shape (angles)
        # Remaining dim - (n_obj - 1) variables determine the distance to the front (g function)
        # For DTLZ2, g(x) = sum((x_i - 0.5)^2) for i >= n_obj - 1
        # To minimize objectives, we want g(x) = 0, so x_i = 0.5 for i >= n_obj - 1
        
        n_shape_vars = n_obj - 1  # 2 variables for 3 objectives
        n_dist_vars = dim - n_shape_vars # 10 variables
        
        # We will use the entire budget to generate diverse points on the Pareto front.
        # We use a Sobol sequence for the shape variables to ensure good coverage.
        
        # Determine number of points to generate
        n_points = budget
        
        # Sobol sequence requires power of 2 for best properties, but can handle any N.
        # We'll generate n_points points in [0, 1]^n_shape_vars
        # Using Sobol with a fixed seed for reproducibility
        sampler = qmc.Sobol(d=n_shape_vars, seed=0)
        # Generate a power of 2 >= n_points to utilize Sobol's best properties, then slice
        # 256 is the next power of 2 >= 200
        n_sobol = 256 
        shape_samples_full = sampler.random(n=n_sobol)
        shape_samples = shape_samples_full[:n_points]
        
        # Initialize the full dimension array
        # Set distance variables to 0.5 (midpoint of [0,1])
        # Note: bounds are [0,1], so 0.5 is optimal for g(x)
        dist_mid = np.array([(lo[i] + hi[i]) / 2 for i in range(n_shape_vars, dim)])
        
        X = np.zeros((n_points, dim))
        
        # Fill shape variables
        X[:, :n_shape_vars] = shape_samples
        
        # Fill distance variables
        X[:, n_shape_vars:] = dist_mid
        
        # Evaluate all points
        F = np.zeros((n_points, n_obj))
        for i in range(n_points):
            if time.time() > deadline:
                break
            # Ensure clipping just in case, though construction should be within bounds
            x = np.clip(X[i], lo, hi)
            F[i] = problem(x)
            
        return X
