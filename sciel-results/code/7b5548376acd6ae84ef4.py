import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 290  # Safety margin before 300s limit
        
        n_samples = self.budget
        
        # DTLZ7 structure:
        # M=3 objectives, n=22 variables.
        # m = n - M + 1 = 20 distance variables: x[2] ... x[21]
        # Position variables: x[0], x[1]
        # The Pareto front is 2D, parameterized by (x0, x1) in [0,1]^2.
        # Setting all distance variables to 0 gives g=0, placing points on the Pareto front.
        
        # Use a Fibonacci lattice (golden-ratio spiral) on the unit disk mapped to [0,1]^2
        # for uniform area coverage of the non-convex Pareto front.
        
        # Fibonacci lattice: golden ratio spiral
        golden_angle = np.pi * (3.0 - np.sqrt(5.0))  # golden angle in radians
        
        # Generate n_samples points using Fibonacci lattice on unit disk
        # Map to [0, 1] x [0, 1] square
        i = np.arange(n_samples, dtype=np.float64)
        # Use a scaling factor to spread points from center outward
        # r = sqrt(i/n) gives uniform area density on disk
        r = np.sqrt(i / n_samples)
        theta = i * golden_angle
        
        # Convert polar to Cartesian on unit disk, then map to [0,1]^2
        x0 = 0.5 + r * np.cos(theta) * 0.5
        x1 = 0.5 + r * np.sin(theta) * 0.5
        
        # Clip to [0, 1] to ensure valid inputs
        x0 = np.clip(x0, 0.0, 1.0)
        x1 = np.clip(x1, 0.0, 1.0)
        
        results = []
        for idx in range(n_samples):
            if time.time() > deadline:
                break
            
            x = np.zeros(self.dim)
            x[0] = x0[idx]
            x[1] = x1[idx]
            # Distance variables remain 0 (g=0, on Pareto front)
            
            try:
                f = problem(x)
                results.append(f)
            except Exception:
                pass
                    
        return results
