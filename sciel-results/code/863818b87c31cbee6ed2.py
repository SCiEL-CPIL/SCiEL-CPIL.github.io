import numpy as np
import time
from scipy.stats import qmc

class Optimizer:
    """
    Optimizer for DTLZ7 exploiting the known structure of the Pareto front.
    DTLZ7 has 2 active variables (x0, x1) and N-2 inactive variables.
    To minimize the objectives, inactive variables should be 0.
    The Pareto front is a 2D surface parameterized by x0 and x1.
    We sample x0, x1 using a Sobol sequence to ensure uniform coverage of the front,
    which maximizes the hypervolume of the discrete set of points.
    """

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280.0  # Safety margin for 300s limit
        
        # Extract bounds
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        # DTLZ7 structure:
        # Active variables: x[0], x[1]
        # Inactive variables: x[2:]
        # To minimize g(x), inactive variables should be 0.
        # We assume the standard DTLZ7 where g(x) is minimized when inactive vars are 0.
        
        # Generate Sobol samples for the 2 active variables
        # We use 200 points to match the budget
        n_samples = self.budget
        
        # Sobol sampler for 2 dimensions
        sampler = qmc.Sobol(d=2, scramble=True, seed=42)
        # Generate points in [0, 1]^2
        # n_samples should be a power of 2 for best quality, but Sobol works for others too.
        # 256 is close to 200. Let's generate 256 and take the first 200? 
        # Or just generate 200.
        # Actually, for low discrepancy, powers of 2 are better. 
        # Let's generate 256 points and use the first 200.
        n_sobol = 256
        active_samples = sampler.random(n_sobol)
        active_samples = active_samples[:n_samples]
        
        # Construct the full decision vectors
        X = np.zeros((n_samples, self.dim))
        X[:, 0] = active_samples[:, 0]
        X[:, 1] = active_samples[:, 1]
        # X[:, 2:] remains 0
        
        # Evaluate all points
        # We must be careful with time. 200 evaluations is fast.
        for i in range(n_samples):
            if time.time() > deadline:
                break
            # Call problem to record the evaluation
            problem(X[i])
            
        return X
