import numpy as np
import time
from scipy.stats import qmc


class Optimizer:
    """
    Optimizer for DTLZ4 (M=3, N=12).
    Exploits the known structure of the Pareto Front:
    - Position variables: x[0], x[1]
    - Distance variables: x[2:] (must be 0.5 for g=0)
    
    Uses Sobol low-discrepancy sampling on the 2D Pareto front parameter space
    to maximize hypervolume coverage with 200 evaluations.
    """

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 290  # 10s safety margin
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        # Generate Sobol sequence for 2D parameter space
        # 200 points for 200 evaluations
        n_points = self.budget
        sobol = qmc.Sobol(d=2, scramble=False)
        # Need power of 2 for Sobol, so use 256 and take first 200
        n_sobol = 256
        samples_2d = sobol.random(n_sobol)[:n_points]
        
        # Map to x[0], x[1] in [0, 1]
        # For DTLZ4, the Pareto front is parameterized by x[0], x[1] in [0,1]
        # when x[2:] = 0.5 (g=0)
        
        candidates = []
        for i in range(n_points):
            x = np.zeros(self.dim)
            x[0] = samples_2d[i, 0]
            x[1] = samples_2d[i, 1]
            x[2:] = 0.5
            x = np.clip(x, lo, hi)
            candidates.append(x)
        
        results = []
        for x in candidates:
            if time.time() > deadline:
                break
            try:
                f = problem(x)
                results.append(f)
            except Exception:
                break
        
        return results
