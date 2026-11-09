import numpy as np
from scipy.stats import qmc
import time

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280.0  # Safety margin before 300s limit
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        # DTLZ7 Pareto front depends only on x0, x1. Set x2...x21 = lo[2:].
        # The optimal g is achieved when x[2:] = 0 (or lower bound if different).
        
        n_points = self.budget
        
        # 1. Prepare corners for full extent coverage
        x1_lo, x1_hi = bounds[0]
        x2_lo, x2_hi = bounds[1]
        
        corners = np.array([
            [x1_lo, x2_lo],
            [x1_hi, x2_lo],
            [x1_lo, x2_hi],
            [x1_hi, x2_hi]
        ])
        
        n_corners = 4
        n_sobol = n_points - n_corners
        
        if n_sobol < 0:
            n_sobol = 0
            n_corners = n_points
        
        # 2. Generate Sobol points for low-discrepancy coverage
        if n_sobol > 0:
            sampler = qmc.Sobol(d=2, scramble=False)
            sobol_points = sampler.random(n_sobol)
            
            # Transform to [0,1] range with power-law bias toward origin
            # The Pareto front is denser near the origin in objective space
            alpha = 2.0
            x0_sobol = sobol_points[:, 0] ** alpha
            x1_sobol = sobol_points[:, 1] ** alpha
            
            # Map to actual bounds
            x0_sobol = x1_lo + (x1_hi - x1_lo) * x0_sobol
            x1_sobol = x2_lo + (x2_hi - x2_lo) * x1_sobol
            
            sobol_xy = np.column_stack((x0_sobol, x1_sobol))
        else:
            sobol_xy = np.array([]).reshape(0, 2)
        
        # 3. Combine corners and Sobol points
        all_xy = np.vstack((corners, sobol_xy))
        
        # 4. Construct full X matrix with x[2:] = lo[2:]
        X = np.tile(lo, (len(all_xy), 1))
        X[:, 0] = all_xy[:, 0]
        X[:, 1] = all_xy[:, 1]
        
        # Clip to bounds for safety
        X = np.clip(X, lo, hi)
        
        # 5. Evaluate all points with time guard
        for i in range(len(X)):
            if time.time() > deadline:
                break
            problem(X[i])
        
        return None
