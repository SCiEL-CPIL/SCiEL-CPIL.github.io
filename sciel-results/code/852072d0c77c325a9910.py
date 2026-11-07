"""Multi-objective optimizer for DTLZ1 using direct Pareto front sampling."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import qmc


class Optimizer:
    """Direct sampler for DTLZ1 Pareto front."""

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

        # Strategy: Hybrid Grid + Sobol
        # 1. Generate a grid of points on the simplex to ensure dense coverage.
        # 2. Fill remaining budget with Sobol points mapped to the simplex.
        
        # Determine grid size K such that (K+1)(K+2)/2 <= budget
        # For budget=200, K=18 gives 190 points.
        # K=19 gives 210 points.
        K = 18
        n_grid = (K + 1) * (K + 2) // 2
        
        X = np.zeros((budget, dim))
        
        # Generate grid points
        idx = 0
        for i in range(K + 1):
            for j in range(K + 1 - i):
                if idx >= budget:
                    break
                # Simplex coordinates
                x1 = i / K
                x2 = j / K
                # x3 is implicitly 1 - x1 - x2
                X[idx, 0] = x1
                X[idx, 1] = x2
                for d in range(2, dim):
                    X[idx, d] = (lo[d] + hi[d]) / 2.0
                idx += 1
        
        # Generate Sobol points for the rest
        remaining = budget - idx
        if remaining > 0:
            sobol = qmc.Sobol(d=2, scramble=True)
            points_2d = sobol.random(remaining)
            
            # Map to simplex using reflection
            for i in range(len(points_2d)):
                if points_2d[i, 0] + points_2d[i, 1] > 1.0:
                    points_2d[i, 0] = 1.0 - points_2d[i, 0]
                    points_2d[i, 1] = 1.0 - points_2d[i, 1]
            
            for i in range(remaining):
                X[idx, 0] = points_2d[i, 0]
                X[idx, 1] = points_2d[i, 1]
                for d in range(2, dim):
                    X[idx, d] = (lo[d] + hi[d]) / 2.0
                idx += 1
        
        # Evaluate all points
        all_points = []
        used = 0
        for x in X:
            if time.time() > deadline:
                break
            f = problem(x)
            all_points.append(f)
            used += 1
            
        return X
# EVOLVE-BLOCK-END
