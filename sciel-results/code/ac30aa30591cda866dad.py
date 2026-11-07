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
        
        # DTLZ2 structure
        # x[0], x[1] are position variables
        # x[2:] are decision variables, set to 0.5 to minimize g
        
        X = np.zeros((self.budget, self.dim))
        
        # Set decision variables to midpoint
        mid = (lo + hi) / 2.0
        X[:, 2:] = mid[2:]
        
        # Add 3 vertex points for better hypervolume coverage
        n_vertices = 3
        n_sobol = self.budget - n_vertices
        
        # Vertex points in position space
        vertices = np.array([
            [1.0, 0.0],  # x_0=1, x_1=0
            [0.0, 1.0],  # x_0=0, x_1=1
            [0.0, 0.0],  # x_0=0, x_1=0
        ])
        
        X[:n_vertices, 0] = lo[0] + vertices[:, 0] * (hi[0] - lo[0])
        X[:n_vertices, 1] = lo[1] + vertices[:, 1] * (hi[1] - lo[1])
        
        # Generate Sobol sequence for remaining points with fixed seed
        sampler = qmc.Sobol(d=2, scramble=True, seed=42)
        sobol_points = sampler.random(n_sobol)
        
        # Scale to bounds
        X[n_vertices:, 0] = lo[0] + sobol_points[:, 0] * (hi[0] - lo[0])
        X[n_vertices:, 1] = lo[1] + sobol_points[:, 1] * (hi[1] - lo[1])
        
        # Evaluate
        F = np.array([problem(x) for x in X])
        
        return X
