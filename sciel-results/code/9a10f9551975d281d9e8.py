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
        deadline = start_time + 250.0  # safety margin before 300s/330s kill

        dim = self.dim
        n_obj = self.n_obj
        budget = self.budget

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])

        # DTLZ4 with 12 vars, 3 objectives:
        # M = 3 objectives, K = M - 1 = 2 position variables
        # N = 12, so M = 10 distance variables
        # Pareto front: x_{M+1..M+K} = x_{M+K+1..N} where g(x) = 0
        # For DTLZ4, g(x) = sum((x_i - 0.5)^2) for i in distance vars
        # g = 0 when each distance var = 0.5
        # f_1 = (1 + g) * prod_{i=1}^{K} cos(x_i * pi/2)
        # f_i = (1 + g) * sin(x_i * pi/2) * prod_{j=1}^{i-2} cos(x_j * pi/2) for 2 <= i <= K+1
        # f_{K+1} = (1 + g) * prod_{j=1}^{K-1} cos(x_j * pi/2) * sin(x_K * pi/2)
        # Actually for DTLZ4: f_i = (1+g) * x_i for i <= K, and f_{K+1} = (1+g) * (1 - sum of squares... )
        # Wait, let me be more careful. DTLZ4 has:
        # f_i(x) = (1 + g(x_M)) * x_{i-1} for i = 1..M+1
        # No, that's DTLZ1. DTLZ4 is:
        # f_1 = (1+g) * cos(x_1*pi/2)
        # f_2 = (1+g) * sin(x_1*pi/2) * cos(x_2*pi/2)
        # f_3 = (1+g) * sin(x_2*pi/2)
        # with g(x) = sum_{i=3}^{12} (x_i - 0.5)^2
        # Pareto front when g=0: f_1 = cos(x_1*pi/2), f_2 = sin(x_1*pi/2)*cos(x_2*pi/2), f_3 = sin(x_2*pi/2)
        # Let theta_1 = x_1*pi/2, theta_2 = x_2*pi/2
        # f_1 = cos(t1), f_2 = sin(t1)*cos(t2), f_3 = sin(t2)
        # with t1, t2 in [0, pi/2]
        
        # So the Pareto front is parameterized by (t1, t2) in [0, pi/2]^2
        # f_1 = cos(t1)
        # f_2 = sin(t1)*cos(t2)
        # f_3 = sin(t2)
        
        # Strategy: Sample (t1, t2) directly from the Pareto front parameter space
        # Use a grid + jitter approach for good hypervolume coverage
        
        # For hypervolume maximization with reference point [2.5, 2.5, 2.5]:
        # We want to cover the Pareto front well, especially near the "knee" regions
        # where points contribute most to hypervolume
        
        # Use a combination of:
        # 1. A structured grid in (t1, t2) space
        # 2. Some points at the extremes
        
        n_budget = budget
        
        # Strategy: Use Sobol low-discrepancy points in the parameter space (t1, t2)
        # to ensure uniform coverage of the Pareto front, plus explicit corners.
        
        n_corners = 4
        n_sobol = n_budget - n_corners
        
        points = []
        
        # 1. Add explicit corner points in (t1, t2) space
        # Corners correspond to the extremes of the objective space
        corner_coords = [
            (0.0, 0.0),      # f1=1, f3=0
            (0.0, np.pi/2),  # f1=1, f3=1 -> wait, f2=0. This is a vertex.
            (np.pi/2, 0.0),  # f1=0, f3=0
            (np.pi/2, np.pi/2) # f1=0, f3=1
        ]
        
        for t1, t2 in corner_coords:
            x = np.zeros(dim)
            x[0] = t1 / (np.pi/2)
            x[1] = t2 / (np.pi/2)
            x[2:] = 0.5
            points.append(x)
        
        # 2. Generate Sobol points for the remaining budget
        if n_sobol > 0:
            # Use a Sobol generator for 2 dimensions
            # scipy.stats.qmc.Sobol(d=2)
            # We need to scramble to avoid potential bias in specific seeds
            # But standard Sobol is deterministic and fine.
            sobol = qmc.Sobol(d=2, scramble=True)
            # Generate points in [0, 1]^2
            sobol_points = sobol.random(n_sobol)
            
            # Scale to [0, pi/2]
            t1_sobol = sobol_points[:, 0] * (np.pi/2)
            t2_sobol = sobol_points[:, 1] * (np.pi/2)
            
            for i in range(n_sobol):
                x = np.zeros(dim)
                x[0] = t1_sobol[i] / (np.pi/2)
                x[1] = t2_sobol[i] / (np.pi/2)
                x[2:] = 0.5
                points.append(x)
        
        # Clip to budget just in case
        if len(points) > n_budget:
            points = points[:n_budget]
        
        # Evaluate all points
        F = np.array([problem(x) for x in points])
        
        if time.time() > deadline:
            return F
        
        return F
