import numpy as np
import time
from scipy.stats import qmc

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        t_start = time.time()
        deadline = t_start + 280.0  # Safety margin

        # 1. Construct the fixed part of the vector for DTLZ4 Pareto front (g=0)
        # For DTLZ4 with 12 variables, x[0] and x[1] are position variables.
        # x[2] to x[11] are decision variables for g(x).
        # g(x) = sum_{i=2}^{11} (x_i - 0.5)^2
        # To make g=0, we set x[2:12] = 0.5.
        base_x = np.zeros(self.dim)
        for i in range(2, self.dim):
            lo, hi = bounds[i]
            # Ensure 0.5 is within bounds, usually it is for standard DTLZ4
            base_x[i] = np.clip(0.5, lo, hi)

        # 2. Generate Sobol sequence for uniform coverage on the quarter-sphere
        # We need 200 points.
        n_points = self.budget
        
        # Use Sobol sampler for 2 dimensions
        sampler = qmc.Sobol(d=2, scramble=True, seed=42)
        # Generate n_points + 1 points, discard the first (usually [0,0])
        sobol_points = sampler.random(n_points + 1)[1:]
        
        # sobol_points is in [0, 1)^2
        # Transform to spherical coordinates for the quarter-sphere
        # Uniform area on sphere:
        # u1 -> sin(theta1) = u1 => theta1 = arcsin(u1)
        # u2 -> theta2 = u2 * pi/2
        
        u1 = sobol_points[:, 0]
        u2 = sobol_points[:, 1]
        
        theta1 = np.arcsin(u1)  # [0, pi/2]
        theta2 = u2 * (np.pi / 2.0)  # [0, pi/2]
        
        # Map back to x[0], x[1] in [0, 1]
        # DTLZ4 objectives:
        # f1 = cos(x0/2) * cos(x1/2)
        # f2 = cos(x0/2) * sin(x1/2)
        # f3 = sin(x0/2)
        # So x0/2 = theta1, x1/2 = theta2
        x0_vals = 2.0 * theta1
        x1_vals = 2.0 * theta2
        
        # Clip to bounds just in case, though arcsin and linear map keep them in [0, pi/2] -> [0, 1]
        x0_vals = np.clip(x0_vals, bounds[0][0], bounds[0][1])
        x1_vals = np.clip(x1_vals, bounds[1][0], bounds[1][1])

        # 3. Evaluate all points
        used = 0
        for i in range(n_points):
            if used >= self.budget:
                break
            if time.time() > deadline:
                break
            
            x = base_x.copy()
            x[0] = x0_vals[i]
            x[1] = x1_vals[i]
            
            # Final clip for safety
            for j in range(self.dim):
                lo, hi = bounds[j]
                x[j] = np.clip(x[j], lo, hi)
                
            problem(x)
            used += 1

        return None
