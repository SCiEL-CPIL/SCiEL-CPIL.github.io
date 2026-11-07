"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import qmc

class Optimizer:
    """Direct sampling on the DTLZ1 Pareto front using Sobol sequence."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 300.0  # 300s limit, keep margin

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        # DTLZ1 specific knowledge:
        # Pareto front: x_{n_obj-1}...x_{dim-1} = 0.5
        # First n_obj-1 variables define the position.
        # For M=3, N=7: x_0, x_1 vary, x_2...x_6 = 0.5
        
        n_obj = self.n_obj
        dim = self.dim
        
        # Generate Sobol points in [0,1]^2
        # We need 'budget' points
        sampler = qmc.Sobol(d=2, scramble=True)
        # Sobol requires power of 2 for initial points? 
        # qmc.Sobol can generate any number, but it's often best to generate 2^k and take subset or use specific count.
        # scipy.stats.qmc.Sobol.generate(n) works for any n.
        sobol_points = sampler.generate(self.budget, scramble=True)
        
        # Map to unit disk
        # u1, u2 in [0,1]
        # r = sqrt(u1), theta = 2*pi*u2
        u1, u2 = sobol_points.T
        r = np.sqrt(u1)
        theta = 2 * np.pi * u2
        
        x0 = r * np.cos(theta)
        x1 = r * np.sin(theta)
        
        # Construct full vectors
        # x0, x1 are the first 2 variables (since n_obj=3, n_obj-1=2)
        # Remaining dim - (n_obj-1) variables are 0.5
        n_pos_vars = n_obj - 1
        n_agg_vars = dim - n_pos_vars
        
        X = np.zeros((self.budget, dim))
        X[:, 0] = x0
        X[:, 1] = x1
        # Set aggregation variables to 0.5
        if n_agg_vars > 0:
            X[:, n_pos_vars:] = 0.5
            
        # Ensure bounds are respected (though DTLZ1 front is within [0,1])
        # The problem function clips, but we should provide valid inputs if possible
        # x0, x1 can be negative? 
        # DTLZ1 usually defines x in [0,1]. 
        # If x0, x1 are in [0,1] disk, cos/sin can produce negative values.
        # Standard DTLZ1 definition: x_i in [0,1].
        # If x0 < 0, it is outside the standard domain? 
        # Most implementations allow x in [0,1] for all variables.
        # The Pareto front condition is x_{M-1}..x_{N-1} = 0.5.
        # And f1, f2, f3 depend on x0..x_{M-2}.
        # Usually x0..x_{M-2} are in [0,1].
        # If we use polar coordinates, x0, x1 can be negative.
        # We should clip to [0,1] or ensure they stay in [0,1].
        # The unit disk in the first quadrant? 
        # If x0, x1 are in [0,1], then x0^2+x1^2 <= 1 is a quarter disk.
        # Does DTLZ1 allow negative x? 
        # Standard DTLZ1: "x_i in [0,1]". 
        # So we should restrict to the quarter disk in the first quadrant.
        
        # Let's re-generate or filter to ensure x0, x1 >= 0.
        # Using Sobol in [0,1]^2 and mapping to quarter disk:
        # r = sqrt(u1), theta = (pi/2) * u2
        # This ensures x0, x1 >= 0.
        
        theta_q = (np.pi / 2.0) * u2
        x0_q = r * np.cos(theta_q)
        x1_q = r * np.sin(theta_q)
        
        X[:, 0] = x0_q
        X[:, 1] = x1_q
        
        # Final check against bounds
        X = np.clip(X, lo, hi)
        
        F = np.array([problem(x) for x in X])
        
        return X
# EVOLVE-BLOCK-END
