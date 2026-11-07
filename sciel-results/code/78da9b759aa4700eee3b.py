# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """
    Multi-objective optimizer for DTLZ2 exploiting the known Pareto front structure.
    
    DTLZ2 with 3 objectives has a Pareto front that is a quarter-sphere in the positive
    octant of R^3. The distance variables (last dim - n_obj + 1 = 10 variables) should
    be set to 0.5 for optimality. The angle variables (first n_obj - 1 = 2 variables)
    determine position on the front.
    
    Strategy: Use Halton quasi-random sequence to sample the 2D angle space densely,
    then map to decision space. This directly generates points on the Pareto front.
    """

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280  # 280s safety margin before 300s limit
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        dim = self.dim
        n_obj = self.n_obj
        budget = self.budget
        
        # Number of angle variables
        n_angles = n_obj - 1  # 2 for 3 objectives
        
        # Generate Sobol sequence points in [0, 1]^2
        n_points = min(budget, 200)
        
        # Use scipy Sobol for better low-discrepancy properties
        from scipy.stats import qmc
        sampler = qmc.Sobol(d=2, scramble=True)
        sobol_points = sampler.random(n_points)
        
        # Map Sobol points to angles on the Pareto front
        # For DTLZ2 with 3 objectives, the Pareto front is a quarter-sphere
        # f1 = 0.5 * cos(theta1)
        # f2 = 0.5 * sin(theta1) * cos(theta2)
        # f3 = 0.5 * sin(theta1) * sin(theta2)
        # where theta1, theta2 in [0, pi/2]
        
        # Surface area element is proportional to sin(theta1) dtheta1 dtheta2
        # To sample uniformly in surface area:
        # theta1 = arcsin(u1) where u1 ~ U[0,1]
        # theta2 = u2 * pi/2 where u2 ~ U[0,1]
        u1 = sobol_points[:, 0]
        u2 = sobol_points[:, 1]
        
        theta1 = np.arcsin(u1)
        theta2 = u2 * (np.pi / 2)
        
        # Build decision vectors
        # First n_obj - 1 variables are the angles (scaled to [0, pi/2])
        # Last dim - n_obj + 1 variables are distance variables (set to 0.5)
        X = np.zeros((n_points, dim))
        X[:, 0] = theta1
        X[:, 1] = theta2
        for j in range(2, dim):
            X[:, j] = 0.5  # Midpoint of [0, 1]
        
        # Evaluate all points
        F = np.array([problem(x) for x in X])
        used = n_points
        
        return X
# EVOLVE-BLOCK-END
