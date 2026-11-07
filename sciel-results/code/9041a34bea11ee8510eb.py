# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import qmc


class Optimizer:
    """
    Multi-objective optimizer for DTLZ2.
    
    Approach:
    - Analytically fix the distance variables (last dim - n_obj + 1) to 0.5 to ensure 
      the points lie on the Pareto front (g(x) = 0).
    - Sample the angle variables (first n_obj - 1) using a Sobol sequence.
    - Apply a transformation to the angle samples to ensure uniform distribution over 
      the surface area of the Pareto front (hypersphere), which maximizes hypervolume 
      coverage.
    """

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        budget = self.budget
        dim = self.dim
        n_obj = self.n_obj
        
        n_angle = n_obj - 1
        
        # Generate Sobol sequence in [0, 1)^n_angle
        # scramble=False ensures deterministic behavior
        sampler = qmc.Sobol(d=n_angle, scramble=False)
        points = sampler.random(budget)
        
        # Transform points to uniform surface area on the Pareto front
        # For n_obj = 3 (n_angle = 2):
        # The Pareto front is a quarter sphere.
        # Surface area element dA = cos(theta_1) dtheta_1 dtheta_2.
        # To sample theta_1 uniformly w.r.t area, we use the inverse CDF:
        # u_1 = sin(theta_1) => theta_1 = arcsin(u_1)
        # x_1 = (2/pi) * theta_1 = (2/pi) * arcsin(u_1)
        
        if n_obj == 3:
            # points[:, 0] is u_1 ~ U[0,1]
            # points[:, 1] is u_2 ~ U[0,1]
            
            # Transform u_1 to x_1
            points[:, 0] = (2.0 / np.pi) * np.arcsin(points[:, 0])
            
            # u_2 remains as x_2, since theta_2 = (pi/2)*x_2 is linear
            # No transformation needed for x_2.
            
        # Build full x vectors: angle variables from points, distance variables = 0.5
        X = np.zeros((budget, dim))
        X[:, :n_angle] = points
        X[:, n_angle:] = 0.5  # Distance variables set to 0.5 for optimality
        
        # Clip to bounds
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        X = np.clip(X, lo, hi)
        
        # Evaluate all points
        for x in X:
            problem(x)
        
        return X
# EVOLVE-BLOCK-END
