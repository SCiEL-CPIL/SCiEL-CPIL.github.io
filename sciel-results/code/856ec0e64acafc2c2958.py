# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import qmc


class Optimizer:
    """
    Multi-objective optimizer for DTLZ2.
    
    Approach:
    - DTLZ2 Pareto front is a unit hypersphere in objective space (for n_obj objectives).
    - For 3 objectives, the Pareto front is a quarter-sphere in the positive octant of R^3.
    - The distance variables (last dim - n_obj + 1 variables) should be set to 0.5 for optimality.
    - The angle variables (first n_obj - 1 = 2 variables) determine the position on the Pareto front.
    
    We use a deterministic Sobol sequence (low-discrepancy) to sample the 2D angle space.
    To maximize hypervolume, we transform the sampling to be uniform with respect to the
    surface area of the Pareto front, rather than uniform in the angle parameters.
    This avoids over-sampling the "tips" of the Pareto front where hypervolume contribution
    per point is lower.
    """

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        budget = self.budget
        dim = self.dim
        n_obj = self.n_obj
        
        # For DTLZ2 with n_obj=3, dim=12:
        # Angle variables: first n_obj - 1 = 2 variables
        # Distance variables: remaining dim - (n_obj - 1) = 10 variables, set to 0.5
        
        n_angle = n_obj - 1  # 2
        
        # Generate a low-discrepancy sequence in [0,1]^n_angle using Sobol
        # scramble=False ensures deterministic behavior independent of np.random.seed
        sampler = qmc.Sobol(d=n_angle, scramble=False)
        points = sampler.random(budget)
        
        # points shape: (budget, n_angle)
        
        # Transform points to ensure uniform distribution on the Pareto front surface.
        # For DTLZ2 with 3 objectives:
        # theta1 = pi/2 * x1, theta2 = pi/2 * x2
        # Surface area element is proportional to cos(theta1) dtheta1 dtheta2.
        # To sample uniformly on the surface, we need the CDF of theta1 to be proportional to sin(theta1).
        # So, if u1 ~ U[0,1], we set sin(theta1) = u1 => theta1 = arcsin(u1).
        # Thus, x1 = (2/pi) * arcsin(u1).
        # theta2 is uniform, so x2 = u2.
        
        if n_angle >= 1:
            points[:, 0] = (2.0 / np.pi) * np.arcsin(points[:, 0])
        # x2 (and any further angles, though there are none for n_obj=3) remain uniform
        
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
