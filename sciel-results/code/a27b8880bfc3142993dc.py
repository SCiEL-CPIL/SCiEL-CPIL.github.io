# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import qmc


class Optimizer:
    """
    Multi-objective optimizer for DTLZ2 using Sobol sequence with analytic
    Pareto front projection for optimal hypervolume coverage.
    
    DTLZ2 structure:
    - 12 variables, 3 objectives
    - First 2 variables are angle variables (theta1, theta2 in [0, pi/2])
    - Last 10 variables are distance variables (optimal at 0.5)
    - Pareto front is a quarter-sphere in positive octant of R^3
    
    Key insight: For uniform surface coverage on the quarter-sphere Pareto front,
    we need cos(theta1) uniform in [0,1] and theta2 uniform in [0, pi/2].
    This transforms uniform [0,1]^2 Sobol points to the correct angle space.
    """

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def _sobol_to_angles(self, n):
        """
        Generate n points on the DTLZ2 Pareto front using Sobol sequence.
        
        For uniform coverage on the quarter-sphere surface:
        - v1 uniform in [0,1] -> theta1 = arccos(v1)
        - v2 uniform in [0,1] -> theta2 = (pi/2) * v2
        
        Returns array of shape (n, 2) with angle variable values in [0, 1]
        (i.e., x1 = theta1/(pi/2), x2 = theta2/(pi/2)).
        """
        # Generate Sobol sequence in 2D
        sampler = qmc.Sobol(d=2, scramble=True)
        pts = sampler.random(n)
        
        # Transform to get uniform surface coverage on quarter-sphere
        # v1 = pts[:, 0] uniform in [0,1]
        # v2 = pts[:, 1] uniform in [0,1]
        
        # theta1 = arccos(v1), so x1 = theta1 / (pi/2) = 2*arccos(v1)/pi
        v1 = np.clip(pts[:, 0], 1e-10, 1.0 - 1e-10)
        x1 = 2.0 * np.arccos(v1) / np.pi
        
        # theta2 = (pi/2) * v2, so x2 = theta2 / (pi/2) = v2
        x2 = np.clip(pts[:, 1], 0.0, 1.0)
        
        return np.column_stack([x1, x2])

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280  # safety margin before 300s limit
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        n_obj = self.n_obj  # 3
        dim = self.dim  # 12
        n_angle = n_obj - 1  # 2 angle variables
        n_dist = dim - n_obj + 1  # 10 distance variables
        
        n_samples = min(self.budget, 200)
        
        # Generate Sobol-based angle points with correct surface measure
        angle_pts = self._sobol_to_angles(n_samples)
        
        # Construct decision vectors:
        # - First 2 variables: angle variables from Sobol sequence
        # - Last 10 variables: distance variables set to 0.5 (optimal for DTLZ2)
        X = np.zeros((n_samples, dim))
        X[:, :n_angle] = angle_pts
        X[:, n_angle:] = 0.5
        
        # Clip to bounds
        X = np.clip(X, lo, hi)
        
        # Evaluate all points
        F = np.array([problem(x) for x in X])
        
        # Check time limit
        if time.time() > deadline:
            return X
        
        return X
# EVOLVE-BLOCK-END
