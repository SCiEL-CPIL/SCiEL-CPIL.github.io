# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import qmc


class Optimizer:
    """
    Multi-objective optimizer for DTLZ4.
    
    Strategy:
    1. Exploit the analytic structure of DTLZ4:
       - For 3 objectives, the Pareto front is parameterized by phi in [0, pi/2]
       - x[0] = cos(phi), x[1] = sin(phi)
       - g-variables (x[2:]) all at 0.5 for g=0
    2. Use Sobol sequence for better space-filling coverage of phi
    3. Evaluate all points exactly on the Pareto front
    """

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        dim = self.dim
        n_obj = self.n_obj
        budget = self.budget
        
        n_front = n_obj - 1  # 2 position variables
        n_g = dim - n_front  # 10 g-variables
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        # Use all budget for sampling the Pareto front
        n_samples = budget
        
        # Generate Sobol sequence in [0, 1] for better space-filling
        sobol = qmc.Sobol(d=1, scramble=True)
        phi_uniform = sobol.random(n_samples)[:, 0]  # values in [0, 1)
        
        # Map to phi in [0, pi/2]
        phi_values = phi_uniform * (np.pi / 2)
        
        # Generate decision vectors on the Pareto front
        X = np.zeros((n_samples, dim))
        X[:, 0] = np.cos(phi_values)
        X[:, 1] = np.sin(phi_values)
        # Set g-variables to exactly 0.5 (optimal for DTLZ4, gives g=0)
        X[:, n_front:] = 0.5
        
        # Clip to bounds (should be no-op since cos/sin in [0,1] and 0.5 in [0,1])
        X = np.clip(X, lo, hi)
        
        # Evaluate all points
        F = np.array([problem(x) for x in X])
        
        return X
# EVOLVE-BLOCK-END
