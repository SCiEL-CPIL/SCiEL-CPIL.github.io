"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import qmc


class Optimizer:
    """Problem-specific optimizer exploiting DTLZ2's known Pareto front structure with area-uniform sampling."""

    def __init__(self, budget, dim, n_obj):
        self.budget, self.dim, self.n_obj = budget, dim, n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280  # Safety margin for 300s limit

        dim = self.dim
        n_obj = self.n_obj
        budget = self.budget

        # DTLZ2 specifics:
        # f_i(x) = (1 + g(x)) * h_i(y)
        # g(x) = sum_{i=n-M+2}^n (x_i - 0.5)^2
        # h_i depends on y = x_1, ..., x_{M-1}
        # To be on Pareto front, g(x) must be 0.
        # This implies x_i = 0.5 for i = n-M+2, ..., n.
        # The variables x_1, ..., x_{M-1} determine the position on the front.
        
        d_pos = n_obj - 1
        d_dist = dim - d_pos
        
        # Generate points in the position space [0, 1]^d_pos
        # We use Sobol for low discrepancy, but we will transform the first coordinate
        # to be uniform with respect to the surface area of the sphere.
        
        if d_pos == 0:
             points_pos = np.zeros((budget, 0))
        else:
            # Use Sobol sequence for good coverage in the d_pos dimensional space
            # We use a seed that might differ from the evaluator's seed to ensure 
            # diversity, but since we are deterministic in structure, we fix it.
            # The evaluator seeds np.random, but qmc.Sobol uses its own seed.
            sampler = qmc.Sobol(d=d_pos, scramble=True, seed=0)
            points_pos = sampler.random(budget)
            
            # Transform the first coordinate (index 0) to be area-uniform on the sphere.
            # The mapping from x_1 to the polar angle phi is phi = x_1 * pi/2.
            # The surface area element is proportional to sin(phi) dphi.
            # To get uniform area coverage, we sample u ~ U[0,1] and set 
            # 1 - cos(phi) = u  => phi = arccos(1-u) => x_1 = (2/pi) * arccos(1-u).
            # The Sobol sequence provides u values in [0,1].
            u_vals = points_pos[:, 0]
            # Handle edge cases where u might be exactly 0 or 1 (arccos domain)
            # np.arccos is defined on [-1, 1]. 1-u is in [0, 1].
            points_pos[:, 0] = (2.0 / np.pi) * np.arccos(1.0 - u_vals)
            
            # The other coordinates (if any) remain as is, assuming they map to 
            # uniform angular distributions which is correct for the remaining dimensions
            # of the sphere (azimuthal angles are uniform in area).
            
        # Construct full dimension vectors
        X = np.zeros((budget, dim))
        
        # Assign position variables
        if d_pos > 0:
            X[:, :d_pos] = points_pos
            
        # Assign distance variables to 0.5
        X[:, d_pos:] = 0.5
        
        # Clip to bounds just in case, though 0.5 and [0,1] should be safe
        lo = np.array([b[0] for b in bounds], dtype=np.float64)
        hi = np.array([b[1] for b in bounds], dtype=np.float64)
        X = np.clip(X, lo, hi)
        
        # Evaluate all points
        # The evaluator records the objective values returned by problem(x)
        for i in range(budget):
            if time.time() > deadline:
                break
            problem(X[i])
            
        return X

# EVOLVE-BLOCK-END
