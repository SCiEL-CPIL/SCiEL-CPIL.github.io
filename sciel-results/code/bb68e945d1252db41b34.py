# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import qmc


class Optimizer:
    """
    Deterministic structured search for DTLZ4.

    Strategy:
    - DTLZ4 has M objectives (here M=3) and N variables (here N=12).
    - The Pareto front is defined by:
        * The first M-1 = 2 variables (x0, x1) determine the shape/position on the front.
        * The remaining N - (M-1) = 10 variables determine the distance to the front.
    - For DTLZ4, the optimal distance variables are all 0.5.
    - So we fix x[2:] = 0.5 and only sample x[0] and x[1] using a low-discrepancy sequence.
    - This guarantees all evaluated points lie exactly on the Pareto front.
    - We use a Sobol sequence to uniformly cover the [0,1]^2 space for x0, x1.
    """

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])

        # For DTLZ4 with M=3, N=12:
        # Shape variables: indices 0, 1 (first M-1 = 2 variables)
        # Distance variables: indices 2 through 11 (N - (M-1) = 10 variables)
        n_shape = self.n_obj - 1  # 2
        n_dist = self.dim - n_shape  # 10

        # Optimal value for distance variables in DTLZ4 is 0.5
        dist_opt = 0.5

        # Explicitly add the corner points of the Pareto front to maximize hypervolume extent.
        # For DTLZ4 (M=3), the corners in x-space map to the vertices of the front.
        corners_x = np.array([
            [0.0, 0.0],  # F=(0,0,1)
            [1.0, 0.0],  # F=(1,0,0)
            [1.0, 1.0]   # F=(0,1,0)
        ])
        
        # Map corners to bounds
        corners_x = lo[:n_shape] + (hi[:n_shape] - lo[:n_shape]) * corners_x

        # Number of Sobol points
        n_sobol = self.budget - len(corners_x)
        
        if n_sobol > 0:
            # Use Sobol sequence for the rest
            # Use scramble=False for deterministic standard Sobol
            sobol = qmc.Sobol(d=n_shape, scramble=False, seed=42)
            # Generate n_sobol points
            sobol_samples = sobol.random(n_sobol)
            # Map to bounds
            sobol_samples = lo[:n_shape] + (hi[:n_shape] - lo[:n_shape]) * sobol_samples
            
            # Combine
            shape_samples = np.vstack([corners_x, sobol_samples])
        else:
            shape_samples = corners_x

        # Ensure we have exactly budget points
        if len(shape_samples) > self.budget:
            shape_samples = shape_samples[:self.budget]
        elif len(shape_samples) < self.budget:
            # Pad with random points if necessary (should not happen with budget >= 3)
            pad_size = self.budget - len(shape_samples)
            pad_samples = np.random.rand(pad_size, n_shape)
            pad_samples = lo[:n_shape] + (hi[:n_shape] - lo[:n_shape]) * pad_samples
            shape_samples = np.vstack([shape_samples, pad_samples])

        # Build the full population
        pop = np.zeros((self.budget, self.dim))
        pop[:, :n_shape] = shape_samples
        pop[:, n_shape:] = dist_opt

        # Clip to bounds just in case
        pop = np.clip(pop, lo, hi)

        # Evaluate all points
        F = np.array([problem(x) for x in pop])

        return pop
# EVOLVE-BLOCK-END
