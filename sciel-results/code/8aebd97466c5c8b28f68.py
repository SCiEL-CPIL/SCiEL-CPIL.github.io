"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


class Optimizer:
    """Optimize ZDT6 by sampling x1 values that cover the Pareto front."""

    def __init__(self, budget, dim, n_obj):
        self.budget, self.dim, self.n_obj = budget, dim, n_obj

    def __call__(self, problem, bounds):
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        dim = self.dim
        budget = self.budget
        
        # ZDT6: only x1 affects objectives
        # f1 = 1 - exp(-4*x1) * sin(10*pi*x1)
        # f2 = 1 - sqrt(f1)
        # Sample x1 values to cover the Pareto front well
        
        # Use budget evaluations to sample x1 values
        # Spread them to maximize hypervolume coverage
        
        # The Pareto front is parameterized by x1 in [0,1]
        # f1 ranges from 0 to 1, f2 = 1 - sqrt(f1)
        
        # To maximize hypervolume, we want points spread along the front
        # Use a combination of uniform and clustered sampling
        
        # Generate candidate x1 values
        # Use stratified sampling for better coverage
        n_samples = budget
        
        # Create x1 values: mix of uniform grid and random
        # The front is concave, so we want denser sampling where the curve bends more
        
        # Actually, let's just use uniform sampling of x1 since the front is smooth
        x1_values = np.linspace(0.0, 1.0, n_samples)
        
        # Add some random perturbation to avoid exact duplicates if any
        # But linspace should be fine
        
        results = []
        for i in range(n_samples):
            x = np.full(dim, 0.5)  # x2..x10 don't matter, set to middle
            x[0] = x1_values[i]
            # Clip to bounds
            x = np.clip(x, lo, hi)
            f = problem(x)
            results.append(f)
        
        return results
# EVOLVE-BLOCK-END
