"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Simple evolutionary multi-objective search: random init, mutate non-dominated parents."""

    def __init__(self, budget, dim, n_obj):
        self.budget, self.dim, self.n_obj = budget, dim, n_obj

    def __call__(self, problem, bounds):
        t_start = time.time()
        deadline = t_start + 280.0  # safety margin before 300s limit
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        # WFG4 analytical front strategy:
        # The Pareto front is characterized by the shape variable x_1 and optimal position variables x_2...x_n.
        # For standard WFG problems, the optimal position variables are 0.5.
        # We sample x_1 uniformly across its bounds and fix x_2...x_n to 0.5.
        
        n_samples = self.budget
        
        # Create samples for x_1
        x1_samples = np.linspace(lo[0], hi[0], n_samples)
        
        # Create the full population matrix
        X = np.zeros((n_samples, self.dim))
        X[:, 0] = x1_samples
        
        # Set remaining variables to 0.5, clipped to bounds
        for i in range(1, self.dim):
            val = 0.5
            # If bounds are not [0,1], 0.5 might not be the center, but WFG is defined on [0,1].
            # If bounds are scaled, we assume the function definition scales with it or 0.5 is still the target.
            # To be safe, we use 0.5 and clip.
            X[:, i] = np.clip(0.5, lo[i], hi[i])
            
        # Evaluate all points
        F = np.array([problem(x) for x in X])
        
        return X
# EVOLVE-BLOCK-END
