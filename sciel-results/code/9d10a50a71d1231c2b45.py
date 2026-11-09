"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """Multi-objective optimizer using Deterministic Latin Hypercube Sampling (LHS)."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 320.0  # Safety margin before 330s kill

        dim = self.dim
        budget = self.budget

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        # Strategy: Use the entire budget for a deterministic Latin Hypercube Sample.
        # This ensures uniform coverage of the 5D search space, maximizing the probability
        # of discovering the true Pareto front boundaries without the clustering artifacts
        # of stochastic local search or random sampling.
        
        # Generate LHS points
        # For each dimension, divide [0, 1] into 'budget' intervals.
        # Pick one random point in each interval, then permute for each dimension.
        
        # We use the global np.random which is seeded by the evaluator.
        # This makes the LHS design deterministic for a given seed.
        
        X = np.zeros((budget, dim))
        
        for d in range(dim):
            # Create intervals: [0, 1/budget), [1/budget, 2/budget), ...
            # Random offsets within each interval
            u = (np.arange(budget) + np.random.random(budget)) / budget
            # Permute the order for this dimension to ensure the Latin property
            perm = np.random.permutation(budget)
            X[:, d] = u[perm]
            
            # Map from [0, 1] to [lo[d], hi[d]]
            X[:, d] = lo[d] + X[:, d] * (hi[d] - lo[d])

        # Evaluate all points
        for i in range(budget):
            if time.time() > deadline:
                break
            problem(X[i])

        return None

# EVOLVE-BLOCK-END
