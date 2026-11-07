# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-objective optimizer for ZDT6 using NSGA-II with time guard."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 300.0

        lo = np.array([b[0] for b in bounds], dtype=np.float64)
        hi = np.array([b[1] for b in bounds], dtype=np.float64)
        
        n_samples = self.budget
        
        # Generate Latin Hypercube Sample
        # This ensures good spread in each dimension, which is crucial for 
        # hypervolume calculation in multi-objective optimization.
        X = np.zeros((n_samples, self.dim))
        
        for d in range(self.dim):
            # Create bins
            bins = np.linspace(lo[d], hi[d], n_samples + 1)
            # Randomize the order of bins
            perm = np.random.permutation(n_samples)
            # For each bin, pick a random point within that bin
            # We can do this vectorized:
            # For each sample i, it falls into bin perm[i].
            # The bin boundaries are bins[perm[i]] and bins[perm[i]+1]
            # Random offset within the bin:
            offsets = np.random.random(n_samples)
            X[:, d] = bins[perm] + offsets * (bins[perm + 1] - bins[perm])
        
        # Evaluate all points
        # The evaluator records the results of these calls.
        for i in range(n_samples):
            if time.time() > deadline - 5.0:
                break
            problem(X[i])
            
        return X
# EVOLVE-BLOCK-END
