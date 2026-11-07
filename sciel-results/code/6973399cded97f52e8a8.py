import numpy as np

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        n_lhs = self.budget
        
        # 1. Generate LHS samples using global np.random
        # Implementation of simple LHS
        lhs_samples = np.zeros((n_lhs, self.dim))
        for d in range(self.dim):
            bin_width = (hi[d] - lo[d]) / n_lhs
            # Random points in each bin
            offsets = np.random.random(n_lhs) * bin_width
            points_in_bin = lo[d] + (np.arange(n_lhs) + offsets) * bin_width
            # Shuffle
            np.random.shuffle(points_in_bin)
            lhs_samples[:, d] = points_in_bin
        
        # Evaluate all samples
        # The evaluator records the objective values returned by problem(x).
        # We do not need to maintain an archive for the return value.
        for i in range(n_lhs):
            x = lhs_samples[i]
            problem(x)
            
        return None
