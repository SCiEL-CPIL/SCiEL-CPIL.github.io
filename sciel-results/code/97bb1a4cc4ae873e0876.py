import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = 320.0  # safety margin before 330s
        eval_count = 0
        all_F = []
        all_X = []
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        rng = np.random
        
        def evaluate(x):
            nonlocal eval_count
            if eval_count >= self.budget:
                raise RuntimeError("Budget exceeded")
            eval_count += 1
            f = problem(x)
            all_F.append(np.asarray(f, dtype=float))
            all_X.append(np.asarray(x, dtype=float).copy())
            return f
        
        # Phase 1: Latin Hypercube Sampling for uniform coverage
        # LHS ensures each dimension is uniformly sampled across its range
        n_lhs = min(200, self.budget)
        
        # Generate LHS points
        # For each dimension, partition [0,1] into n_lhs equal intervals
        # and sample one point uniformly from each interval
        # Then permute the intervals for each dimension independently
        lhs_points = np.zeros((n_lhs, self.dim))
        for d in range(self.dim):
            # Create intervals: [i/n_lhs, (i+1)/n_lhs) for i in range(n_lhs)
            intervals = np.arange(n_lhs) / n_lhs
            # Sample one point from each interval
            samples = intervals + rng.random(n_lhs) / n_lhs
            # Permute to decorrelate dimensions
            perm = rng.permutation(n_lhs)
            lhs_points[:, d] = samples[perm]
        
        # Map from [0,1]^dim to bounds
        for i in range(n_lhs):
            if eval_count >= self.budget:
                break
            x = lo + (hi - lo) * lhs_points[i]
            evaluate(x)
        
        # Phase 2: If budget remains, add random points for fine-tuning
        while eval_count < self.budget:
            if time.time() > deadline:
                break
            x = lo + (hi - lo) * rng.random(self.dim)
            evaluate(x)
        
        return np.array(all_X)
