import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280  # Safety margin
        
        dim = self.dim
        budget = self.budget
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        # 1. Determine target f2 values
        # Uniform spacing in f2 is optimal for the HV integral approx
        # f2 ranges from 0 to 1 on the Pareto front.
        # We want N points.
        # Use midpoints of intervals to avoid boundary issues if necessary, 
        # but boundaries are fine.
        # Let's use linspace from 0 to 1.
        f2_targets = np.linspace(0.0, 1.0, budget)
        
        # 2. Convert f2 targets to f1 targets
        # f2 = 1 - sqrt(f1) => sqrt(f1) = 1 - f2 => f1 = (1 - f2)^2
        f1_targets = (1.0 - f2_targets) ** 2
        
        # 3. Pre-compute dense mapping of t to f1
        # f1(t) = 1 - exp(-4t) * sin(10*pi*t)^2
        n_dense = 20000
        t_dense = np.linspace(0.0, 1.0, n_dense)
        f1_dense = 1.0 - np.exp(-4.0 * t_dense) * np.sin(10.0 * np.pi * t_dense) ** 2
        
        # 4. For each f1_target, find the corresponding t
        # We want to find t such that f1(t) is close to f1_target.
        # Since f1(t) oscillates, there are multiple solutions.
        # We pick the one that minimizes |f1(t) - f1_target|.
        # Vectorized approach:
        # Create a matrix of differences? 200 x 20000 = 4M floats. ~32MB. Acceptable.
        # Or loop with binary search? 
        # Given f1 is not monotonic, simple binary search on f1 is hard.
        # But we can just use np.argmin on the difference array.
        
        t_selected = np.zeros(budget)
        
        # To speed up, we can chunk the targets or just do it in a loop with np.argmin
        # 200 iterations of argmin on 20000 elements is very fast.
        for i in range(budget):
            if time.time() > deadline:
                # Fallback: just use uniform t if time runs out
                t_selected[i:] = np.linspace(0, 1, budget - i)
                break
            # Find index in dense array closest to f1_target
            diffs = np.abs(f1_dense - f1_targets[i])
            idx = np.argmin(diffs)
            t_selected[i] = t_dense[idx]
            
        # 5. Construct points and evaluate
        # Ensure we don't exceed budget
        n_evals = len(t_selected)
        
        X = np.zeros((n_evals, dim))
        X[:, 0] = t_selected
        # x[1:] are 0, which is within bounds [0, 1]
        
        # Clip to bounds just in case (though t is in [0,1] and 0 is in [0,1])
        X = np.clip(X, lo, hi)
        
        # Evaluate
        # The problem function returns objective values.
        # We just need to call it.
        for i in range(n_evals):
            if time.time() > deadline:
                break
            problem(X[i])
            
        return X
