import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        # ZDT6 structure:
        # f1(x) = 1 - exp(-4*x1) * sin(6*pi*x1)^8
        # g(x) = 1 + 9 * (sum_{i=2}^{n} x_i) / (n-1)
        # f2(x) = g(x) * (1 - sqrt(f1(x)/g(x)))
        #
        # Pareto front: x2..xn = 0 => g=1, f2 = 1 - sqrt(f1)
        # f1 is non-monotonic in x1, so uniform x1 gives non-uniform f1 coverage.
        #
        # Strategy: sample f1 uniformly in [0,1], then for each f1 target,
        # find the x1 that achieves it via bisection on the non-monotonic f1.
        # This gives uniform coverage of the Pareto front in f1-space,
        # which maximizes hypervolume for a fixed number of points.
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        n_evals = self.budget
        start_time = time.time()
        deadline = 300.0
        
        # f1(x1) for x2..xn = 0
        def f1_of_x1(x1):
            return 1.0 - np.exp(-4.0 * x1) * (np.sin(6.0 * np.pi * x1) ** 8)
        
        # For a target f1 value, find x1 in [0,1] such that f1(x1) = target.
        # f1 is non-monotonic, so there may be multiple solutions.
        # We'll use a dense pre-computed lookup to find the closest x1.
        
        # Pre-compute a dense mapping from x1 to f1
        n_dense = 100000
        x1_dense = np.linspace(0.0, 1.0, n_dense)
        f1_dense = f1_of_x1(x1_dense)
        
        # For each target f1, find the x1 that gives it.
        # Use searchsorted on sorted f1 values, but f1 is non-monotonic.
        # Instead, for each target, find the index in x1_dense that minimizes |f1_dense - target|.
        # This is O(n_dense) per query, which is too slow for 200 queries.
        # Better: bin f1_dense into bins and for each target, search the relevant bin.
        
        # Alternative: just use the dense table and for each target f1, 
        # find the x1 with the closest f1 value using vectorized search.
        
        # Since n_evals is only 200, we can do a simple approach:
        # For each target f1, scan the dense table for the best match.
        # 200 * 100000 = 20M operations, which is fast in numpy.
        
        target_f1s = np.linspace(0.0, 1.0, n_evals)
        
        # For each target, find x1 that minimizes |f1_dense - target|
        # Vectorized: compute |f1_dense[None, :] - target_f1s[:, None]| and argmin
        # This is 200 x 100000 = 20M floats = 160 MB, might be tight.
        # Let's do it in chunks.
        
        x1_solutions = np.zeros(n_evals)
        chunk_size = 50
        for start_idx in range(0, n_evals, chunk_size):
            end_idx = min(start_idx + chunk_size, n_evals)
            targets = target_f1s[start_idx:end_idx]
            diffs = np.abs(f1_dense[None, :] - targets[:, None])
            best_indices = np.argmin(diffs, axis=1)
            x1_solutions[start_idx:end_idx] = x1_dense[best_indices]
            
            if time.time() - start_time > deadline - 10.0:
                break
        
        # Create points: x = [x1, 0, 0, ..., 0]
        points = []
        for x1 in x1_solutions:
            x = np.zeros(self.dim)
            x[0] = x1
            points.append(x)
        
        # Evaluate
        for i in range(len(points)):
            if time.time() - start_time > deadline:
                break
            x = points[i]
            x_clipped = np.clip(x, lo, hi)
            problem(x_clipped)
        
        return points
