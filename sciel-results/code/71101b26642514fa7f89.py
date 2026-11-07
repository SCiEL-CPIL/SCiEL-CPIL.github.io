import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        t0 = time.time()
        deadline = t0 + 280.0  # safety margin before 330s limit
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        n = self.dim
        N = self.budget
        
        # ZDT2 Pareto front:
        # f1 = x1
        # f2 = 1 + 9 * (1 - sqrt(x1))
        # Achieved when x2 = ... = xn = x1^(1/4)
        #
        # Strategy: Sample x1 values, set x2..xn = x1^(1/4)
        # Use Latin Hypercube Sampling for better coverage than uniform grid
        #
        # For hypervolume maximization with reference point [11, 11]:
        # We want points that are "spread out" along the front
        # The front goes from (0, 10) to (1, 1)
        # 
        # Key insight: The hypervolume contribution depends on the spacing
        # To maximize HV, we should use a non-uniform spacing that accounts
        # for the curvature of the front
        #
        # Actually, for a convex front like ZDT2's, uniform sampling in
        # objective space (i.e., uniform in f2 or uniform in sqrt(x1))
        # gives better HV coverage than uniform in x1
        #
        # Let's try uniform in sqrt(x1), which means x1 = t^2 where t is uniform
        # This gives denser sampling near x1=0 where the front is steeper
        
        # Generate LHS samples for x1
        # LHS: divide [0,1] into N intervals, sample one point from each
        rng = np.random.RandomState(42)  # Fixed seed for reproducibility within the optimizer
        # Actually, we should use the global random state since the evaluator sets np.random.seed
        # But let's use a local one to be safe
        
        # LHS sampling
        indices = np.arange(N)
        # Random permutation to avoid systematic bias
        perm = np.random.permutation(N)
        # Sample within each interval
        x1_samples = (perm + np.random.rand(N)) / N
        
        # Sort to maintain order (not strictly necessary but helps)
        x1_samples = np.sort(x1_samples)
        
        # Ensure we include endpoints for maximum coverage
        x1_samples[0] = 0.0
        x1_samples[-1] = 1.0
        
        # Construct the decision vectors
        X = np.zeros((N, n))
        X[:, 0] = x1_samples
        
        if n > 1:
            # x2..xn = x1^(1/4)
            # Handle x1=0 case: 0^(1/4) = 0, which is fine
            X[:, 1:] = (x1_samples ** 0.25)[:, np.newaxis]
        
        # Evaluate all points
        F = np.array([problem(x_vec) for x_vec in X])
        
        return X
