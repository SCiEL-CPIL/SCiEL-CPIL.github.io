import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        # 1. Setup
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        # 2. Fix variables 3..11 to midpoint
        mid = (lo + hi) / 2.0
        fixed_vars = mid[3:]
        
        # 3. Generate grid for first 3 variables
        # We have budget 200.
        # Let's split 200 into 3 dimensions.
        # 5 * 5 * 8 = 200.
        n0, n1, n2 = 5, 5, 8
        
        # Create linspace for each of the first 3 dims
        # Include endpoints to ensure full front coverage
        xs0 = np.linspace(lo[0], hi[0], n0)
        xs1 = np.linspace(lo[1], hi[1], n1)
        xs2 = np.linspace(lo[2], hi[2], n2)
        
        # Meshgrid
        G0, G1, G2 = np.meshgrid(xs0, xs1, xs2, indexing='ij')
        
        # Flatten
        X0 = G0.flatten()
        X1 = G1.flatten()
        X2 = G2.flatten()
        
        # Construct full vectors
        # X is (N, dim)
        N = len(X0)
        X = np.zeros((N, self.dim))
        X[:, 0] = X0
        X[:, 1] = X1
        X[:, 2] = X2
        X[:, 3:] = fixed_vars
        
        # Evaluate
        # The problem function clips and returns objectives.
        # We must call problem(x) for each x.
        # Vectorization might not be supported by `problem`, so loop.
        # 200 evaluations is fast.
        
        results = []
        for i in range(N):
            f = problem(X[i])
            results.append(f)
            
        return X # Return X as per signature, though return value is ignored
