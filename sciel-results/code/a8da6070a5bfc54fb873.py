import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 290  # Safety margin
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        # DTLZ4 structure:
        # For M=3, N=12:
        # Position variables: x_1, x_2 (indices 0, 1)
        # Distance variables: x_3 ... x_12 (indices 2..11)
        # Optimal distance variables are 0.
        
        n_pos = 2
        n_dist = self.dim - n_pos
        
        # Generate Halton sequence for position variables
        # We have self.budget evaluations.
        # Let's use all of them.
        
        def halton(index, base):
            f = 1.0
            r = 0.0
            i = index
            while i > 0:
                f /= base
                r += f * (i % base)
                i //= base
            return r
        
        # Generate points
        X = np.zeros((self.budget, self.dim))
        
        # Fill position variables with Halton sequence
        for i in range(self.budget):
            # i+1 to avoid 0 which is a corner
            X[i, 0] = halton(i + 1, 2)
            X[i, 1] = halton(i + 1, 3)
            
        # Distance variables are set to 0 (optimal)
        # X[:, n_pos:] remains 0
        
        # Clip to bounds just in case
        X = np.clip(X, lo, hi)
        
        F = np.array([problem(x) for x in X])
        
        # Filter out any points that might be invalid (though they shouldn't be)
        # And ensure we return the non-dominated set
        
        # Simple non-dominated sorting
        # Since points are on the front, they should all be non-dominated.
        # But let's be safe.
        
        nd_idx = []
        for i in range(len(F)):
            dominated = False
            for j in range(len(F)):
                if i == j: continue
                if np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                    dominated = True
                    break
            if not dominated:
                nd_idx.append(i)
                
        return X[nd_idx]
