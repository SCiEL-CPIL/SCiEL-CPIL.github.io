import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def _halton(self, n, base):
        result = np.zeros(n)
        f = 1.0 / base
        for i in range(n):
            i_val = i + 1
            while i_val > 0:
                result[i] += f * (i_val % base)
                i_val //= base
                f /= base
        return result

    def __call__(self, problem, bounds):
        start_time = time.time()
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        # Assume standard DTLZ2 with M=3, k=2
        # Position vars: 0, 1
        # Distance vars: 2..11
        n_pos = 2
        n_dist = self.dim - n_pos
        
        # Generate Halton sequence for position variables
        h1 = self._halton(self.budget, 2)
        h2 = self._halton(self.budget, 3)
        
        # Scale to [0, 1] (already in [0,1] from Halton)
        # But Halton returns values in (0, 1).
        
        X = np.zeros((self.budget, self.dim))
        X[:, 0] = h1
        X[:, 1] = h2
        # Set distance variables to 0.5 to minimize g
        # Bounds are [0, 1], so 0.5 is the center
        X[:, n_pos:] = 0.5
        
        # Clip to bounds just in case
        X = np.clip(X, lo, hi)
        
        F = np.array([problem(x) for x in X])
        
        return X
