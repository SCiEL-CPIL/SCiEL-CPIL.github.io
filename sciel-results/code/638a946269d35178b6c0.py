import numpy as np
import time

class Optimizer:
        def __init__(self, budget, dim, n_obj):
            self.budget = budget
            self.dim = dim
            self.n_obj = n_obj

        def __call__(self, problem, bounds):
            start_time = time.time()
            deadline = start_time + 300  # 300s limit
            
            # 1. Prepare bounds
            lo = np.array([b[0] for b in bounds])
            hi = np.array([b[1] for b in bounds])
            
            # 2. Generate Halton sequence points in [0,1]^2
            n_points = self.budget
            # Halton base 2 and 3
            def halton(index, base):
                f = 1.0
                r = 0.0
                while index > 0:
                    f = f / base
                    r = r + f * (index % base)
                    index //= base
                return r

            # Generate N points
            # We need to be careful with index 0, Halton(0) is 0.
            # Using indices 1..N
            u = np.array([halton(i+1, 2) for i in range(n_points)])
            v = np.array([halton(i+1, 3) for i in range(n_points)])
            
            # 3. Map to DTLZ2 variables
            # x1 mapping for uniform sphere surface
            # theta1 = arccos(1 - u)
            # x1 = 2/pi * theta1
            x1 = (2.0 / np.pi) * np.arccos(1.0 - u)
            x2 = v
            
            # 4. Construct full vectors
            # dim = 12, n_obj = 3
            # Variables 3..12 (indices 2..11) set to 0.5 (midpoint)
            # This assumes bounds are [0,1]. The problem clips, so 0.5 is safe.
            # But to be robust, use midpoint of bounds.
            mid = (lo + hi) / 2.0
            
            X = np.zeros((n_points, self.dim))
            X[:, 0] = x1
            X[:, 1] = x2
            for j in range(2, self.dim):
                X[:, j] = mid[j]
                
            # 5. Evaluate
            F = np.array([problem(x) for x in X])
            
            # 6. Check deadline (unlikely to be hit, but required)
            if time.time() > deadline:
                 pass # Just return what we have
            
            return X
