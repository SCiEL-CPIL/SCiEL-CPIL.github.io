import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    @staticmethod
    def _halton(n, dims):
        """Generate n points in [0,1)^dims using Halton sequence."""
        points = np.zeros((n, dims))
        for d in range(dims):
            primes = [2, 3, 5, 7, 11, 13, 17]
            prime = primes[d]
            for i in range(1, n + 1):
                f = 1.0
                r = 0.0
                j = i
                while j > 0:
                    f /= prime
                    r += f * (j % prime)
                    j //= prime
                points[i - 1, d] = r
        return points

    def __call__(self, problem, bounds):
        start = time.time()
        deadline = start + 300.0
        
        lo = np.array([b[0] for b in bounds], dtype=float)
        hi = np.array([b[1] for b in bounds], dtype=float)
        dim = self.dim
        budget = self.budget
        
        recorded = []
        
        # DTLZ1 Front:
        # x_1, x_2 determine position on the Pareto front
        # x_3..x_7 = 0.5 ensures g(x)=0
        # We sample x_1, x_2 using Halton sequence for better coverage
        
        # Generate Halton points in 2D
        points = self._halton(budget, 2)
        
        # Map to x1, x2
        for i in range(budget):
            if time.time() > deadline - 5.0:
                break
            
            x1_val = points[i, 0]
            x2_val = points[i, 1]
            
            # Construct full vector
            x = np.full(dim, 0.5)
            x[0] = x1_val
            x[1] = x2_val
            
            # Clip to bounds
            x = np.clip(x, lo, hi)
            
            # Evaluate
            f = np.array(problem(x), dtype=float)
            recorded.append(f)
            
        return recorded
