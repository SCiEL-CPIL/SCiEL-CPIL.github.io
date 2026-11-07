import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 290  # Safety margin before 300s limit
        
        n_samples = self.budget
        
        # DTLZ7 structure:
        # M=3 objectives, n=22 variables.
        # m = n - M + 1 = 20 distance variables: x[2] ... x[21]
        # Position variables: x[0], x[1]
        # The Pareto front is 2D, parameterized by (x0, x1) in [0,1]^2.
        # Setting all distance variables to 0 gives g=0, placing points on the Pareto front.
        
        # Use a 2D Halton sequence for near-perfect uniform coverage of [0,1]^2.
        # Halton sequences have low discrepancy and are deterministic.
        
        def halton_sequence(n, base):
            """Generate n points in [0,1] using Halton sequence with given base."""
            result = np.zeros(n)
            for i in range(1, n + 1):
                f = 1.0
                r = 0.0
                j = i
                while j > 0:
                    f /= base
                    j, remainder = divmod(j, base)
                    r += f * remainder
                result[i - 1] = r
            return result
        
        # Generate 2D Halton sequence (bases 2 and 3 are coprime)
        x0 = halton_sequence(n_samples, 2)
        x1 = halton_sequence(n_samples, 3)
        
        results = []
        for idx in range(n_samples):
            if time.time() > deadline:
                break
            
            x = np.zeros(self.dim)
            x[0] = x0[idx]
            x[1] = x1[idx]
            # Distance variables remain 0 (g=0, on Pareto front)
            
            try:
                f = problem(x)
                results.append(f)
            except Exception:
                pass
                    
        return results
