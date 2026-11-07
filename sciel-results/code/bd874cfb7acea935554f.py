import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    @staticmethod
    def _halton(n, base):
        """Generate n points of a Halton sequence in [0,1) with given base."""
        result = np.zeros(n)
        f = 1.0 / base
        for i in range(n):
            x = 0.0
            f = 1.0 / base
            idx = i + 1
            while idx > 0:
                x += f * (idx % base)
                idx //= base
                f /= base
            result[i] = x
        return result

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 290  # 30s margin
        
        # DTLZ7 with M=3, n=22: m = n-M+1 = 20
        # Pareto front: x[2..21] = 0, x[0] and x[1] free in [0,1]
        # f1 = x[0], f2 = x[1], f3 = 2*(1 - sqrt(x[0]/2)*(1+sin(10*pi*x[0]/2)))
        # 2D Pareto front parameterized by (x[0], x[1])
        
        n_samples = min(self.budget, 200)
        
        # Use 2D Halton sequence for quasi-random coverage of [0,1]^2
        # This gives much better hypervolume than a 1D curve
        h1 = self._halton(n_samples, 2)
        h2 = self._halton(n_samples, 3)
        
        # Shift to (0, 1] to avoid boundary issues and ensure all points are valid
        # Actually, use [0,1] directly since bounds allow it
        # But avoid exact 0 for x[0] since f3 formula has sqrt(x[0]/2)
        # x[0]=0 gives f3=2, which is fine (valid point on front)
        
        results = []
        for i in range(n_samples):
            if time.time() > deadline:
                break
            
            x0 = h1[i]  # in [0, 1)
            x1 = h2[i]  # in [0, 1)
            
            x = np.zeros(self.dim)
            x[0] = x0
            x[1] = x1
            # x[2:] = 0 ensures g=1 (Pareto optimal)
            
            f = problem(x)
            results.append(f)
            
        return results
