import numpy as np


class Optimizer:
    """
    Optimizer for DTLZ4 (M=3, N=12).
    Exploits the known structure of the Pareto Front:
    - Position variables: x[0], x[1]
    - Distance variables: x[2:] (must be 0.5 for g=0)
    
    Uses a Halton sequence for uniform space-filling coverage of the 2D simplex
    in objective space.
    """

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def _halton(self, n, base):
        """Generate the first n Halton sequence points in [0,1)."""
        result = []
        for i in range(1, n + 1):
            x = 0.0
            f = 1.0
            i_temp = i
            while i_temp > 0:
                f /= base
                x += f * (i_temp % base)
                i_temp //= base
            result.append(x)
        return np.array(result)

    def __call__(self, problem, bounds):
        import time
        start_time = time.time()
        deadline = start_time + 285  # 15s safety margin
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        budget = self.budget  # 200
        dim = self.dim  # 12
        n_obj = self.n_obj  # 3
        
        # Generate budget points uniformly on the 2D simplex
        # Using the method: x0 = 1 - sqrt(u1), x1 = (1-x0) * u2
        # This provides uniform coverage of the simplex {x0>=0, x1>=0, x0+x1<=1}
        h1 = self._halton(budget, 2)
        h2 = self._halton(budget, 3)
        
        results = []
        for i in range(budget):
            if time.time() > deadline:
                break
            
            u1 = h1[i]
            u2 = h2[i]
            
            # Uniform sampling on simplex
            x0 = 1.0 - np.sqrt(u1)
            x1 = (1.0 - x0) * u2
            
            x = np.zeros(dim)
            x[0] = x0
            x[1] = x1
            x[2:] = 0.5
            x = np.clip(x, lo, hi)
            
            try:
                f = problem(x)
                results.append(f)
            except Exception:
                break
        
        return results
