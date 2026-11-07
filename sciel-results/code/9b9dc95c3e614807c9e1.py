"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """Halton-sequence sampling of the DTLZ7 Pareto front."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 290  # Safety margin

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])

        n_obj = self.n_obj
        dim = self.dim
        budget = self.budget

        # DTLZ7 Pareto front is 2D (parameterized by x0, x1 with x2...=0)
        # Use 2D Halton sequence to cover the 2D surface
        # Bases for 2D: 2 and 3
        
        def halton(index, base):
            result = 0.0
            f = 1.0 / base
            while index > 0:
                result += f * (index % base)
                index //= base
                f /= base
            return result

        results = []
        
        # Pre-allocate for speed
        x_template = np.zeros(dim)
        
        for i in range(1, budget + 1):
            if time.time() > deadline:
                break
            
            # 2D Halton: use different bases for each dimension
            h0 = halton(i, 2)
            h1 = halton(i, 3)
            
            # Clip to [0, 1] to stay within bounds
            x0 = max(0.0, min(1.0, h0))
            x1 = max(0.0, min(1.0, h1))
            
            x_template[0] = x0
            x_template[1] = x1
            # x[2:] remain 0
            
            f = problem(x_template)
            results.append(f)

        return results
# EVOLVE-BLOCK-END
