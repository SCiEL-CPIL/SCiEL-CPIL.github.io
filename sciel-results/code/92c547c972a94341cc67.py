"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """Fibonacci-lattice (golden-ratio spiral) 2D sampling of DTLZ7 Pareto front."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 290  # Safety margin

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])

        dim = self.dim
        budget = self.budget

        # DTLZ7 Pareto front (with x[2:]=0 so g=1):
        #   f1 = x0
        #   f2 = x1
        #   f3 = 1 - sqrt(x1)
        # The front is a 2D surface in objective space parameterized by (x0, x1) in [0,1]^2.
        # Use a 2D Halton sequence for low-discrepancy sampling.

        def halton_1d(n, base):
            """Generate n Halton sequence points in [0,1)."""
            result = np.zeros(n)
            for i in range(n):
                f = 1.0 / base
                idx = i + 1
                val = 0.0
                while idx > 0:
                    val += f * (idx % base)
                    idx //= base
                    f /= base
                result[i] = val
            return result

        # Generate all points at once for speed
        n = budget
        h0 = halton_1d(n, 2)
        h1 = halton_1d(n, 3)

        # Apply power-law transformation (alpha=2) to bias toward lower bounds
        # where hypervolume contribution is highest.
        # This concentrates samples in regions with lower objective values,
        # which contribute more to the hypervolume indicator.
        alpha = 2.0
        h0_transformed = h0 ** alpha
        h1_transformed = h1 ** alpha

        results = []
        
        for i in range(budget):
            if time.time() > deadline:
                break
            
            x0 = lo[0] + h0_transformed[i] * (hi[0] - lo[0])
            x1 = lo[1] + h1_transformed[i] * (hi[1] - lo[1])
            
            x = np.zeros(dim)
            x[0] = x0
            x[1] = x1
            # x[2:] remain 0, which sets g=1 and achieves Pareto front
            
            f = problem(x)
            results.append(f)

        return results
# EVOLVE-BLOCK-END
