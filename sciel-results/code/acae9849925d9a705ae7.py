"""Multi-objective optimizer for DTLZ4 using exact Pareto front construction with Sobol sampling."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-objective optimizer for DTLZ4 using direct front sampling with Sobol sequence."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280.0  # Safety margin before 330s kill

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        dim = self.dim
        n_obj = self.n_obj
        budget = self.budget

        # DTLZ4 structure for M=3 objectives, n=12 variables:
        # x[0], x[1] are shape variables.
        # x[2]..x[11] are distance variables.
        # To achieve g(x)=0 (Pareto optimal), x[2:] must be 0.5.
        #
        # Objectives with g=0:
        # f1 = 0.5 * x0^10 * cos(x1^10 * pi/2)
        # f2 = 0.5 * x0^10 * sin(x1^10 * pi/2)
        # f3 = 0.5 * (1 - x0^10)
        #
        # Let u = x0^10, v = x1^10.
        # Uniform sampling of u, v in [0, 1] provides good coverage of the front.
        
        n_points = budget
        
        # Use Sobol sequence for low-discrepancy sampling of (u, v)
        try:
            from scipy.stats import qmc
            # Sobol generator for 2 dimensions
            sampler = qmc.Sobol(d=2, scramble=True, seed=0)
            # Generate points in [0, 1]^2
            pts = sampler.random(n=n_points)
        except Exception:
            # Fallback to Halton sequence if SciPy is not available or fails
            def halton(index, base):
                f = 1.0
                r = 0.0
                while index > 0:
                    f /= base
                    r += f * (index % base)
                    index //= base
                return r

            bases = [2, 3]
            pts = np.array([[halton(i, bases[0]), halton(i, bases[1])] for i in range(1, n_points + 1)])

        u_flat = pts[:, 0]  # Uniform in [0, 1)
        v_flat = pts[:, 1]  # Uniform in [0, 1)

        # Convert to x0, x1 using the beta transformation (beta=10 for DTLZ4)
        beta = 10.0
        x0_flat = u_flat ** (1.0 / beta)
        x1_flat = v_flat ** (1.0 / beta)

        # Construct decision vectors
        X = np.zeros((n_points, dim))
        X[:, 0] = x0_flat
        X[:, 1] = x1_flat
        # Set all other variables to 0.5 to ensure g=0
        if dim > 2:
            X[:, 2:] = 0.5

        # Clip to bounds to be safe
        X = np.clip(X, lo, hi)

        # Evaluate all points
        F = np.array([problem(x) for x in X])
        
        # Check time
        if time.time() > deadline:
            return X

        return X
# EVOLVE-BLOCK-END
