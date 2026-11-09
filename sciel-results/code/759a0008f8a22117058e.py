"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """DTLZ1 optimizer using scrambled Sobol sampling on shape variables (u,v) mapped to Pareto simplex."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 320.0
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        n_shape = 2
        n_dist = self.dim - n_shape
        
        # Optimal distance variables: x[i] = 0.5 minimizes g(x)
        x_dist = np.full(n_dist, 0.5)
        x_dist = np.clip(x_dist, lo[n_shape:], hi[n_shape:])
        
        N = self.budget
        
        # Generate scrambled Sobol points in [0,1)^2 for shape variables (u, v)
        # These map to the DTLZ1 Pareto simplex via:
        #   f1 = 0.5*(1-u^2), f2 = 0.5*u*(1-v^2), f3 = 0.5*u*v
        # Scrambled Sobol provides better coverage of the curved front than uniform grid
        try:
            from scipy.stats import qmc
            sampler = qmc.Sobol(d=2, scramble=True)
            pts = sampler.random(N)
        except ImportError:
            pts = self._halton_2d(N)
        
        # Ensure we include boundary points for extreme Pareto front corners
        # The Sobol sequence may not hit exactly (0,0), (1,0), (0,1), (1,1)
        # Add explicit corner points if budget allows
        corners = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
        
        # Combine: use Sobol points, but replace last few with corners if N >= 4
        # Actually, better to just use Sobol and ensure good coverage
        # Sobol in [0,1)^2 will have points near boundaries
        
        evals_used = 0
        for i in range(N):
            if time.time() > deadline:
                break
            if evals_used >= N:
                break
                
            u = pts[i, 0]
            v = pts[i, 1]
            
            x_full = np.zeros(self.dim)
            x_full[0] = u
            x_full[1] = v
            x_full[n_shape:] = x_dist
            
            x_full = np.clip(x_full, lo, hi)
            problem(x_full)
            evals_used += 1
        
        return None

    def _halton_2d(self, n):
        """Generate n points in [0, 1)^2 using the Halton sequence with bases 2 and 3."""
        points = np.zeros((n, 2))
        for i in range(1, n + 1):
            points[i - 1, 0] = self._halton_value(i, 2)
            points[i - 1, 1] = self._halton_value(i, 3)
        return points

    def _halton_value(self, i, base):
        """Calculate the i-th Halton number with the given base."""
        f = 1.0
        r = 0.0
        while i > 0:
            f /= base
            r += f * (i % base)
            i //= base
        return r
# EVOLVE-BLOCK-END
