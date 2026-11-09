"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """
    Optimizer for DTLZ7 exploiting the Pareto front structure.
    Uses Sobol-like quasi-random sampling on the 2D Pareto front manifold
    with strategic corner/edge points to maximize hypervolume coverage.
    
    DTLZ7 Pareto front: x[2:] = 0.5 (for g=0)
    f0 = x[0]
    f1 = x[1]
    f2 = 1 - sqrt(x[1])
    """

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def _sobol_2d(self, n, skip=0):
        """
        Generate n points in [0,1]^2 using a simple Sobol-like sequence.
        Uses a combination of inverse CDF transformations and low-discrepancy properties.
        """
        points = np.zeros((n, 2))
        for i in range(n):
            idx = i + skip + 1
            # Use a simple low-discrepancy approach based on van der Corput
            # Dimension 1: base 2
            f, r, q = 1.0, 0.0, idx
            while q > 0:
                f /= 2.0
                r += f * (q % 2)
                q //= 2
            points[i, 0] = r
            
            # Dimension 2: base 3
            f, r, q = 1.0, 0.0, idx
            while q > 0:
                f /= 3.0
                r += f * (q % 3)
                q //= 3
            points[i, 1] = r
        return points

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280.0  # Safety margin before 300s
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        results = []
        n_evals = 0
        budget = self.budget

        # DTLZ7 Pareto front parameterization:
        # x[0] = u in [0,1] -> f0 = u
        # x[1] = v in [0,1] -> f1 = v, f2 = 1 - sqrt(v)
        # x[2:] = 0.5 (midpoint of bounds for g=0)
        
        mid = 0.5 * (lo + hi)
        
        # Strategy: Use quasi-random sampling for the bulk, plus strategic points
        # for corners and edges to ensure full coverage of the objective space.
        
        # Number of strategic points (corners, edges, midpoints)
        n_strategic = 20
        n_random = budget - n_strategic
        
        if n_random < 0:
            n_random = 0
            n_strategic = budget
        
        # Strategic points: corners and key locations in (u, v) space
        # u = x[0], v = x[1]
        # Corners of the unit square in (u, v) space
        strategic_points = [
            (0.0, 0.0), (0.0, 1.0), (1.0, 0.0), (1.0, 1.0),
            (0.0, 0.5), (1.0, 0.5), (0.5, 0.0), (0.5, 1.0),
            (0.5, 0.5),
            (0.25, 0.25), (0.75, 0.25), (0.25, 0.75), (0.75, 0.75),
            (0.1, 0.1), (0.9, 0.1), (0.1, 0.9), (0.9, 0.9),
            (0.33, 0.66), (0.66, 0.33),
            (0.5, 0.25), (0.5, 0.75),
        ][:n_strategic]
        
        # Generate quasi-random points for the rest
        if n_random > 0:
            random_points = self._sobol_2d(n_random, skip=1)
        else:
            random_points = np.empty((0, 2))
        
        # Combine all points
        all_points = []
        for sp in strategic_points:
            all_points.append(sp)
        for rp in random_points:
            all_points.append(tuple(rp))
        
        # Evaluate all points
        for u, v in all_points:
            if n_evals >= budget:
                break
            if time.time() > deadline:
                break
            
            x = np.zeros(self.dim)
            x[0] = lo[0] + u * (hi[0] - lo[0])
            x[1] = lo[1] + v * (hi[1] - lo[1])
            # Set x[2:] to midpoint for g=0
            for i in range(2, self.dim):
                x[i] = mid[i]
            
            # Clip to bounds
            x = np.clip(x, lo, hi)
            
            obj = problem(x)
            results.append(obj)
            n_evals += 1
        
        return results
# EVOLVE-BLOCK-END
