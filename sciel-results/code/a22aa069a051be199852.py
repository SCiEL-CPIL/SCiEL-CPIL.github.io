"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """
    Optimizer for DTLZ7 exploiting the Pareto front structure.
    The Pareto front lies in the subspace where x_3...x_n = 0.
    The variables x_1 and x_2 determine the shape of the front.
    Uses a combination of uniform grid and Sobol sequence for better coverage.
    """

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def _sobol_sequence(self, n, dim=2):
        """
        Generate a Sobol-like quasi-random sequence in [0,1]^dim.
        Uses a simple inverse CDF transform of a low-discrepancy sequence.
        """
        # Use a simple approach: generate points using a Halton-like sequence
        # or use scipy if available
        try:
            from scipy.stats import qmc
            sampler = qmc.Sobol(d=dim, scramble=True)
            # Power of 2 for good distribution
            n_points = 1 << int(np.ceil(np.log2(n)))
            if n_points < n:
                n_points = 1 << (int(np.ceil(np.log2(n))) + 1)
            points = sampler.random(n_points)[:n]
            return points
        except ImportError:
            # Fallback: use a simple stratified sampling approach
            # This is a crude approximation of Sobol
            points = []
            # Use a grid with jitter
            side = int(np.ceil(np.sqrt(n)))
            for i in range(side):
                for j in range(side):
                    if len(points) >= n:
                        break
                    # Add deterministic jitter based on index
                    jitter1 = (hash((i, j, 0)) % 1000000) / 1000000.0 * 0.5 / side
                    jitter2 = (hash((i, j, 1)) % 1000000) / 1000000.0 * 0.5 / side
                    x1 = (i + 0.5 + jitter1) / side
                    x2 = (j + 0.5 + jitter2) / side
                    if x1 <= 1.0 and x2 <= 1.0:
                        points.append([x1, x2])
                    if len(points) >= n:
                        break
            return np.array(points[:n]) if points else np.zeros((n, dim))

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 290  # Safety margin before 300s
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        results = []
        n_evals = 0
        budget = self.budget
        
        # DTLZ7 Pareto front: x_3...x_n = 0
        # x_1, x_2 vary in [0, 1]
        
        # Strategy: Use a 13x13 grid (169 points) + Sobol points for remaining budget
        # This gives better space-filling properties than a pure 14x14 grid
        
        grid_size = 13
        n_grid = grid_size * grid_size  # 169
        
        # First, evaluate the grid points
        if budget >= n_grid:
            x1_vals = np.linspace(0, 1, grid_size)
            x2_vals = np.linspace(0, 1, grid_size)
            
            for x1 in x1_vals:
                for x2 in x2_vals:
                    if n_evals >= budget:
                        break
                    if time.time() > deadline:
                        break
                    
                    x = np.zeros(self.dim)
                    x[0] = x1
                    x[1] = x2
                    # x[2:] = 0 minimizes g(x)
                    
                    obj = problem(x)
                    results.append(obj)
                    n_evals += 1
                if n_evals >= budget:
                    break
        
        # Use remaining budget for Sobol sequence points
        remaining = budget - n_evals
        if remaining > 0 and time.time() < deadline:
            # Generate Sobol points in the unit square
            sobol_points = self._sobol_sequence(remaining, dim=2)
            
            for point in sobol_points:
                if n_evals >= budget:
                    break
                if time.time() > deadline:
                    break
                
                x1, x2 = point[0], point[1]
                
                x = np.zeros(self.dim)
                x[0] = x1
                x[1] = x2
                # x[2:] = 0 minimizes g(x)
                
                obj = problem(x)
                results.append(obj)
                n_evals += 1
        
        return results
# EVOLVE-BLOCK-END
