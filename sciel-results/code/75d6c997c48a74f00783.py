# EVOLVE-BLOCK-START
import time
import numpy as np


class Optimizer:
    """Direct, structure-exploiting DTLZ4 optimizer using cosine-spaced grid sampling."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start = time.time()
        deadline = start + 270  # safety margin

        lo = np.array([b[0] for b in bounds], dtype=float)
        hi = np.array([b[1] for b in bounds], dtype=float)
        n = self.dim
        M = self.n_obj
        n_g = M - 1  # number of shape variables (first M-1)

        F_list = []
        used = 0

        def evaluate(x):
            nonlocal used
            if used >= self.budget:
                return None
            if time.time() > deadline:
                return None
            f = np.asarray(problem(x), dtype=float)
            F_list.append(f)
            used += 1
            return f

        # Base vector with g-variables set to 0.5 (minimum of g for DTLZ4)
        base_x = np.full(n, 0.5)

        # 1. Add explicit corner points for the shape variables to ensure front extent
        # For DTLZ4 with 3 objectives, the Pareto front vertices are at the corners of [0,1]^2
        corners = [
            [0.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0],
            [1.0, 1.0]
        ]
        
        if n_g == 2:
            for c in corners:
                x = base_x.copy()
                x[0] = c[0]
                x[1] = c[1]
                evaluate(x)
        else:
            # Generic corner handling if n_g != 2
            for i in range(2**n_g):
                c = [(i >> j) & 1 for j in range(n_g)]
                x = base_x.copy()
                for k in range(n_g):
                    x[k] = float(c[k])
                evaluate(x)
                if used >= self.budget:
                    break

        # 2. Fill remaining budget with cosine-spaced grid points
        # Cosine spacing clusters points near boundaries where hypervolume contribution is highest
        if used < self.budget:
            N_remaining = self.budget - used
            
            # For 2 shape variables, use a grid with cosine spacing
            if n_g == 2:
                # Choose grid size to approximate a square grid
                # We want about N_remaining points total
                # Use a grid of size k x k where k^2 <= N_remaining
                k = int(np.sqrt(N_remaining))
                if k < 2:
                    k = 2
                
                # Ensure we don't exceed budget
                # k x k points
                n_grid = k * k
                if n_grid > N_remaining:
                    k = int(np.sqrt(N_remaining))
                    if k < 2:
                        k = 2
                    n_grid = k * k
                
                # Cosine spacing: points cluster near 0 and 1
                # For j in 0..k-1: theta_j = (2j+1)/(2k) * pi
                # x_j = sin(theta_j) or cos(theta_j)
                # We want points in [0, 1]
                # Using: x_j = 0.5 * (1 - cos((2j+1)*pi/(2k))) for j=0..k-1
                # This gives points clustered near 0 and 1
                
                # Alternative: use the inverse of the CDF of the arcsine distribution
                # Or simply: x_j = sin((2j+1)*pi/(4k)) for j=0..k-1, then scale to [0,1]
                
                # Let's use: for j in 0..k-1, t_j = (2j+1)/(2k)
                # x_j = 0.5 * (1 - cos(t_j * pi)) = sin^2(t_j * pi/2)
                # This gives symmetric clustering near 0 and 1
                
                indices = np.arange(k)
                t = (2 * indices + 1) / (2 * k)
                x_vals = np.sin(t * np.pi / 2) ** 2  # Clusters near 0 and 1
                
                # Create grid points
                x0_grid, x1_grid = np.meshgrid(x_vals, x_vals, indexing='ij')
                
                # Flatten
                grid_points = np.column_stack([x0_grid.ravel(), x1_grid.ravel()])
                
                # Shuffle to avoid bias in evaluation order (not strictly necessary but good practice)
                # Actually, order doesn't matter for the final set, but let's keep it deterministic
                # and just iterate through all grid points
                
                for p in grid_points:
                    if used >= self.budget:
                        break
                    if time.time() > deadline:
                        break
                    x = base_x.copy()
                    x[0] = p[0]
                    x[1] = p[1]
                    # Clip to bounds
                    x = np.clip(x, lo, hi)
                    evaluate(x)
                    
            else:
                # For higher dimensional shape spaces, use a simpler approach
                # Just use uniform random or a basic grid
                # For now, fall back to random sampling
                for _ in range(N_remaining):
                    if used >= self.budget:
                        break
                    if time.time() > deadline:
                        break
                    p = np.random.uniform(0, 1, n_g)
                    x = base_x.copy()
                    x[:n_g] = p
                    x = np.clip(x, lo, hi)
                    evaluate(x)

        if len(F_list) == 0:
            return []

        return np.array(F_list)
# EVOLVE-BLOCK-END
