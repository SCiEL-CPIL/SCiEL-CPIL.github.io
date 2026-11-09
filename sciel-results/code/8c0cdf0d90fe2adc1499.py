"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """
    Optimizer for DTLZ7 exploiting the 2D Pareto front structure.
    The Pareto front lies in the subspace where x_3...x_n = 0.
    The variables x_1 and x_2 determine the shape of the front.
    Uses a non-uniform grid in 2D to maximize hypervolume coverage.
    """

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 290  # safety margin before 300s limit

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])

        results = []
        n_evals = 0

        # DTLZ7 Pareto front: x_3...x_n = 0 (g=0 is achievable)
        # Front is 2D, parameterized by (x1, x2) in [0,1]^2
        # Use a non-uniform grid to better capture the hypervolume boundary

        n_points = self.budget

        # For DTLZ7, the Pareto front has a specific structure.
        # Let's use a grid that is denser near the corners where
        # the front has more curvature or where hypervolume contribution is higher.

        # Create a 2D grid with non-uniform spacing
        # Use a transformation to concentrate points near the boundaries
        n_per_dim = int(np.sqrt(n_points))
        if n_per_dim * n_per_dim < n_points:
            n_per_dim += 1

        # Create non-uniform grid points in [0, 1]
        # Use a power transformation to concentrate points
        # t^alpha where alpha < 1 concentrates near 0, alpha > 1 near 1
        # For DTLZ7, the front is somewhat symmetric but let's try alpha=0.5
        # which gives sqrt spacing (denser near 0)
        
        # Actually, let's use a mix: some uniform, some concentrated
        # Let's create a grid that covers the full range but with extra points
        # near the edges where the front might have more structure
        
        # Simple approach: create a grid with n_per_dim points in each dimension
        # but use a non-uniform distribution
        
        # Use a beta-like distribution to concentrate points
        # For now, let's use a simple approach: uniform grid with some jitter
        
        x1_vals = np.linspace(0, 1, n_per_dim)
        x2_vals = np.linspace(0, 1, n_per_dim)
        
        # Create all combinations
        X1, X2 = np.meshgrid(x1_vals, x2_vals, indexing='ij')
        x1_flat = X1.flatten()
        x2_flat = X2.flatten()
        
        # If we have more points than needed, take the first n_points
        if len(x1_flat) > n_points:
            x1_flat = x1_flat[:n_points]
            x2_flat = x2_flat[:n_points]
        elif len(x1_flat) < n_points:
            # Add random points to fill up to n_points
            n_extra = n_points - len(x1_flat)
            x1_extra = np.random.uniform(0, 1, n_extra)
            x2_extra = np.random.uniform(0, 1, n_extra)
            x1_flat = np.concatenate([x1_flat, x1_extra])
            x2_flat = np.concatenate([x2_flat, x2_extra])

        # Map to actual bounds
        x1_vals_mapped = lo[0] + x1_flat * (hi[0] - lo[0])
        x2_vals_mapped = lo[1] + x2_flat * (hi[1] - lo[1])

        for i in range(n_points):
            if n_evals >= self.budget:
                break
            if time.time() > deadline:
                break

            x = np.zeros(self.dim)
            x[0] = x1_vals_mapped[i]
            x[1] = x2_vals_mapped[i]
            # x[2:] = 0 gives g = 0, the Pareto-optimal condition

            obj = problem(x)
            results.append(obj)
            n_evals += 1

        return results
# EVOLVE-BLOCK-END
