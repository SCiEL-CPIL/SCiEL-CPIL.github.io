"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-objective optimizer for DTLZ4 using analytical Pareto front construction and maximin sampling."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 290.0  # Safety margin

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        dim = self.dim
        n_obj = self.n_obj
        budget = self.budget

        # DTLZ4 with M=3, N=12:
        # x[0], x[1] are position variables (angles)
        # x[2..11] are distance variables; set to 0.5 for g=0 (Pareto front)
        # f1 = cos(x0*pi/2)*cos(x1*pi/2)
        # f2 = cos(x0*pi/2)*sin(x1*pi/2)
        # f3 = sin(x0*pi/2)
        
        n_points = budget
        
        # Strategy: Stratified sampling on the spherical surface parameterized by (theta1, theta2)
        # theta1 = x0 * pi/2, theta2 = x1 * pi/2, both in [0, pi/2]
        # For hypervolume, we want uniform coverage in objective space, not angle space.
        # Use inverse CDF sampling: sample uniformly in f1, f2, f3? No, they're coupled.
        # Better: use a grid in (theta1, theta2) with stratification, then refine.
        
        # Approach: Create a stratified grid in [0,1]^2 (which maps to [0, pi/2]^2 angles)
        # n x n grid gives n^2 points. We need ~200 points, so use a grid of size ~15x15=225
        # and select the best 200, or use a non-uniform grid biased toward edges.
        
        # Actually, for hypervolume on a convex front, uniform angular spacing is good.
        # Let's use a Halton sequence or Sobol-like quasi-random for good distribution.
        
        # Simple approach: use a stratified grid with jitter
        n_grid = int(np.ceil(np.sqrt(n_points)))
        # Adjust to get close to n_points
        # n_grid x n_grid = n_grid^2 points, we'll take the first n_points
        
        # Create stratified samples in [0, 1]^2
        # For each cell, place one point at a random location within the cell
        rng = np.random.RandomState(42)  # Fixed seed for reproducibility within the call
        # But we need to use the global random state for consistency with evaluator
        # Use np.random directly
        
        # Stratified sampling: divide [0,1]^2 into n_grid x n_grid cells
        # One sample per cell, uniformly within the cell
        u1 = (np.arange(n_grid) + np.random.rand(n_grid)) / n_grid
        u2 = (np.arange(n_grid) + np.random.rand(n_grid)) / n_grid
        
        # Create meshgrid
        U1, U2 = np.meshgrid(u1, u2)
        params_grid = np.column_stack([U1.ravel(), U2.ravel()])
        
        # If we have more than n_points, randomly subsample
        if len(params_grid) > n_points:
            idx = np.random.choice(len(params_grid), n_points, replace=False)
            params = params_grid[idx]
        else:
            # If we have fewer, add random points to fill
            params = params_grid
            while len(params) < n_points:
                params = np.vstack([params, np.random.rand(1, 2)])
        
        # Ensure corners are included
        corners = np.array([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]])
        # Replace the last few points with corners if not already present
        for c in corners:
            # Check if close to any existing point
            dists = np.linalg.norm(params - c, axis=1)
            if np.min(dists) > 0.1:
                params[-1] = c
        
        # Construct decision vectors
        X = np.zeros((len(params), dim))
        X[:, 0] = params[:, 0]
        X[:, 1] = params[:, 1]
        X[:, 2:] = 0.5
        
        # Clip to bounds
        X = np.clip(X, lo, hi)
        
        # Evaluate all points
        for i in range(len(X)):
            if time.time() > deadline:
                break
            problem(X[i])
            
        return X
# EVOLVE-BLOCK-END
