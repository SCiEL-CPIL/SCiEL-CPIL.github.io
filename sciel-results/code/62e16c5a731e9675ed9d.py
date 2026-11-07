"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """Multi-objective optimizer using structured radial sweep for WFG9."""

    def __init__(self, budget, dim, n_obj):
        self.budget, self.dim, self.n_obj = budget, dim, n_obj

    def __call__(self, problem, bounds):
        t_start = time.time()
        deadline = t_start + 290  # safety margin before 300s limit
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        range_ = hi - lo
        dim = self.dim
        budget = self.budget
        
        # WFG9 is a 12-dimensional problem, but the Pareto front is primarily
        # determined by the first few variables. The remaining variables often
        # contribute to noise or are less critical for the front's shape.
        # We focus our budget on a structured sweep of the first 3-4 dimensions
        # and fix the rest to a mid-range value to reduce dimensionality.
        
        # Strategy:
        # 1. Fix dimensions 4..11 to the midpoint of their bounds.
        # 2. Perform a dense grid/radial sweep on dimensions 0, 1, 2.
        # 3. Use a small portion of the budget for random exploration in full space
        #    to catch any unexpected non-linearities, but prioritize the manifold.
        
        # Midpoint for fixed dimensions
        mid = (lo + hi) / 2.0
        
        # Determine how many points to allocate to the structured sweep
        # Let's use 90% for the structured sweep and 10% for random exploration
        n_sweep = int(budget * 0.9)
        n_random = budget - n_sweep
        
        # Generate structured points
        # We will sweep x[0], x[1], x[2]
        # A good heuristic for WFG9 is that the front is often curved.
        # We create a 3D grid in the normalized [0,1] space for the first 3 dims.
        # To get ~n_sweep points in a 3D grid, we need roughly cube root of n_sweep per axis.
        # Let's be smarter: use a spherical shell or a dense line sweep if we suspect a 2D front.
        # Actually, WFG9 is a 3-objective problem. The Pareto front is a 2D surface.
        # The decision variables controlling this are usually x[0] and x[1] (or similar).
        # Let's try a 2D grid on x[0] and x[1], and fix x[2] to a few specific values,
        # or better, just do a dense 2D grid on x[0] and x[1] and let x[2] vary slightly or be fixed.
        
        # Let's assume the main trade-off is between x[0] and x[1].
        # We will create a grid of size sqrt(n_sweep) x sqrt(n_sweep).
        n_per_axis = int(np.sqrt(n_sweep))
        if n_per_axis < 2:
            n_per_axis = 2
            
        # Generate grid points in [0,1]
        u = np.linspace(0, 1, n_per_axis)
        v = np.linspace(0, 1, n_per_axis)
        U, V = np.meshgrid(u, v)
        
        # Flatten
        U_flat = U.flatten()
        V_flat = V.flatten()
        
        # If we have more budget, we can add a third dimension sweep or refine.
        # For now, let's map these to x[0] and x[1].
        # We need to handle the case where n_per_axis^2 might be less than n_sweep.
        # Let's fill the rest with a finer grid or random points on the manifold.
        
        X_sweep = np.zeros((n_sweep, dim))
        # Fill with midpoints first
        X_sweep[:, :] = mid
        
        # Assign the grid points to the first two dimensions
        # We might have more grid points than needed if n_per_axis^2 > n_sweep, 
        # but we defined n_per_axis = sqrt(n_sweep), so it's close.
        # Let's just take the first n_sweep points from a larger potential grid if needed,
        # or generate exactly n_sweep points.
        
        # Better approach: Generate exactly n_sweep points using a space-filling curve or simple grid.
        # Let's use a simple grid with variable resolution.
        # To ensure coverage, let's use a grid of size k x k where k = int(np.ceil(np.sqrt(n_sweep)))
        k = int(np.ceil(np.sqrt(n_sweep)))
        u = np.linspace(0, 1, k)
        v = np.linspace(0, 1, k)
        U, V = np.meshgrid(u, v)
        U_flat = U.flatten()
        V_flat = V.flatten()
        
        # If k*k > n_sweep, we take the first n_sweep.
        # If k*k < n_sweep, we duplicate or add randoms.
        n_grid = min(n_sweep, k*k)
        
        X_sweep[:n_grid, 0] = lo[0] + range_[0] * U_flat[:n_grid]
        X_sweep[:n_grid, 1] = lo[1] + range_[1] * V_flat[:n_grid]
        
        # For the remaining points in the sweep budget (if any), let's vary x[2]
        if n_grid < n_sweep:
            # Create a line sweep on x[2] for the first few grid points
            n_extra = n_sweep - n_grid
            # Pick a subset of the grid points and vary x[2]
            # Or just generate random points in the 3D box [0,1]^3 for x[0],x[1],x[2]
            # but constrained to the "front" region if possible.
            # Simplest: random in the 3D box for the first 3 dims.
            rng = np.random.default_rng(42)
            extra_points = rng.random((n_extra, 3))
            X_sweep[n_grid:n_sweep, 0] = lo[0] + range_[0] * extra_points[:, 0]
            X_sweep[n_grid:n_sweep, 1] = lo[1] + range_[1] * extra_points[:, 1]
            X_sweep[n_grid:n_sweep, 2] = lo[2] + range_[2] * extra_points[:, 2]
            
        # Generate random exploration points (full space)
        if n_random > 0:
            rng_rand = np.random.default_rng(123)
            X_random = lo + range_ * rng_rand.random((n_random, dim))
        else:
            X_random = np.empty((0, dim))
            
        # Combine
        X = np.vstack([X_sweep, X_random]) if n_random > 0 else X_sweep
        
        # Ensure we don't exceed budget
        X = X[:budget]
        
        # Evaluate all points
        F = np.array([problem(x) for x in X])
        
        return X
# EVOLVE-BLOCK-END
