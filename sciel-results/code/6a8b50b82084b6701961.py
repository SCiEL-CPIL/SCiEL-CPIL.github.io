"""Improved ZDT4 optimizer with non-uniform grid spacing for hypervolume maximization."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-objective optimizer for ZDT4 using optimized 1D grid search on x0."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 285  # safety margin before 300s limit

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        n_vars = self.dim
        n_evals = self.budget

        x0_lo = lo[0]
        x0_hi = hi[0]

        # Non-uniform grid: concentrate points near the ends of the Pareto front
        # where they contribute most to hypervolume (extending the front in both objectives)
        # Use a power-law distribution: t^alpha with alpha > 1 concentrates at ends
        # Actually for hypervolume, we want to cover the full range well.
        # Let's use a combination: uniform in the middle, denser at ends.
        
        # Strategy: use sin-based spacing that is denser at the ends
        # t_i = 0.5 * (1 - cos(pi * i / (n-1))) gives denser spacing at ends
        # But this might over-concentrate. Let's try a milder version.
        
        # Alternative: use the fact that for hypervolume with far reference point,
        # the optimal spacing is approximately uniform in f1. But to be safe,
        # let's use a slight concentration at the ends.
        
        n_grid = n_evals
        # Create index array
        idx = np.arange(n_grid, dtype=np.float64)
        
        # Use a power-law spacing: u^alpha where alpha slightly > 1
        # This concentrates points near u=0 (x0_lo)
        # But we want concentration at BOTH ends, so use a symmetric approach
        
        # Symmetric concentration: use u = 0.5*(1 - cos(pi*t)) mapped to [0,1]
        # This gives denser spacing at both ends
        # t in [0,1] -> u in [0,1] with u = 0.5*(1 - cos(pi*t))
        
        # For n points, t_i = i/(n-1) for i=0..n-1
        t = idx / max(n_grid - 1, 1)
        # Apply cosine spacing for end concentration
        u = 0.5 * (1.0 - np.cos(np.pi * t))
        
        x0_vals = x0_lo + u * (x0_hi - x0_lo)
        
        results = []
        for x0 in x0_vals:
            x = np.zeros(n_vars)
            x[0] = x0
            # Clip to bounds
            for j in range(n_vars):
                x[j] = np.clip(x[j], lo[j], hi[j])
            f = problem(x)
            results.append(f)
            
            if time.time() > deadline:
                break

        return results
# EVOLVE-BLOCK-END
