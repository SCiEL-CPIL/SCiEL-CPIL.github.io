"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """Adaptive Grid Subdivision Optimizer for Rastrigin."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        start_time = time.time()
        deadline = start_time + 290.0

        calls_made = 0
        best_x = np.zeros(dim)
        best_f = np.inf

        def counted_func(x):
            nonlocal calls_made, best_x, best_f
            if calls_made >= budget:
                return np.inf
            calls_made += 1
            fx = func(x)
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            return fx

        # Adaptive Grid Search
        # Start with the full domain [-5, 5]^dim
        # Iteratively subdivide the current best cell into 2^dim sub-cells.
        # This creates a deterministic hierarchy of resolution.
        
        # Initial cell: the entire box
        lo = np.full(dim, -5.0)
        hi = np.full(dim, 5.0)
        
        # We will perform levels of subdivision.
        # Level 0: 1 cell (the whole box) -> evaluate center? Or just start subdividing?
        # The prompt suggests: "starts with a coarse 2^5 grid ... identify single best cell, then recursively subdivides only that cell"
        # This implies:
        # 1. Generate 2^dim points in the current cell (subdividing it into 2^dim smaller cells and evaluating their centers or corners? Usually centers for local search).
        # Let's interpret "grid" as evaluating the centers of the 2^dim sub-cells created by bisecting the current best cell.
        
        # However, 2^5 = 32 points per level.
        # Budget is 500.
        # 500 / 32 ~ 15 levels.
        # Resolution after 15 levels: 10 / 2^15 ~ 0.0003. This is very fine.
        
        # Strategy:
        # 1. Evaluate the center of the current best cell? No, that's just one point.
        # The prompt says: "coarse 2^5 grid over [-5, 5]^5 to identify the single best cell"
        # This implies we divide [-5,5]^5 into 2^5 cells (by splitting each dimension in 2) and evaluate the center of each.
        # Then we take the best cell and repeat.
        
        # Let's implement this loop.
        
        while calls_made < budget and time.time() < deadline:
            # Generate 2^dim points: centers of the sub-cells formed by bisecting [lo, hi]
            # For each dimension, the split point is mid = (lo[i] + hi[i]) / 2
            # The sub-cells are defined by choosing either [lo[i], mid] or [mid, hi[i]] for each dimension.
            # The center of such a sub-cell is the average of its bounds.
            
            # To generate all 2^dim centers efficiently:
            # We can use bitwise operations or itertools.product.
            # dim is small (5), so 32 points is fast.
            
            mids = (lo + hi) / 2.0
            
            # Create all combinations of lower/upper choices
            # For each dimension d, the center coordinate is either (lo[d] + mids[d])/2 or (mids[d] + hi[d])/2
            # Let's just generate the grid points explicitly.
            
            points = []
            # 2^dim combinations
            # Using a simple loop or vectorized approach
            # Since dim=5, we can just loop
            for mask in range(1 << dim):
                x = np.zeros(dim)
                for d in range(dim):
                    if (mask >> d) & 1:
                        # Upper half
                        x[d] = (mids[d] + hi[d]) / 2.0
                    else:
                        # Lower half
                        x[d] = (lo[d] + mids[d]) / 2.0
                points.append(x)
            
            points = np.array(points)
            
            # Evaluate all points in this level
            best_local_f = np.inf
            best_local_idx = -1
            
            for i in range(len(points)):
                if calls_made >= budget or time.time() > deadline:
                    break
                x = points[i]
                fx = counted_func(x)
                if fx < best_local_f:
                    best_local_f = fx
                    best_local_idx = i
            
            if best_local_idx == -1:
                break # No points evaluated? Shouldn't happen unless budget/time hit immediately
            
            # Update the search region to the best cell
            # The best cell corresponds to the mask of best_local_idx
            mask = best_local_idx
            for d in range(dim):
                if (mask >> d) & 1:
                    # Upper half: new lo is mid, hi stays
                    lo[d] = mids[d]
                else:
                    # Lower half: new hi is mid, lo stays
                    hi[d] = mids[d]
            
            # Check for convergence or if we are stuck
            # If the cell size becomes very small, we might just stop or continue if budget allows
            # The loop condition handles budget/time.
            # We can also break if the improvement is negligible, but AOCC rewards early progress, 
            # so continuing to refine is good as long as we are finding lower values.
            # However, if we are stuck in a local minimum, this deterministic approach will just refine that minimum.
            # Rastrigin has many local minima. The initial 2^5 grid might pick the wrong basin.
            # But the prompt specifically asks for this "deterministic, memoryless hierarchy" strategy.
            # It relies on the initial grid being fine enough to catch the global basin or the basin being wide.
            # Given the "hidden seeded shift, rotation", the basin might be anywhere.
            # 2^5 = 32 points in 5D is quite coarse.
            # Maybe we should do a few random restarts? 
            # The prompt says "rethinking the approach... grid-based adaptive search". 
            # It doesn't explicitly forbid randomization, but "deterministic... memoryless" suggests pure grid.
            # Let's stick to the pure grid strategy as requested, but perhaps initialize with a slightly finer grid if possible?
            # No, "starts with a coarse 2^5 grid".
            
            # One potential issue: if the first 32 points all land in bad local minima, we refine the worst one.
            # But Rastrigin's global minimum is usually at 0 (shifted). 
            # If the shift is small, the 0-point might be in the grid.
            # The grid points are centers of sub-boxes.
            # Level 0: 
            # lo=-5, hi=5, mid=0.
            # Points are at +/- 2.5 in each dim.
            # None of these are 0.
            # So the first 32 points are all at corners of the 8 octants (centered at 2.5).
            # If the true minimum is near 0, all these points are far away.
            # The best of these 32 will be the one closest to 0 (in terms of Rastrigin value, which is dominated by distance to 0 for large x).
            # So it will pick the sub-box containing 0.
            # Then it subdivides that box.
            # Next level: 16 points in that box.
            # This effectively performs a binary search towards the global minimum if the landscape is unimodal near the minimum or if the gradient points towards it.
            # Rastrigin is multimodal, but the "quadratic" part dominates far from 0.
            # So this should work well for finding the basin.
            
            # To be safer against local minima traps in the early coarse stages, 
            # we could evaluate the center of the *parent* cell as well? 
            # No, let's keep it simple as per the directive.
            
            # Optimization: If we have time, we can just let it run.
            # 500 calls / 32 points per level = ~15 levels.
            # 2^15 = 32768. 10/32768 ~ 3e-4 resolution.
            # This is plenty for Rastrigin.

        return best_x, best_f
# EVOLVE-BLOCK-END
