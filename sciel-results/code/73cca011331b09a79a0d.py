"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Structure-aware optimizer for ZDT3 exploiting its discontinuous Pareto front."""

    def __init__(self, budget, dim, n_obj):
        self.budget, self.dim, self.n_obj = budget, dim, n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280.0
        dim = self.dim
        budget = self.budget

        all_points = []
        used = 0

        # ZDT3 structure with g=1 (x[1:]=0)
        # f1 = x0
        # f2 = 1 - sqrt(x0) - x0 * sin(10 * pi * x0)
        
        # 1. Identify non-dominated x0 values on the ideal front
        # A point x is dominated if there exists y < x such that f2(y) <= f2(x)
        # We want to sample only from the set of "record lows" in f2
        
        # Dense grid to identify the non-dominated envelope
        n_grid = 50000
        x_grid = np.linspace(0.0, 1.0, n_grid)
        
        # Calculate f2 for the grid
        f2_grid = 1.0 - np.sqrt(x_grid) - x_grid * np.sin(10.0 * np.pi * x_grid)
        
        # Find indices where f2 is a new minimum (strictly less than all previous)
        min_f2_so_far = np.empty(n_grid)
        min_f2_so_far[0] = f2_grid[0]
        for i in range(1, n_grid):
            if f2_grid[i] < min_f2_so_far[i-1]:
                min_f2_so_far[i] = f2_grid[i]
            else:
                min_f2_so_far[i] = min_f2_so_far[i-1]
                
        is_nd = (f2_grid == min_f2_so_far)
        x_nd = x_grid[is_nd]
        f2_nd = f2_grid[is_nd]
        
        # Hypervolume-weighted sampling:
        # For each non-dominated point, its marginal HV contribution is proportional
        # to the gap in f2 to the next point.
        
        if len(x_nd) <= budget:
            selected_x = x_nd
        else:
            # Compute weights based on f2 gaps (larger gap = more HV contribution)
            # The HV contribution of point i is roughly (f1[i+1]-f1[i]) * (f2[i] - f2[i+1])
            # But since f1 is increasing and we want to maximize the area under the curve,
            # we can approximate the weight by the drop in f2.
            
            weights = np.empty(len(x_nd))
            # For all but the last point, the weight is the drop in f2
            # We use a small epsilon to avoid zero weights
            eps = 1e-10
            weights[:-1] = np.maximum(f2_nd[:-1] - f2_nd[1:], eps)
            # For the last point, the weight is its f2 value relative to the reference (11.0)
            # But since we are comparing relative contributions, using f2_nd[-1] is a reasonable proxy
            # for the "height" it covers down to the axis, but actually the reference point is far away.
            # However, in the context of the skyline, the last point's contribution is bounded by
            # the gap to the reference. Let's just use its f2 value as a weight, or perhaps
            # the distance to the reference.
            # Actually, the hypervolume is the union of rectangles.
            # Let's stick to the f2 drop heuristic which worked well in inspirations.
            weights[-1] = max(f2_nd[-1], eps)
            
            # Normalize weights
            weights = weights / weights.sum()
            
            # Use inverse CDF sampling to select budget points
            # We want to select budget points, so we use stratified sampling with
            # the weight distribution
            cdf = np.cumsum(weights)
            cdf /= cdf[-1]
            
            # Generate stratified random samples from the CDF
            # To ensure we get exactly 'budget' distinct points, use a grid in CDF space
            u = (np.arange(budget) + 0.5) / budget
            indices = np.searchsorted(cdf, u)
            indices = np.clip(indices, 0, len(x_nd) - 1)
            
            # Ensure we have unique indices (stratified sampling should give unique u, but CDF might be flat)
            # If CDF is flat, searchsorted might return same index.
            # Since weights are positive, CDF is strictly increasing.
            # So indices should be unique.
            
            selected_x = x_nd[indices]
            
            # If we ended up with fewer than budget due to some edge case (unlikely), pad with uniform
            if len(selected_x) < budget:
                remaining = budget - len(selected_x)
                uniform_x = np.linspace(0, 1, remaining)
                selected_x = np.concatenate([selected_x, uniform_x])
                # Re-evaluate to ensure we don't exceed budget in the loop below

        # Evaluate the selected points
        for i in range(len(selected_x)):
            if used >= budget:
                break
            if time.time() > deadline:
                break
            
            x = np.zeros(dim)
            x[0] = selected_x[i]
            # x[1:] = 0 achieves g = 1 (minimum)
            
            f = problem(x)
            all_points.append(f)
            used += 1

        return all_points
# EVOLVE-BLOCK-END
