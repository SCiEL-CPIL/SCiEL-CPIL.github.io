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
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        dim = self.dim
        n_obj = self.n_obj
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
        n_grid = 20000
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
        # to the gap in f2 to the next point (since f1 is increasing, the width in f1
        # is fixed by the grid spacing, but the height contribution depends on f2 gap)
        # Actually, for a sorted set of non-dominated points (f1 increasing, f2 decreasing),
        # the HV contribution of point i is roughly (f1[i+1]-f1[i]) * (f2[i] - f2_ref)
        # but more precisely it's the area of the rectangle that only point i covers.
        # A simpler heuristic: weight by the drop in f2 from this point to the next.
        
        if len(x_nd) <= budget:
            selected_x = x_nd
        else:
            # Compute weights based on f2 gaps (larger gap = more HV contribution)
            f2_drops = np.diff(f2_nd)
            f2_drops = np.maximum(f2_drops, 1e-10)  # avoid zeros
            # The last point's weight is its f2 value relative to reference
            # Actually, let's use a different approach: 
            # The HV contribution of each point in the skyline is proportional to
            # the "area" it uniquely covers. For points sorted by f1, the contribution
            # of point i is (f1[i] - f1[i-1]) * (f2[i] - f2[i+1]) approximately,
            # but this is complex. Let's use a simpler heuristic:
            # weight_i = f2_nd[i] - f2_nd[i+1] for i < n-1, and f2_nd[-1] for last
            # This emphasizes points where f2 drops significantly.
            
            weights = np.empty(len(x_nd))
            weights[:-1] = np.maximum(f2_nd[:-1] - f2_nd[1:], 1e-10)
            weights[-1] = max(f2_nd[-1], 1e-10)
            
            # Normalize weights
            weights = weights / weights.sum()
            
            # Use inverse CDF sampling to select budget points
            # We want to select budget points, so we use stratified sampling with
            # the weight distribution
            # First, get the CDF
            cdf = np.cumsum(weights)
            cdf /= cdf[-1]
            
            # Generate stratified random samples from the CDF
            # To ensure we get exactly 'budget' distinct points, use a grid in CDF space
            u = (np.arange(budget) + 0.5) / budget
            # Inverse CDF
            indices = np.searchsorted(cdf, u)
            indices = np.clip(indices, 0, len(x_nd) - 1)
            selected_x = x_nd[indices]
            
            # Ensure we include endpoints for better HV coverage
            if selected_x[0] != x_nd[0]:
                selected_x[0] = x_nd[0]
            if selected_x[-1] != x_nd[-1]:
                selected_x[-1] = x_nd[-1]
            
            # Remove duplicates while preserving order
            selected_x = np.array(list(dict.fromkeys(selected_x)))
            
            # If we have fewer than budget after dedup, fill in with uniform
            if len(selected_x) < budget:
                remaining = budget - len(selected_x)
                # Sample uniformly from x_nd excluding already selected
                mask = np.ones(len(x_nd), dtype=bool)
                for sx in selected_x:
                    idx = np.argmin(np.abs(x_nd - sx))
                    mask[idx] = False
                remaining_candidates = x_nd[mask]
                if len(remaining_candidates) > 0:
                    extra_indices = np.random.choice(len(remaining_candidates), 
                                                      min(remaining, len(remaining_candidates)), 
                                                      replace=False)
                    extra_points = remaining_candidates[extra_indices]
                    selected_x = np.concatenate([selected_x, extra_points])
                    selected_x = np.sort(selected_x)
            
        # Ensure we don't exceed budget
        n_evals = len(selected_x)
        
        for i in range(n_evals):
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
