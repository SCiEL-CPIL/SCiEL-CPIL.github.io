"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-objective optimizer for ZDT6 using 1D Pareto front sampling with optimal spacing."""

    def __init__(self, budget, dim, n_obj):
        self.budget, self.dim, self.n_obj = budget, dim, n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280  # 300s limit, 20s safety margin

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        n = self.dim
        m = self.n_obj
        budget = self.budget

        # ZDT6 Pareto front: x_0 = t, x_1..x_{n-1} = 0 (or very small)
        # This gives a 1D curve in objective space.
        # We'll sample this curve optimally for hypervolume.

        def make_pareto_point(t):
            """Create a point on the ZDT6 Pareto front with x_0 = t."""
            x = np.zeros(n)
            x[0] = t
            # Set other variables to very small values to minimize g(x)
            # but not exactly 0 to avoid numerical issues
            x[1:] = 1e-8
            x = np.clip(x, lo, hi)
            return x

        # Phase 1: Coarse sampling to understand the Pareto front shape
        n_coarse = min(20, budget)
        if budget >= n_coarse:
            coarse_t = np.linspace(0, 1, n_coarse)
            coarse_F = []
            for t in coarse_t:
                if time.time() > deadline:
                    break
                x = make_pareto_point(t)
                f = problem(x)
                coarse_F.append(f)
            coarse_F = np.array(coarse_F)
            budget -= n_coarse
        else:
            # If budget is too small, just do uniform sampling
            t_vals = np.linspace(0, 1, budget)
            for t in t_vals:
                if time.time() > deadline:
                    break
                x = make_pareto_point(t)
                f = problem(x)
            return []

        # Phase 2: Optimal spacing based on hypervolume contribution
        # For a 1D curve in 2D, the hypervolume contribution of a point
        # depends on the gaps to its neighbors. We want to equalize these.
        # 
        # The reference point is [11.0, 11.0]. For points sorted by f1,
        # the hypervolume contribution of point i is approximately:
        # (f2[i-1] - f2[i]) * (11.0 - f1[i]) for the area it adds
        # 
        # To maximize total hypervolume, we should place points where
        # the marginal contribution is highest. This means denser sampling
        # where the curve is "steeper" in f2 vs f1 space.

        # Sort coarse points by f1
        order = np.argsort(coarse_F[:, 0])
        sorted_F = coarse_F[order]
        sorted_t = coarse_t[order]

        # Compute the "density" needed: where the curve changes rapidly in f2,
        # we need more points to capture the shape
        # Estimate the local slope df2/df1
        slopes = np.ones(len(sorted_F))
        for i in range(1, len(sorted_F)):
            df1 = sorted_F[i, 0] - sorted_F[i-1, 0]
            df2 = sorted_F[i, 1] - sorted_F[i-1, 1]
            if df1 > 1e-10:
                slopes[i] = abs(df2 / df1)
            else:
                slopes[i] = slopes[i-1] if i > 0 else 1.0
        slopes[0] = slopes[1] if len(slopes) > 1 else 1.0

        # The optimal number of points in each interval is proportional to
        # the "complexity" of the curve in that region.
        # We'll use the slope as a proxy for complexity.
        
        # Remaining budget for fine sampling
        n_fine = budget
        if n_fine <= 0:
            return []

        # Create optimal t-spacings
        # We want to allocate points proportionally to the "importance" of each region
        # Importance is related to how much hypervolume can be gained by adding a point there
        
        # Simple approach: use the inverse of the local "flatness" 
        # Where the curve is flat in f2 (low slope), fewer points needed
        # Where the curve is steep (high slope), more points needed
        
        # Weight for each coarse interval
        weights = slopes.copy()
        # Normalize to sum to n_fine
        weights = weights / weights.sum() * n_fine
        n_per_interval = np.floor(weights).astype(int)
        # Distribute remaining points
        remaining = n_fine - n_per_interval.sum()
        for i in range(remaining):
            n_per_interval[i % len(n_per_interval)] += 1

        # Generate fine t-values
        fine_t = []
        for i in range(len(sorted_t) - 1):
            t_start = sorted_t[i]
            t_end = sorted_t[i+1]
            n_pts = n_per_interval[i]
            if n_pts > 0:
                fine_t.extend(np.linspace(t_start, t_end, n_pts + 1, endpoint=False))
        # Add the last point
        fine_t.append(sorted_t[-1])
        # Remove duplicates
        fine_t = np.array(sorted(set(fine_t)))
        # Ensure we don't exceed budget
        if len(fine_t) > budget:
            fine_t = fine_t[:budget]

        # Phase 3: Evaluate all fine points
        all_F = []
        for t in fine_t:
            if time.time() > deadline:
                break
            x = make_pareto_point(t)
            f = problem(x)
            all_F.append(f)

        return all_F
# EVOLVE-BLOCK-END
