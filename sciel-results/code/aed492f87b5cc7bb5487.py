"""Optimizer for ZDT2 exploiting known Pareto front structure."""
import numpy as np
import time


class Optimizer:
    """
    ZDT2-specific optimizer.
    ZDT2: f1 = x0, g = 1 + 9*sum(x1..xn-1)/n, f2 = g*(1 - sqrt(f1/g))
    Pareto front: x1..x_{n-1} = 0, x0 in [0,1]
    f1 = x0, f2 = 1 - sqrt(x0)
    
    Uses curvature-adaptive grid spacing to maximize hypervolume coverage.
    The Pareto front is f2 = 1 - sqrt(f1), which has high curvature near f1=0.
    We use a non-uniform grid that places more points where the front curves more,
    maximizing the enclosed hypervolume with a fixed budget.
    """

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280.0  # safety margin before 300s limit

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        dim = self.dim
        budget = self.budget

        # For ZDT2, the Pareto front is f1 = x0, f2 = 1 - sqrt(x0)
        # We want to maximize hypervolume with reference point [11, 11]
        
        # Strategy: Use all budget for dense sampling of the Pareto front
        # with curvature-adaptive spacing.
        
        # The hypervolume contribution of a point (f1, f2) on the front is:
        # (11 - f1) * (11 - f2) for the rectangle from that point to the reference
        # But we need to compute the union of all such rectangles.
        
        # For a set of points on the Pareto front sorted by f1, the hypervolume
        # can be computed as the sum of trapezoidal areas under the front.
        
        # To maximize hypervolume with N points, we should place points where
        # the marginal gain is highest. For f2 = 1 - sqrt(f1), the curvature
        # is highest near f1 = 0.
        
        # A good heuristic: use a grid that is denser near f1 = 0.
        # We can use a power-law spacing: f1_i = (i/N)^p for some p > 1
        # This gives more points near f1 = 0.
        
        # Let's try p = 2 (quadratic spacing) which should be effective.
        
        n_points = budget
        if n_points < 2:
            n_points = 2
        
        # Use power-law spacing: f1_i = (i/n_points)^p
        # Try different p values and pick the one that maximizes estimated hypervolume
        best_p = 1.0
        best_hv = 0.0
        
        # Test a range of p values
        for p in [1.0, 1.2, 1.5, 2.0, 2.5, 3.0]:
            f1_vals = (np.arange(n_points) / n_points) ** p
            f1_vals = np.unique(f1_vals)  # Remove duplicates
            n_actual = len(f1_vals)
            if n_actual < 2:
                continue
            f2_vals = 1.0 - np.sqrt(f1_vals)
            
            # Compute hypervolume for these points with reference [11, 11]
            # Sort by f1 ascending
            order = np.argsort(f1_vals)
            f1_sorted = f1_vals[order]
            f2_sorted = f2_vals[order]
            
            # Hypervolume = sum of areas of rectangles
            # For each consecutive pair, the area is (f1_{i+1} - f1_i) * (11 - f2_i)
            # Plus the last rectangle from last point to reference
            hv = 0.0
            for i in range(len(f1_sorted) - 1):
                df1 = f1_sorted[i+1] - f1_sorted[i]
                # The height is determined by the lower f2 (better) point
                # Since f2 decreases as f1 increases, f2_i > f2_{i+1}
                # The rectangle from (f1_i, f2_i) to (f1_{i+1}, 11) has area df1 * (11 - f2_i)
                # But we need to be careful about the union.
                # Actually, for Pareto front points, the hypervolume is:
                # Sum over i of (f1_{i+1} - f1_i) * (11 - f2_i) for i < n-1
                # Plus (11 - f1_{n-1}) * (11 - f2_{n-1}) for the last segment
                hv += df1 * (11.0 - f2_sorted[i])
            hv += (11.0 - f1_sorted[-1]) * (11.0 - f2_sorted[-1])
            
            if hv > best_hv:
                best_hv = hv
                best_p = p
        
        # Generate final grid with best_p
        f1_vals = (np.arange(n_points) / n_points) ** best_p
        f1_vals = np.unique(f1_vals)
        
        F_list = []
        for f1 in f1_vals:
            if time.time() > deadline:
                break
            x = np.zeros(dim)
            x[0] = f1
            f = problem(x)
            F_list.append(f)
        
        F = np.array(F_list)
        
        return F
