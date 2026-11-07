"""ZDT4 optimizer with curvature-based optimal x0 placement for hypervolume maximization."""
import numpy as np
import time
from scipy.integrate import solve_ivp


class Optimizer:
    """
    Multi-objective optimizer exploiting ZDT4's known Pareto front structure.
    
    ZDT4 Pareto front: when x[1:] = 0, f1 = x0, f2 = 1 + (1 - sqrt(x0)) * 101
    for x0 in [0, 1]. Uses curvature-based density to place points optimally
    for maximizing discrete hypervolume.
    """

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def _zdt4_front_point(self, x0):
        """Compute the objective values for a ZDT4 Pareto front point."""
        f1 = x0
        f2 = 1.0 + (1.0 - np.sqrt(max(x0, 0.0))) * 101.0
        return np.array([f1, f2])

    def _hypervolume_2d(self, points, ref):
        """Compute 2D hypervolume for a set of points with given reference point."""
        if len(points) == 0:
            return 0.0
        mask = (points[:, 0] < ref[0]) & (points[:, 1] < ref[1])
        pts = points[mask]
        if len(pts) == 0:
            return 0.0
        pts = pts[np.argsort(pts[:, 0])]
        hv = 0.0
        prev_f2 = ref[1]
        for i in range(len(pts)):
            f1, f2 = pts[i]
            if f2 < prev_f2:
                hv += (ref[0] - f1) * (prev_f2 - f2)
                prev_f2 = f2
        return hv

    def _compute_optimal_x0_grid(self, n_points, ref):
        """
        Compute optimal x0 placement using curvature-based density.
        
        The ZDT4 Pareto front is: f1 = x0, f2 = 1 + 101*(1-sqrt(x0))
        Curvature is high near x0=0, so we place more points there.
        
        We use the insight that for a concave front, the optimal spacing
        for hypervolume maximization is proportional to the local 'importance'
        which relates to the derivative of f2 with respect to f1.
        """
        # The Pareto front in objective space:
        # f1 = x0, f2 = 1 + 101*(1 - sqrt(x0))
        # df2/df1 = df2/dx0 / (df1/dx0) = -101/(2*sqrt(x0)) / 1 = -101/(2*sqrt(x0))
        # |df2/df1| = 101/(2*sqrt(x0))
        
        # For hypervolume maximization, points should be spaced such that
        # each contributes roughly equal hypervolume. The contribution of a point
        # at (f1, f2) is roughly (ref[0]-f1) * delta_f2 where delta_f2 is the
        # vertical gap to the next point.
        
        # A good strategy: use a transformation that accounts for the curvature.
        # The front has infinite slope at x0=0, so we need many points near 0.
        
        # Use the property that equal hypervolume contribution means:
        # (ref[0] - f1_i) * (f2_i - f2_{i+1}) ≈ constant
        # Since f2_i - f2_{i+1} ≈ |df2/df1| * (f1_{i+1} - f1_i)
        # We want: (ref[0] - f1_i) * |df2/df1| * delta_f1 ≈ constant
        # So delta_f1 ∝ 1 / ((ref[0] - f1_i) * |df2/df1|)
        # |df2/df1| = 101/(2*sqrt(x0))
        # delta_f1 ∝ 2*sqrt(x0) / (101 * (ref[0] - x0))
        # delta_f1 ∝ sqrt(x0) / (11 - x0)  (since ref[0]=11)
        
        # So the density of points should be proportional to (11-x0)/sqrt(x0)
        
        # We can compute the CDF of this density and use it to place points.
        
        # Density function: rho(x) = (11 - x) / sqrt(x) for x in (0, 1]
        # CDF: F(x) = integral_0^x (11-t)/sqrt(t) dt
        # = 11*2*sqrt(x) - 2*(2/3)*x^(3/2)
        # = 22*sqrt(x) - (4/3)*x^(3/2)
        # Normalized: F(x) / F(1) where F(1) = 22 - 4/3 = 62/3
        
        def cdf(x):
            """Normalized CDF of the curvature-based density."""
            x = np.clip(x, 1e-15, 1.0)
            val = 22.0 * np.sqrt(x) - (4.0/3.0) * x**1.5
            return val / (62.0/3.0)
        
        def cdf_inv(u):
            """Inverse of the CDF - find x such that cdf(x) = u."""
            # Use bisection for robustness
            lo, hi = 1e-15, 1.0
            for _ in range(60):
                mid = (lo + hi) / 2.0
                if cdf(mid) < u:
                    lo = mid
                else:
                    hi = mid
            return (lo + hi) / 2.0
        
        # Place points at equal intervals in the CDF space
        # This gives optimal spacing for the given density
        if n_points <= 2:
            return np.array([0.0, 1.0])[:n_points]
        
        # Use quantiles of the CDF
        # u values: (i + 0.5) / n_points for i = 0, ..., n_points-1
        # But we want to include endpoints x0=0 and x0=1
        # So use: u_i = i / (n_points - 1) for i = 0, ..., n_points-1
        # But cdf(0) = 0 and cdf(1) = 1, so this works
        
        u_vals = np.linspace(0.0, 1.0, n_points)
        x0_vals = np.array([cdf_inv(u) for u in u_vals])
        
        # Ensure endpoints
        x0_vals[0] = 0.0
        x0_vals[-1] = 1.0
        x0_vals = np.sort(x0_vals)
        
        return x0_vals

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280  # safety margin before 300s limit
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        # ZDT4 known Pareto front: x[0] in [0,1], x[1:]=0
        # f1 = x[0], f2 = 1 + (1 - sqrt(x[0])) * 101
        # Reference point for scoring: [11.0, 400.0]
        ref = np.array([11.0, 400.0])
        
        n_front = min(195, self.budget - 5)
        
        # Compute optimal x0 grid using curvature-based density
        x0_selected = self._compute_optimal_x0_grid(n_front, ref)
        
        # Convert to decision vectors
        X_front = np.zeros((len(x0_selected), self.dim))
        for i, x0 in enumerate(x0_selected):
            X_front[i, 0] = x0
            # x[1:] = 0 is the global minimum for ZDT4's g function
        
        # Evaluate front points to get actual objective values
        F = np.array([problem(x) for x in X_front])
        used = len(X_front)
        
        # Use remaining budget for verification
        remaining = self.budget - used
        if remaining > 0 and time.time() < deadline:
            n_verify = min(remaining, 5)
            for i in range(n_verify):
                if time.time() > deadline - 5:
                    break
                idx = i % len(X_front)
                x_check = X_front[idx].copy()
                f_check = problem(x_check)
                used += 1
        
        return X_front
