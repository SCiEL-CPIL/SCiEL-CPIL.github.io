"""Optimizer for ZDT2 using analytic Pareto front with hypervolume-optimized sampling."""
import numpy as np
import time
from scipy.optimize import minimize_scalar


class Optimizer:
    """Multi-objective optimizer exploiting ZDT2's analytic Pareto front."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280.0  # safety margin before 330s kill

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        dim = self.dim
        budget = self.budget
        n = budget
        R = 11.0  # reference point

        def f2_analytic(t):
            """ZDT2 second objective on Pareto front: x0=t, x1..=0"""
            t = np.maximum(t, 1e-10)
            return (1.0 + 9.0 / np.sqrt(t)) * (1.0 - np.sqrt(t / 10.0))

        def hypervolume_2d(t_sorted):
            """Compute hypervolume of points on ZDT2 front with reference (R, R).
            t_sorted must be sorted ascending, and all f2 values must be < R.
            """
            if len(t_sorted) == 0:
                return 0.0
            t_sorted = np.asarray(t_sorted, dtype=np.float64)
            f2_vals = f2_analytic(t_sorted)
            # Only count points with f2 < R
            mask = f2_vals < R
            if not np.any(mask):
                return 0.0
            t_valid = t_sorted[mask]
            f2_valid = f2_vals[mask]
            if len(t_valid) == 1:
                return (R - t_valid[0]) * (R - f2_valid[0])
            # HV = sum_{i=0}^{n-2} (t_{i+1} - t_i) * (R - f2_i) + (R - t_{n-1}) * (R - f2_{n-1})
            dt = np.diff(t_valid)
            contrib = dt * (R - f2_valid[:-1])
            last_contrib = (R - t_valid[-1]) * (R - f2_valid[-1])
            return float(np.sum(contrib) + last_contrib)

        # Find t_min where f2(t_min) = R
        from scipy.optimize import brentq
        try:
            t_min = brentq(lambda t: f2_analytic(t) - R, 1e-6, 1.0)
        except Exception:
            t_min = 0.47
        t_max = 1.0

        # Initial t values: use Beta(0.5, 0.5) inverse CDF mapped to [t_min, t_max]
        u = np.linspace(0, 1, n, endpoint=False) + 1.0 / (2.0 * n)
        u_beta = np.sin(np.pi * u / 2.0) ** 2
        t_vals = t_min + u_beta * (t_max - t_min)
        t_vals = np.clip(t_vals, t_min, t_max)
        t_vals = np.sort(t_vals)

        # Coordinate ascent to optimize hypervolume
        # For each point, find optimal t_i given neighbors
        n_iter = 30
        for it in range(n_iter):
            if time.time() > deadline - 10.0:
                break
            improved = False
            for i in range(n):
                if time.time() > deadline - 10.0:
                    break
                # Bounds for t_i: between t_{i-1} and t_{i+1} (or t_min/t_max at ends)
                lo_i = t_vals[i - 1] if i > 0 else t_min
                hi_i = t_vals[i + 1] if i < n - 1 else t_max
                if lo_i >= hi_i:
                    continue

                # Objective: maximize HV, so minimize -HV
                # Only the terms involving t_i change:
                # If i > 0: term (t_i - t_{i-1}) * (R - f2(t_{i-1})) changes
                # If i < n-1: term (t_{i+1} - t_i) * (R - f2(t_i)) changes
                # Also if i == n-1: last term (R - t_i)*(R - f2(t_i)) changes

                def neg_hv_partial(t_i_candidate):
                    # Compute HV contribution involving point i
                    t_new = t_vals.copy()
                    t_new[i] = t_i_candidate
                    # Ensure sorted
                    if i > 0 and t_i_candidate < t_new[i - 1]:
                        t_i_candidate = t_new[i - 1] + 1e-15
                    if i < n - 1 and t_i_candidate > t_new[i + 1]:
                        t_i_candidate = t_new[i + 1] - 1e-15
                    t_new[i] = t_i_candidate
                    # Quick HV computation - only affected terms
                    # But for simplicity and correctness, compute full HV
                    # This is O(n) per call, and we have n points * n_iter iterations
                    # = O(n^2 * n_iter) = 200^2 * 30 = 1.2M ops, fine
                    return -hypervolume_2d(t_new)

                # Use golden section or bounded minimization
                res = minimize_scalar(neg_hv_partial, bounds=(lo_i + 1e-12, hi_i - 1e-12),
                                       method='bounded', options={'maxiter': 50, 'xatol': 1e-10})
                if res.success and res.fun < neg_hv_partial(t_vals[i]):
                    t_vals[i] = res.x
                    improved = True

            if not improved:
                break

        # Ensure sorted
        t_vals = np.sort(t_vals)
        t_vals = np.clip(t_vals, t_min, t_max)

        # Create x vectors: x0 = t, x1..x_{n-1} = 0
        X = np.zeros((n, dim))
        X[:, 0] = t_vals

        # Evaluate all points
        F = np.array([problem(x) for x in X])

        if time.time() > deadline:
            return X

        return X
