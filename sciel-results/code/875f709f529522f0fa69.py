"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


class Optimizer:
    """Multi-start L-BFGS-B + Nelder-Mead refinement with deadline guard."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        start_time = time.time()
        deadline = start_time + 280  # 20s safety margin before 300s limit

        # Wrap func to count calls and clip
        call_count = [0]
        max_calls = budget

        def safe_func(x):
            if call_count[0] >= max_calls:
                return 1e10
            x = np.asarray(x, dtype=float)
            if x.shape != (dim,):
                x = x.reshape(dim)
            x = np.clip(x, -5.0, 5.0)
            call_count[0] += 1
            try:
                val = float(func(x))
            except Exception:
                val = 1e10
            if not np.isfinite(val):
                val = 1e10
            return val

        # Generate diverse starting points using a fixed-seed RNG for reproducibility
        rng = np.random.default_rng(42)

        # Create a set of candidate starting points
        # Mix of: center, corners, random points, and points biased toward origin
        starts = []

        # Center
        starts.append(np.zeros(dim))

        # Corners of the box
        for i in range(2 ** dim):
            corner = np.array([(i >> j) & 1 for j in range(dim)], dtype=float)
            corner = corner * 10.0 - 5.0  # map {0,1} to {-5,5}
            starts.append(corner)

        # Random points
        for _ in range(15):
            starts.append(rng.uniform(-5, 5, dim))

        # Points near the origin (since minimizer is in [-4,4]^5)
        for _ in range(10):
            starts.append(rng.uniform(-4, 4, dim))

        # Sort starts by initial function value (use a few to find good basins)
        # But first, evaluate a subset to find the best starting region
        # We'll do a quick screening with the first 20 starts
        screened = starts[:20]
        screened_vals = [safe_func(x) for x in screened]
        screened_sorted = sorted(zip(screened_vals, screened), key=lambda t: t[0])

        # Take the best 10 starting points for full optimization
        best_starts = [x for _, x in screened_sorted[:10]]

        # Track the best solution found so far
        best_x = best_starts[0]
        best_f = safe_func(best_x)

        # For each starting point, run L-BFGS-B then Nelder-Mead refinement
        for x0 in best_starts:
            # Check time and budget
            if time.time() > deadline or call_count[0] >= max_calls:
                break

            # Phase 1: L-BFGS-B
            try:
                res = minimize(
                    safe_func,
                    x0,
                    method='L-BFGS-B',
                    bounds=[(-5.0, 5.0)] * dim,
                    options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-8}
                )
                if res.success or res.fun < best_f:
                    x1 = np.clip(res.x, -5.0, 5.0)
                    f1 = safe_func(x1)
                    if f1 < best_f:
                        best_x, best_f = x1, f1
            except Exception:
                pass

            # Phase 2: Nelder-Mead refinement from the L-BFGS-B result
            if time.time() > deadline or call_count[0] >= max_calls:
                break

            x_start_nm = np.clip(res.x, -5.0, 5.0) if 'res' in dir() else x0
            try:
                res_nm = minimize(
                    safe_func,
                    x_start_nm,
                    method='Nelder-Mead',
                    options={
                        'maxiter': 500,
                        'xatol': 1e-10,
                        'fatol': 1e-10,
                        'adaptive': True
                    }
                )
                if res_nm.fun < best_f:
                    x2 = np.clip(res_nm.x, -5.0, 5.0)
                    f2 = safe_func(x2)
                    if f2 < best_f:
                        best_x, best_f = x2, f2
            except Exception:
                pass

            # If we still have budget and time, do another NM from the best found
            if call_count[0] < max_calls - 50 and time.time() < deadline:
                try:
                    res_nm2 = minimize(
                        safe_func,
                        best_x,
                        method='Nelder-Mead',
                        options={
                            'maxiter': 300,
                            'xatol': 1e-12,
                            'fatol': 1e-12,
                            'adaptive': True
                        }
                    )
                    if res_nm2.fun < best_f:
                        x3 = np.clip(res_nm2.x, -5.0, 5.0)
                        f3 = safe_func(x3)
                        if f3 < best_f:
                            best_x, best_f = x3, f3
                except Exception:
                    pass

        return best_x, best_f
# EVOLVE-BLOCK-END
