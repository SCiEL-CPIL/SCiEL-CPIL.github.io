"""
Affine Model Recovery for BBOB f5 (Linear slope).

Strategy:
1. f5 is affine: f(x) = c + v·x.
2. Recover c and v using d+1 = 6 function calls (origin + d axis probes).
3. Compute the exact box vertex that minimizes the linear function.
4. Verify with a few additional evaluations for robustness.
5. Stall on the best point for the remaining budget.
This finds the global minimum in ~6-10 calls instead of 32 (corner search),
giving much better early AOCC progress.
"""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """
    Optimizer that exploits the affine structure of f5 by recovering
    the gradient and offset with minimal calls, then jumping to the
    optimal box vertex.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280  # Safety margin for 300s limit
        d = self.dim
        calls_made = 0

        def safe_call(x):
            nonlocal calls_made
            if calls_made >= self.budget:
                return np.inf
            if time.time() > deadline:
                return np.inf
            val = func(x)
            calls_made += 1
            return val

        # Phase 1: Recover affine model f(x) = c + v·x
        x0 = np.zeros(d)
        f0 = safe_call(x0)
        if not np.isfinite(f0):
            # Fallback: just return origin
            return x0, f0

        # Probe along each axis with step h=1.0
        h = 1.0
        v = np.zeros(d)
        for i in range(d):
            xi = np.zeros(d)
            xi[i] = h
            fi = safe_call(xi)
            if not np.isfinite(fi):
                break
            v[i] = (fi - f0) / h

        # Phase 2: Compute the optimal box vertex
        best_x = np.zeros(d)
        for i in range(d):
            if v[i] > 0:
                best_x[i] = -5.0
            elif v[i] < 0:
                best_x[i] = 5.0
            else:
                best_x[i] = 0.0

        f_best = safe_call(best_x)
        if not np.isfinite(f_best):
            return best_x, f_best

        # Phase 3: Robustness checks — evaluate a few other key points
        # Check the opposite vertex and the origin
        check_points = [
            -best_x.copy(),  # opposite vertex
            np.zeros(d),     # origin
        ]
        # Also check vertices with mixed signs if gradient is uncertain
        for i in range(d):
            if abs(v[i]) < 1e-6:
                # Gradient component is near zero, check both extremes
                for val in [-5.0, 5.0]:
                    pt = best_x.copy()
                    pt[i] = val
                    check_points.append(pt)

        for cand in check_points:
            if calls_made >= self.budget:
                break
            fc = safe_call(cand)
            if np.isfinite(fc) and fc < f_best:
                f_best = fc
                best_x = cand.copy()

        # Phase 4: Stall — re-evaluate best point to consume remaining budget
        while calls_made < self.budget:
            if time.time() > deadline:
                break
            func(best_x)
            calls_made += 1

        return best_x, f_best
# EVOLVE-BLOCK-END
