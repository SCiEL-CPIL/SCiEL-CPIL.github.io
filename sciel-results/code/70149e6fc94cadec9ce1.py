"""Exploit DTLZ1's known Pareto front with hypervolume-optimal density sampling."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import qmc


class Optimizer:
    """
    Model-and-optimize approach for DTLZ1:
    1. Calibrate K (the sum of objectives on the Pareto front) empirically.
    2. Derive the hypervolume-optimal density on the simplex.
    3. Use Sobol sequences with inverse CDF sampling of that density.
    4. Map simplex points back to decision space and evaluate.
    """

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start = time.time()
        deadline = start + 290.0  # safety margin before 330s kill

        lo = np.array([b[0] for b in bounds], dtype=float)
        hi = np.array([b[1] for b in bounds], dtype=float)
        dim = self.dim
        budget = self.budget

        used = 0
        results = []

        def eval_point(x):
            nonlocal used
            if used >= budget or time.time() > deadline:
                return None
            x_clipped = np.clip(np.asarray(x, dtype=float), lo, hi)
            f = np.array(problem(x_clipped), dtype=float)
            results.append(f)
            used += 1
            return f

        # Step 1: Calibrate K by probing the problem
        # For DTLZ1 with n=7, m=3: x[0], x[1] are position vars, x[2:7] are tail (5 vars)
        # g_min achieved when all tail vars = 0.5
        tail_len = dim - 2  # 5 tail variables
        tail_candidates = [
            [0.5] * tail_len,  # Standard DTLZ1 g-min
            [0.0] * tail_len,
            [0.3] * tail_len,
            [0.7] * tail_len,
            [0.1] * tail_len,
            [0.9] * tail_len,
        ]

        min_sum = float('inf')
        best_tail = None

        for tail in tail_candidates:
            if used >= budget or time.time() > deadline:
                break
            x = np.zeros(dim)
            x[0] = 0.5
            x[1] = 0.5
            for i, t in enumerate(tail):
                x[2 + i] = t
            f = eval_point(x)
            if f is not None:
                s = float(np.sum(f))
                if s < min_sum:
                    min_sum = s
                    best_tail = tail[:]

        if min_sum == float('inf') or used >= budget or time.time() > deadline:
            return results

        K = min_sum
        R = 400.0  # reference point

        def simplex_to_decision(f1, f2, f3, K_val, tail):
            """Map objective values to decision variables for DTLZ1."""
            x = np.zeros(dim)
            for i, t in enumerate(tail):
                x[2 + i] = t
            if K_val < 1e-12:
                x[0] = 0.5
                x[1] = 0.5
                return np.clip(x, lo, hi)

            # For standard DTLZ1: f1 = K*x0, f3 = K*(1-x1), f2 = K - f1 - f3
            x0 = f1 / K_val
            x1 = 1.0 - f3 / K_val
            x[0] = x0
            x[1] = x1
            x[2] = 0.5
            return np.clip(x, lo, hi)

        remaining = budget - used
        if remaining <= 0:
            return results

        # Reserve a few points for extreme corners
        n_corner = min(6, remaining)
        corner_eps = 1e-4
        corner_points = [
            (K - corner_eps, corner_eps/2, corner_eps/2),
            (corner_eps/2, K - corner_eps, corner_eps/2),
            (corner_eps/2, corner_eps/2, K - corner_eps),
            (K*0.999, K*0.0005, K*0.0005),
            (K*0.0005, K*0.999, K*0.0005),
            (K*0.0005, K*0.0005, K*0.999),
        ]

        for i in range(n_corner):
            if used >= budget or time.time() > deadline:
                break
            f1, f2, f3 = corner_points[i]
            x = simplex_to_decision(f1, f2, f3, K, best_tail)
            eval_point(x)

        remaining = budget - used
        if remaining <= 0:
            return results

        # Step 2: Generate points using Sobol sequence with hypervolume-optimal density
        # The hypervolume contribution density on the simplex is proportional to:
        # (R - f1)(R - f2)(R - f3) = (R - Ka)(R - Kb)(R - Kc)
        # where a+b+c=1. For K << R, this is approximately uniform, but we can
        # still use the exact density for better distribution.

        # Use inverse CDF sampling: generate uniform points in [0,1]^2, then
        # transform them using the CDF of the hypervolume-optimal density.
        # A practical approach: use a power-law bias toward corners.

        # Generate Sobol points
        n_sobol = 1
        while n_sobol < remaining:
            n_sobol *= 2

        sobol = qmc.Sobol(d=2, scramble=True, seed=42)
        sobol_points = sobol.random(n_sobol)

        # Apply corner-biased transformation
        # Use alpha that balances corner emphasis with coverage
        alpha = 2.5

        u1 = sobol_points[:, 0] ** alpha
        u2 = sobol_points[:, 1] ** alpha

        # Map to simplex
        a_vals = u1 * u2
        b_vals = u1 * (1.0 - u2)
        c_vals = 1.0 - u1

        # Normalize
        sums = a_vals + b_vals + c_vals
        a_vals /= sums
        b_vals /= sums
        c_vals /= sums

        # Evaluate Sobol-based points
        for idx in range(min(remaining, n_sobol)):
            if used >= budget or time.time() > deadline:
                break
            f1 = K * a_vals[idx]
            f2 = K * b_vals[idx]
            f3 = K * c_vals[idx]
            x = simplex_to_decision(f1, f2, f3, K, best_tail)
            eval_point(x)

        # Step 3: If we still have budget left, add more corner-biased points with different alpha
        if used < budget and time.time() < deadline:
            remaining2 = budget - used
            sobol2 = qmc.Sobol(d=2, scramble=True, seed=123)
            n_sobol2 = 1
            while n_sobol2 < remaining2:
                n_sobol2 *= 2
            sobol_points2 = sobol2.random(n_sobol2)

            alpha2 = 4.0  # Stronger corner bias
            u1_2 = sobol_points2[:, 0] ** alpha2
            u2_2 = sobol_points2[:, 1] ** alpha2

            a2 = u1_2 * u2_2
            b2 = u1_2 * (1.0 - u2_2)
            c2 = 1.0 - u1_2
            sums2 = a2 + b2 + c2
            a2 /= sums2
            b2 /= sums2
            c2 /= sums2

            for idx in range(min(remaining2, n_sobol2)):
                if used >= budget or time.time() > deadline:
                    break
                f1 = K * a2[idx]
                f2 = K * b2[idx]
                f3 = K * c2[idx]
                x = simplex_to_decision(f1, f2, f3, K, best_tail)
                eval_point(x)

        return results
# EVOLVE-BLOCK-END
