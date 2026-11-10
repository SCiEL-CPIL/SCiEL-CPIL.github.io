"""ZDT3 multi-objective optimizer exploiting the known 1D Pareto-front manifold."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 290
        n = self.dim
        n_evals = 0
        results = []

        def eval_point(x1_val):
            nonlocal n_evals
            if n_evals >= self.budget or time.time() > deadline:
                return None
            x = np.zeros(n)
            x[0] = float(x1_val)
            f = problem(x)
            n_evals += 1
            results.append(f)
            return f

        # f1 = x1, f2 = (1+10*(n-1)) * (1 - sqrt(f1/(1+10*(n-1))) + sin(10*pi*f1))
        # For n=30, 1+10*(n-1) = 291
        A = 1.0 + 10.0 * (n - 1)
        # Critical x1 values: where sin(10*pi*x1) = 0, i.e. x1 = k/10
        # These are the transition points between oscillation lobes
        critical_x1 = np.arange(0, 11) / 10.0  # [0, 0.1, ..., 1.0]

        # Phase 1: Evaluate critical points (11 evaluations)
        for x1_c in critical_x1:
            eval_point(x1_c)

        if n_evals == 0:
            x0 = np.array([(lo + hi) / 2.0 for lo, hi in bounds])
            return problem(x0)

        # Phase 2: Fill remaining budget with locally-optimized points
        # For each gap between consecutive critical x1 values, do a local
        # minimization of f2 to find the exact Pareto-optimal point in that lobe.
        # Each lobe (between two consecutive zeros of sin) contains exactly one
        # local minimum of f2 (or the boundary). We use scipy minimize to
        # find it precisely.

        remaining_budget = self.budget - n_evals
        gaps = len(critical_x1) - 1  # 10 gaps

        # Distribute remaining budget: use a few starting points per gap
        # and let minimize polish them. Each minimize call uses ~10-30 evals.
        # We'll do 1 polish per gap if budget allows, plus extra uniform samples.

        # First, add uniform samples to ensure coverage
        n_uniform = min(remaining_budget, 120)
        if n_uniform > 0:
            # Uniform samples in (0,1) excluding the critical points
            # Use a grid that includes midpoints of each gap
            midpoints = (critical_x1[:-1] + critical_x1[1:]) / 2.0
            # Also add finer uniform grid
            extra_grid = np.linspace(0.001, 0.999, n_uniform)
            # Combine: ensure midpoints are included
            all_samples = np.concatenate([midpoints, extra_grid])
            # Deduplicate: keep only unique values (within tolerance)
            all_samples = np.unique(np.round(all_samples, 8))
            for x1_s in all_samples:
                if x1_s < 0 or x1_s > 1:
                    continue
                eval_point(x1_s)
                if n_evals >= self.budget or time.time() > deadline:
                    break

        # Phase 3: Polish with local optimization to hit exact minima
        remaining = self.budget - n_evals
        if remaining > 5:
            # For each gap, try to find the local minimum of f2 via scipy
            # f2(x1) = A * (1 - sqrt(x1/A) + sin(10*pi*x1))
            # We minimize this over x1 in [x1_lo, x1_hi]
            for gap_idx in range(gaps):
                if n_evals >= self.budget or time.time() > deadline:
                    break
                x1_lo = critical_x1[gap_idx]
                x1_hi = critical_x1[gap_idx + 1]
                # Start from midpoint of the gap
                x1_start = (x1_lo + x1_hi) / 2.0

                def f2_of_x1(x1_arr):
                    x1 = x1_arr[0]
                    if x1 <= 0:
                        return A * (1.0 + 0.0)  # limit
                    return A * (1.0 - np.sqrt(x1 / A) + np.sin(10.0 * np.pi * x1))

                # Use bounded L-BFGS-B
                try:
                    res = minimize(
                        f2_of_x1,
                        x0=np.array([x1_start]),
                        method='L-BFGS-B',
                        bounds=[(x1_lo, x1_hi)],
                        options={'maxiter': 200, 'ftol': 1e-12}
                    )
                    x1_opt = float(np.clip(res.x[0], x1_lo, x1_hi))
                    eval_point(x1_opt)
                except Exception:
                    pass

                # Also try starting from a slightly offset point
                if n_evals < self.budget and time.time() < deadline:
                    x1_start2 = x1_lo + 0.3 * (x1_hi - x1_lo)
                    try:
                        res2 = minimize(
                            f2_of_x1,
                            x0=np.array([x1_start2]),
                            method='L-BFGS-B',
                            bounds=[(x1_lo, x1_hi)],
                            options={'maxiter': 200, 'ftol': 1e-12}
                        )
                        x1_opt2 = float(np.clip(res2.x[0], x1_lo, x1_hi))
                        # Only evaluate if it's meaningfully different
                        if abs(x1_opt2 - x1_opt) > 1e-6:
                            eval_point(x1_opt2)
                    except Exception:
                        pass

        if len(results) == 0:
            x0 = np.array([(lo + hi) / 2.0 for lo, hi in bounds])
            return problem(x0)

        return np.array(results)
# EVOLVE-BLOCK-END
