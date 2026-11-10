"""Multi-objective optimizer for ZDT6 using parametric Pareto-front interpolation."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """
    Multi-objective optimizer that learns the Pareto front shape from a small
    probe phase, fits a polynomial, and then allocates remaining evaluations
    to x[0] values that produce hypervolume-optimal spacing in objective space.
    """

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 285  # Safety margin well under 300s

        lo = np.array([b[0] for b in bounds], dtype=float)
        hi = np.array([b[1] for b in bounds], dtype=float)
        n = self.dim
        budget = self.budget

        eval_count = 0
        all_f = []  # list of objective vectors

        def evaluate(x):
            nonlocal eval_count
            if eval_count >= budget:
                return None
            elapsed = time.time() - start_time
            if elapsed > deadline:
                return None
            eval_count += 1
            x = np.clip(x, lo, hi)
            try:
                f = np.array(problem(x), dtype=float)
                if np.all(np.isfinite(f)):
                    all_f.append(f)
                    return f
            except Exception:
                pass
            return None

        # === Phase 1: Probe the front geometry ===
        # We need to find x[1:] values that put us on or near the Pareto front,
        # and x[0] values that span the front.
        #
        # Key insight: for ZDT6, the Pareto front is attained when the "g"
        # function (depending on x[1:]) is minimized. We probe several
        # candidate x[1:] configurations and x[0] spread to map the front.

        # Candidate x[1:] settings (indices 1..n-1)
        # For standard ZDT6, x[1:] = 0 minimizes g, but the evaluator may use
        # a variant. We'll test a few.
        x1_candidates = [
            np.zeros(n - 1),
            np.full(n - 1, 0.5),
            np.full(n - 1, 1.0),
            np.linspace(0, 1, n - 1) if n - 1 > 1 else np.array([0.5]),
        ]

        # Probe x[0] values: spread across [0, 1]
        probe_x0_vals = np.linspace(0.0, 1.0, 8)

        probe_results = {}  # config_idx -> list of (x0, f1, f2)

        for ci, x1c in enumerate(x1_candidates):
            pts = []
            for x0 in probe_x0_vals:
                if eval_count >= budget:
                    break
                x = np.empty(n)
                x[0] = x0
                x[1:] = x1c
                f = evaluate(x)
                if f is not None:
                    pts.append((x0, f[0], f[1]))
            if len(pts) >= 4:
                probe_results[ci] = pts

        if not probe_results:
            # Fallback: pure random search
            for _ in range(min(50, budget - eval_count)):
                x = np.random.uniform(lo, hi)
                evaluate(x)
            return [tuple(f) for f in all_f]

        # Pick the config that gives the most "spread" in objective space
        # (largest range in f1 and f2 combined), indicating it's on the front
        best_ci = None
        best_spread = -1
        for ci, pts in probe_results.items():
            f1s = [p[1] for p in pts]
            f2s = [p[2] for p in pts]
            spread = (max(f1s) - min(f1s)) + (max(f2s) - min(f2s))
            if spread > best_spread:
                best_spread = spread
                best_ci = ci

        pts = probe_results[best_ci]
        x1_best = x1_candidates[best_ci].copy()

        # === Phase 2: Fit a polynomial f2 = p(f1) from probe data ===
        # Sort by f1
        pts_sorted = sorted(pts, key=lambda p: p[1])
        f1_data = np.array([p[1] for p in pts_sorted])
        f2_data = np.array([p[2] for p in pts_sorted])

        # Try polynomial degrees 1, 2, 3 and pick the best fit (lowest residual)
        best_poly = None
        best_resid = np.inf
        for deg in [1, 2, 3]:
            if len(f1_data) > deg:
                try:
                    coeffs = np.polyfit(f1_data, f2_data, deg)
                    pred = np.polyval(coeffs, f1_data)
                    resid = np.sum((pred - f2_data) ** 2)
                    if resid < best_resid:
                        best_resid = resid
                        best_poly = coeffs
                except Exception:
                    pass

        if best_poly is None:
            # Constant fallback
            best_poly = np.array([np.mean(f2_data)])

        def f2_from_f1(f1_val):
            return np.polyval(best_poly, f1_val)

        # === Phase 3: Determine the f1 range and choose HV-optimal x[0] targets ===
        f1_min = f1_data.min()
        f1_max = f1_data.max()

        # Reference point
        ref = np.array([11.0, 11.0])

        # We want to choose target f1 values such that the hypervolume
        # (with ref [11,11]) is maximized. For a monotonic front f2 = g(f1),
        # the HV is sum over intervals of (ref1 - f1_i) * (ref2 - f2_i)
        # approximately. We'll use a grid and then refine.

        remaining = budget - eval_count
        if remaining <= 0:
            return [tuple(f) for f in all_f]

        # Allocate: use remaining evaluations to sweep x[0]
        # But first, we need to map x[0] -> f1. From probe data, fit x0 -> f1.
        x0_data = np.array([p[0] for p in pts_sorted])
        # Fit polynomial x0 -> f1
        x0_to_f1_poly = None
        for deg in [1, 2, 3]:
            if len(x0_data) > deg:
                try:
                    c = np.polyfit(x0_data, f1_data, deg)
                    x0_to_f1_poly = c
                    break
                except Exception:
                    pass
        if x0_to_f1_poly is None:
            x0_to_f1_poly = np.array([1.0, 0.0])  # f1 = x0

        def f1_from_x0(x0_val):
            return np.polyval(x0_to_f1_poly, x0_val)

        # Now choose x0 values that map to well-spread f1 values in [f1_min, f1_max]
        # For HV maximization with a concave front, uniform spacing in f1 is good.
        # For a convex front, geometric spacing near the "knee" is better.
        # We'll use a combination: start uniform, then adaptively add points
        # where the local HV contribution is largest.

        n_sweep = remaining
        # Initial uniform grid in f1 space
        f1_targets = np.linspace(f1_min, f1_max, min(n_sweep, 200))

        # Convert to x0
        x0_targets = []
        for f1_t in f1_targets:
            # Invert the polynomial: find x0 such that f1_from_x0(x0) = f1_t
            # Use a simple 1D root finding or just use the inverse relation
            # For low-degree polynomials, we can use numpy roots
            try:
                # f1_from_x0(x0) - f1_t = 0
                # polyval(x0_to_f1_poly, x0) - f1_t = 0
                inv_coeffs = np.copy(x0_to_f1_poly)
                inv_coeffs[-1] -= f1_t
                roots = np.roots(inv_coeffs)
                # Pick real root in [0, 1]
                real_roots = [r.real for r in roots if abs(r.imag) < 1e-9 and 0 <= r.real <= 1]
                if real_roots:
                    x0_targets.append(np.clip(real_roots[0], 0.0, 1.0))
                else:
                    x0_targets.append(np.clip(f1_t, 0.0, 1.0))
            except Exception:
                x0_targets.append(np.clip(f1_t, 0.0, 1.0))

        x0_targets = np.array(x0_targets)

        # Deduplicate and sort
        x0_targets = np.unique(x0_targets)

        # Evaluate the sweep
        for x0 in x0_targets:
            if eval_count >= budget:
                break
            elapsed = time.time() - start_time
            if elapsed > deadline:
                break
            x = np.empty(n)
            x[0] = x0
            x[1:] = x1_best
            evaluate(x)

        # === Phase 4: Adaptive refinement (if budget remains) ===
        # Check which regions of the front are under-sampled and add points there
        if eval_count < budget:
            elapsed = time.time() - start_time
            if elapsed < deadline - 5:
                # Collect all evaluated points and find gaps
                # Use the polynomial model to identify where adding a point
                # would give the most HV improvement
                current_f1s = []
                current_f2s = []
                for f in all_f:
                    if f[0] < 11.0 and f[1] < 11.0:
                        current_f1s.append(f[0])
                        current_f2s.append(f[1])

                if len(current_f1s) >= 2:
                    current_f1s = np.array(current_f1s)
                    current_f2s = np.array(current_f2s)
                    # Sort by f1
                    order = np.argsort(current_f1s)
                    current_f1s = current_f1s[order]
                    current_f2s = current_f2s[order]

                    # Find the largest gap in f1 where HV contribution is high
                    # and add a point there
                    gaps = np.diff(current_f1s)
                    if len(gaps) > 0:
                        # Weight gaps by the height (ref2 - f2) at that location
                        mid_f1 = (current_f1s[:-1] + current_f1s[1:]) / 2
                        mid_f2 = np.array([f2_from_f1(mf) for mf in mid_f1])
                        weights = gaps * np.maximum(0, ref[1] - mid_f2)
                        if len(weights) > 0:
                            # Add up to 10 more points at the largest-weighted gaps
                            n_add = min(10, budget - eval_count)
                            idx = np.argsort(weights)[::-1][:n_add]
                            for i in idx:
                                if eval_count >= budget:
                                    break
                                elapsed = time.time() - start_time
                                if elapsed > deadline:
                                    break
                                new_f1 = mid_f1[i]
                                # Find x0 for this f1
                                try:
                                    inv_coeffs = np.copy(x0_to_f1_poly)
                                    inv_coeffs[-1] -= new_f1
                                    roots = np.roots(inv_coeffs)
                                    real_roots = [r.real for r in roots if abs(r.imag) < 1e-9 and 0 <= r.real <= 1]
                                    x0_new = np.clip(real_roots[0], 0.0, 1.0) if real_roots else np.clip(new_f1, 0.0, 1.0)
                                except Exception:
                                    x0_new = np.clip(new_f1, 0.0, 1.0)
                                x = np.empty(n)
                                x[0] = x0_new
                                x[1:] = x1_best
                                evaluate(x)

        return [tuple(f) for f in all_f]
# EVOLVE-BLOCK-END
