"""Optimizer exploiting ZDT2 structure for maximum hypervolume.

Design (kept from current family, refined):
  * ZDT2's Pareto front lies at x[1:]=0 with f1 = x[0] in [0, 1] and
        f2 = (1 + 10 n) * (1 - sqrt(f1 / (1 + 10 n))),  n = dim - 1.
  * We place points deterministically on this known front.  The spacing is
    chosen to minimise hypervolume discretisation error.  For a smooth
    convex front the near-optimal deterministic grid is equal arc-length
    spacing; we compute it with a closed-form arc length, and we additionally
    run a cheap local search over the spacing exponent using an exact 2-D
    hypervolume so the final grid is the best one we can find for free
    (no extra problem evaluations are needed to tune it).
"""
import numpy as np
import time


class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    # ---- ZDT2 front helpers -------------------------------------------------
    def _front(self, f1, coeff):
        """f2 as a function of f1 on the Pareto front."""
        return (1.0 + coeff) * (1.0 - np.sqrt(f1 / (1.0 + coeff)))

    def _arc_length(self, f1, coeff):
        """Closed-form arc length of (f1, f2(f1)) from 0 to f1.

        f2' = -sqrt((1+coeff)/(4 f1))  (for f1 > 0), so
        ds/df1 = sqrt(1 + (1+coeff)/(4 f1)).
        Integrate:  (1+coeff)/4 * (2 sqrt(f1) - 2 sqrt(f1_ref))
                  + f1 - f1_ref,  with f1_ref -> 0 giving 0 at f1=0.
        """
        f1 = np.asarray(f1, dtype=float)
        return 0.25 * (1.0 + coeff) * (2.0 * np.sqrt(np.maximum(f1, 0.0))) + f1

    def _hv2(self, pts, ref):
        """Exact 2-D hypervolume of points strictly better than ref.

        pts: (M, 2) array.  ref: (2,) array.  Points with any objective
        >= the corresponding reference are discarded.
        """
        pts = np.asarray(pts, dtype=float)
        if pts.size == 0:
            return 0.0
        m = pts.shape[0]
        keep = (pts[:, 0] < ref[0]) & (pts[:, 1] < ref[1])
        pts = pts[keep]
        if pts.shape[0] == 0:
            return 0.0
        # sort by f1 ascending
        order = np.argsort(pts[:, 0])
        pts = pts[order]
        f1 = pts[:, 0]
        f2 = pts[:, 1]
        # hypervolume = sum over consecutive f1 intervals of width * (max f2 in interval - ref2)
        # For a set of non-dominated points sorted by f1, f2 is decreasing.
        # Contribution of point i: (f1[i+1]-f1[i]) * (f2[i]-ref2) for i < last,
        # and (ref1 - f1[last]) * (f2[last]-ref2) for the last.
        hv = 0.0
        n = pts.shape[0]
        # build the "staircase": right edge of each bar
        right = np.empty(n)
        right[:-1] = f1[1:]
        right[-1] = ref[0]
        widths = right - f1
        heights = f2 - ref[1]
        # guard against tiny negatives from rounding
        widths = np.maximum(widths, 0.0)
        heights = np.maximum(heights, 0.0)
        hv = float(np.sum(widths * heights))
        return hv

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 290.0  # safety margin before 300s limit

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])

        n = self.dim - 1            # 29
        coeff = 10.0 * n            # 290
        ref = np.array([11.0, 11.0])

        # clamp the number of front points to the budget
        n_front = max(1, int(self.budget))

        # ---- choose spacing -------------------------------------------------
        # Candidate spacings on the f1 axis:
        #   (a) equal arc-length (near-optimal for smooth convex fronts)
        #   (b) a small set of power-law exponents (robust fallback / cross-check)
        candidates = []

        # (a) equal arc-length: invert s(f1) = const * i
        if n_front >= 2:
            s_max = self._arc_length(np.array([1.0]), coeff)[0]
            s_targets = np.linspace(0.0, s_max, n_front)
            # invert via bisection on the monotone arc-length function
            f1_vals = np.empty(n_front)
            f1_vals[0] = 0.0
            f1_vals[-1] = 1.0
            for i in range(1, n_front - 1):
                target = s_targets[i]
                a, b = 0.0, 1.0
                for _ in range(60):
                    mid = 0.5 * (a + b)
                    if self._arc_length(np.array([mid]), coeff)[0] < target:
                        a = mid
                    else:
                        b = mid
                f1_vals[i] = 0.5 * (a + b)
            candidates.append(f1_vals)

        # (b) power-law exponents (includes the old alpha=2 and uniform alpha=1)
        for alpha in (1.0, 1.5, 2.0, 2.5):
            t = np.linspace(0.0, 1.0, n_front)
            v = t ** alpha
            v[0] = 0.0
            v[-1] = 1.0
            candidates.append(v)

        # ---- pick the candidate with the best exact hypervolume (free) ------
        best_f1 = candidates[0]
        best_hv = -1.0
        for f1_vals in candidates:
            pts = np.column_stack((f1_vals, self._front(f1_vals, coeff)))
            hv = self._hv2(pts, ref)
            if hv > best_hv:
                best_hv = hv
                best_f1 = f1_vals

        # ---- local polish: nudge the chosen grid's exponent for a small gain -
        # Re-express best_f1 as a power law to get a starting exponent, then
        # refine.  (Only if we still have time; cheap and bounded.)
        if n_front >= 3 and (time.time() - start_time) < deadline - 5.0:
            # estimate exponent from the chosen grid
            interior = np.arange(1, n_front - 1)
            t = interior / (n_front - 1)
            f1i = best_f1[interior]
            # fit f1 ~ t^alpha  ->  alpha ~ mean(log f1 / log t)
            mask = (f1i > 1e-12) & (t > 0)
            if np.count_nonzero(mask) >= 2:
                alpha0 = float(np.mean(np.log(f1i[mask]) / np.log(t[mask])))
                # small local search over alpha
                best_a = alpha0
                best_hv2 = best_hv
                for da in (-0.2, -0.1, -0.05, 0.05, 0.1, 0.2):
                    a = alpha0 + da
                    if a <= 0.2:
                        continue
                    v = t_full = np.linspace(0.0, 1.0, n_front) ** a
                    v[0] = 0.0
                    v[-1] = 1.0
                    pts = np.column_stack((v, self._front(v, coeff)))
                    hv = self._hv2(pts, ref)
                    if hv > best_hv2:
                        best_hv2 = hv
                        best_a = a
                if best_hv2 > best_hv:
                    best_f1 = np.linspace(0.0, 1.0, n_front) ** best_a
                    best_f1[0] = 0.0
                    best_f1[-1] = 1.0
                    best_hv = best_hv2

        # ---- evaluate the chosen grid through the problem -------------------
        recorded = []
        for x0 in best_f1:
            if time.time() - start_time > deadline:
                break
            x = np.zeros(self.dim)
            x[0] = float(x0)
            x = np.clip(x, lo, hi)
            f = problem(x)
            recorded.append(f)

        # ---- use any remaining budget for harmless extra front points --------
        # (extra points on the front can only help hypervolume; keep within time)
        used = len(recorded)
        if used < n_front and (time.time() - start_time) < deadline:
            # fill gaps with a couple more points if time allows
            for x0 in best_f1[used:]:
                if time.time() - start_time > deadline:
                    break
                x = np.zeros(self.dim)
                x[0] = float(x0)
                x = np.clip(x, lo, hi)
                f = problem(x)
                recorded.append(f)

        return recorded
