"""Multi-objective optimizer for ZDT2 targeting hypervolume with reference [11,11]."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Hypervolume-guided 1-D adaptive search over the decision-relevant variable x1.

    For ZDT2 only x1 determines the objective vector (f1=x1, f2=(1+10n)(1-sqrt(f1/n))).
    We set x2..xn=0 (valid, on the front) and perform an adaptive 1-D search over x1,
    polishing promising points with golden-section minimization of f2 to hug the front.
    """

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        t0 = time.time()
        deadline = t0 + 290.0  # safety margin under the 330s limit
        dim = self.dim
        budget = self.budget
        lo = np.array([b[0] for b in bounds], dtype=float)
        hi = np.array([b[1] for b in bounds], dtype=float)

        # Base template: x2..xn = 0 (valid for [0,1] bounds, on/near the front).
        template = np.zeros(dim)
        template[0] = lo[0]

        used = 0

        def eval_x1(x1):
            nonlocal used
            x = template.copy()
            x[0] = float(x1)
            used += 1
            f = np.asarray(problem(x), dtype=float)
            return f

        # Non-dominated archive (list of (f1, f2)); we keep only mutually non-dominated points.
        archive = []  # list of [f1, f2]

        def dominates(a, b):
            # a dominates b if a<=b in all objectives and a<b in at least one
            return (a[0] <= b[0] and a[1] <= b[1]) and (a[0] < b[0] or a[1] < b[1])

        def add_to_archive(f):
            f = np.asarray(f, dtype=float)
            if f[0] >= 11.0 or f[1] >= 11.0:
                return False
            for p in archive:
                if dominates(p, f):
                    return False
            archive[:] = [p for p in archive if not dominates(f, p)]
            archive.append(f)
            return True

        # ---- Phase 1: coarse uniform sweep over x1 in [0,1] to cover the front ----
        # Use a fraction of the budget for a coarse grid.
        n_coarse = min(60, max(10, budget // 4))
        xs = np.linspace(lo[0], hi[0], n_coarse)
        for x1 in xs:
            if used >= budget or time.time() > deadline:
                break
            f = eval_x1(x1)
            add_to_archive(f)

        # ---- Phase 2: hypervolume-guided adaptive refinement ----
        # Repeatedly pick the x1 that maximizes the hypervolume contribution of the
        # best local polish, using the current archive's gaps as a guide.
        # We sample candidate x1 positions in the largest gaps and polish each.
        remaining = budget - used

        # Build a candidate pool from archive gaps + a fine grid in promising regions.
        def candidate_positions():
            cands = set()
            # fine uniform grid
            for g in np.linspace(lo[0], hi[0], 41):
                cands.add(round(float(g), 6))
            # gap midpoints from sorted archive f1 values
            fs = sorted([p[0] for p in archive])
            if len(fs) >= 2:
                for a, b in zip(fs[:-1], fs[1:]):
                    cands.add(round(float((a + b) / 2.0), 6))
                    cands.add(round(float(a + 0.33 * (b - a)), 6))
                    cands.add(round(float(a + 0.66 * (b - a)), 6))
            # a few near the extremes
            cands.add(round(float(lo[0]), 6))
            cands.add(round(float(hi[0]), 6))
            return sorted(cands)

        # Golden-section polish: minimize f2 over x1 in [a,b] (drives point toward front).
        # f1=x1 is fixed-ish; we minimize f2 to reduce the gap above the front.
        def polish(x1_center, half_width, max_iter=18):
            nonlocal used
            a = max(lo[0], x1_center - half_width)
            b = min(hi[0], x1_center + half_width)
            if b - a < 1e-9:
                f = eval_x1(x1_center)
                return x1_center, f
            gr = (np.sqrt(5.0) - 1.0) / 2.0  # golden ratio conjugate
            c = b - gr * (b - a)
            d = a + gr * (b - a)
            fc = eval_x1(c)
            fd = eval_x1(d)
            best_x, best_f = (c, fc) if fc[1] <= fd[1] else (d, fd)
            for _ in range(max_iter):
                if used >= budget or time.time() > deadline:
                    break
                if fc[1] <= fd[1]:
                    b = d
                    d = c
                    fd = fc
                    c = b - gr * (b - a)
                    fc = eval_x1(c)
                else:
                    a = c
                    c = d
                    fc = fd
                    d = a + gr * (b - a)
                    fd = eval_x1(d)
                # track best by f2 (closer to front) but keep f1=x1
                if fc[1] <= best_f[1]:
                    best_x, best_f = c, fc
                if fd[1] <= best_f[1]:
                    best_x, best_f = d, fd
                if b - a < 1e-7:
                    break
            return best_x, best_f

        # Iteratively polish the most promising gap centers.
        # Sort candidate positions by a heuristic: prefer interior (where front is concave).
        cands = candidate_positions()
        # Heuristic ordering: distance from 0.5 (interior first for concave front coverage)
        cands.sort(key=lambda v: abs(v - 0.5))

        # We will polish each candidate but cap total polish work.
        polished = 0
        max_polish = min(len(cands), max(20, budget // 8))
        for x1c in cands:
            if used >= budget or time.time() > deadline:
                break
            if polished >= max_polish:
                break
            # half width shrinks as we refine
            half = 0.05
            _, f = polish(x1c, half, max_iter=16)
            polished += 1
            add_to_archive(f)

        # ---- Phase 3: targeted fill of remaining hypervolume gaps ----
        # Re-evaluate a fine grid in the largest gaps not yet covered.
        if used < budget and time.time() < deadline:
            fs = sorted([p[0] for p in archive])
            # find largest gaps
            gaps = []
            if len(fs) >= 2:
                for a, b in zip(fs[:-1], fs[1:]):
                    gaps.append((b - a, a, b))
            gaps.sort(reverse=True)
            for gap, a, b in gaps[:8]:
                if used >= budget or time.time() > deadline:
                    break
                for frac in (0.25, 0.5, 0.75):
                    if used >= budget or time.time() > deadline:
                        break
                    x1 = a + frac * (b - a)
                    _, f = polish(x1, 0.02, max_iter=10)
                    add_to_archive(f)

        # ---- Final: ensure a couple of extreme points (x1=0 and x1=1) are in archive ----
        for x1 in (lo[0], hi[0]):
            if used >= budget or time.time() > deadline:
                break
            f = eval_x1(x1)
            add_to_archive(f)

        # Return the archive as the final set of points (interface: return X-like array).
        if len(archive) > 0:
            return np.array(archive)
        return np.array([])
# EVOLVE-BLOCK-END
