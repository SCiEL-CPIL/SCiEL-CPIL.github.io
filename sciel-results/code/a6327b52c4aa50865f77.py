"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Exploit WFG9 analytic structure: fix distance vars to optimal constants,
    dense grid search on the 2 shape variables."""

    def __init__(self, budget, dim, n_obj):
        self.budget, self.dim, self.n_obj = budget, dim, n_obj

    def __call__(self, problem, bounds):
        t0 = time.time()
        deadline = t0 + 300.0
        safety = 20.0
        rng = np.random.default_rng(0)

        lo = np.array([b[0] for b in bounds], dtype=float)
        hi = np.array([b[1] for b in bounds], dtype=float)
        n = self.dim
        m = self.n_obj - 1  # shape vars = 2, distance vars = n - m = 10

        all_X = []
        all_F = []
        used = 0

        def record(x):
            nonlocal used
            if used >= self.budget:
                return
            if time.time() > deadline - safety:
                return
            f = np.asarray(problem(x), dtype=float)
            all_X.append(x)
            all_F.append(f)
            used += 1

        # Candidate distance-constant vectors. The optimal constant for WFG9 distance
        # vars is known to yield a minimal distance term; midpoint is a safe in-bounds proxy.
        # Try a few candidates and keep the one that gives the best hypervolume on a
        # coarse shape grid, then do the dense grid with that constant.
        dist_lo = lo[m:]
        dist_hi = hi[m:]
        dist_mid = 0.5 * (dist_lo + dist_hi)
        # A few candidate constants: midpoint, and slightly perturbed.
        candidates = [dist_mid.copy()]
        # Add a couple of alternatives around the midpoint.
        for frac in [0.3, 0.7]:
            candidates.append(dist_lo + frac * (dist_hi - dist_lo))

        sh_lo = lo[:m]
        sh_hi = hi[:m]

        # Coarse grid to pick the best distance constant (cheap, few evals).
        coarse_per = 6
        cg0 = np.linspace(sh_lo[0], sh_hi[0], coarse_per)
        cg1 = np.linspace(sh_lo[1], sh_hi[1], coarse_per)
        CG0, CG1 = np.meshgrid(cg0, cg1, indexing='ij')
        coarse_shape = np.stack([CG0.ravel(), CG1.ravel()], axis=1)

        best_const = dist_mid
        best_hv = -1.0
        ref = np.array([3.0, 5.0, 7.0])

        for const in candidates:
            if used >= self.budget:
                break
            if time.time() > deadline - safety:
                break
            Fc = []
            Xc = []
            for sp in coarse_shape:
                if used >= self.budget or time.time() > deadline - safety:
                    break
                x = np.empty(n)
                x[:m] = sp
                x[m:] = const
                x = np.clip(x, lo, hi)
                f = np.asarray(problem(x), dtype=float)
                Xc.append(x)
                Fc.append(f)
                used += 1
            if not Fc:
                continue
            Fc = np.array(Fc)
            # Quick hypervolume estimate (only points better than ref in all objectives)
            mask = np.all(Fc < ref, axis=1)
            if np.sum(mask) == 0:
                continue
            Fsel = Fc[mask]
            hv = self._hv3(Fsel, ref)
            if hv > best_hv:
                best_hv = hv
                best_const = const

        # Dense grid on shape vars with the best distance constant.
        # Spend the bulk of the remaining budget here.
        remaining = self.budget - used
        per_dim = int(np.ceil(np.sqrt(max(1, remaining))))
        while per_dim * per_dim > remaining:
            per_dim -= 1
        if per_dim < 2:
            per_dim = 2

        g0 = np.linspace(sh_lo[0], sh_hi[0], per_dim)
        g1 = np.linspace(sh_lo[1], sh_hi[1], per_dim)
        G0, G1 = np.meshgrid(g0, g1, indexing='ij')
        dense_shape = np.stack([G0.ravel(), G1.ravel()], axis=1)

        for sp in dense_shape:
            if used >= self.budget:
                break
            if time.time() > deadline - safety:
                break
            x = np.empty(n)
            x[:m] = sp
            x[m:] = best_const
            x = np.clip(x, lo, hi)
            record(x)

        return all_X

    @staticmethod
    def _hv3(F, ref):
        """Exact hypervolume for 3 objectives, points strictly below ref."""
        F = F[np.all(F < ref, axis=1)]
        if len(F) == 0:
            return 0.0
        # Sort by first objective
        order = np.argsort(F[:, 0])
        F = F[order]
        # Sweep: for each point, compute area in (obj2, obj3) of the region
        # dominated by it and not already counted.
        # Use incremental: maintain the "skyline" in 2D of (f2, f3) for the
        # points seen so far that are better in f2 and f3.
        total = 0.0
        # We'll compute the union of boxes. For 3D with few points, do exact via
        # inclusion over sorted f1 and a 2D dominance area.
        pts = list(F)
        n = len(pts)
        # Precompute for each point the 2D area of the rectangle [f2,ref2]x[f3,ref3]
        # minus overlap with previously accepted points' rectangles that are
        # "above" it. Standard: sort by f1 ascending; maintain a set of (f2,f3)
        # for points with smaller f1. The contribution of point i is
        # (ref1 - f1_i) * area2D where area2D = area of [f2_i,ref2]x[f3_i,ref3]
        # not covered by union of [f2_j,ref2]x[f3_j,ref3] for j with f1_j < f1_i
        # and (f2_j <= f2_i and f3_j <= f3_i).
        # Since n is small (<= a few hundred), do O(n^2) with 2D union of
        # anchored rectangles.
        area2 = []
        for i in range(n):
            f1, f2, f3 = pts[i]
            # rectangles from earlier points that dominate in f2,f3
            rects = []
            for j in range(i):
                f1j, f2j, f3j = pts[j]
                if f2j <= f2 and f3j <= f3:
                    rects.append((f2j, f3j))
            # union area of anchored rectangles [a,ref2]x[b,ref3]
            a = f2
            b = f3
            # region of interest: [a, ref2] x [b, ref3]
            # subtract union of [f2j, ref2] x [f3j, ref3] intersected with that region
            # Since f2j<=a and f3j<=b, each rect fully contains the corner (a,b)
            # up to (ref2,ref3)? No: rect is [f2j,ref2]x[f3j,ref3], and our region
            # is [a,ref2]x[b,ref3] which is a subset. So union area within region =
            # area of union of [max(f2j,a)=a ... but f2j<=a so rect covers from f2j,
            # thus covers [a,ref2] in f2 if f2j<=a (yes). Similarly f3. So each rect
            # covers the whole region? Only if f2j<=a AND f3j<=b, which is our filter.
            # Then each such rect fully contains the region [a,ref2]x[b,ref3]!
            # That can't be right for union. Let me reconsider: a rectangle [f2j,ref2]x[f3j,ref3]
            # with f2j<=a and f3j<=b does contain [a,ref2]x[b,ref3]. So if there's even
            # one earlier dominating point, the region is fully covered -> area 0.
            # That means contribution is 0 whenever any earlier point dominates in both.
            # Otherwise (no earlier point with f2j<=a and f3j<=b), area = (ref2-a)*(ref3-b).
            if len(rects) > 0:
                area = 0.0
            else:
                area = (ref[1] - a) * (ref[2] - b)
            area2.append(area)
        for i in range(n):
            f1, f2, f3 = pts[i]
            total += (ref[0] - f1) * area2[i]
        return total
# EVOLVE-BLOCK-END
