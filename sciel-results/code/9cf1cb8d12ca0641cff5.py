"""Multi-objective optimizer on WFG7: LHS global spread + Nelder-Mead local polish."""
# EVOLVE-BLOCK-START
import time
import numpy as np


def _hypervolume3(points, ref):
    """Exact 3D hypervolume of points strictly better than ref in every objective."""
    pts = [tuple(p) for p in points if all(p[i] < ref[i] for i in range(3))]
    if not pts:
        return 0.0
    pts = sorted(pts, key=lambda p: p[0])
    # sweep on x
    # build list of (x, y, z); dominated points in yz can be pruned per x-slice
    # For small sets do O(n^3) incremental: add points in increasing x, maintain
    # the set of (y,z) that are not dominated by any later (larger x) point's region.
    # Simpler exact: sort by x desc, maintain skyline of (y,z) for the 2D problem.
    # HV = sum over points of dx * area of 2D skyline contribution.
    # Use incremental 2D skyline area.
    pts = sorted(pts, key=lambda p: -p[0])  # decreasing x
    # 2D skyline of (y,z): a point (y,z) is on skyline if no other has y<= and z<=
    # We maintain list sorted by y with decreasing z (the skyline).
    # area of skyline = sum over consecutive skyline points of (y_{i+1}-y_i)*(z_i - ref_z)
    # Actually area under skyline between y and ref_y, z and ref_z.
    skyline = []  # list of (y,z) sorted by y increasing, z decreasing
    total = 0.0
    prev_x = None
    # We'll accumulate: for each x-slice, the 2D area of points with x' >= current x.
    # HV = integral over x of A(x) dx where A(x) = 2D area of {(y,z): exists point with x_p>=x, y_p<=y, z_p<=z}
    # Equivalent: sort by x desc, insert point, HV += (x_prev - x_cur) * A_after_insert
    # where x_prev is the previous (larger) x.
    # Let's do: iterate points by decreasing x. Keep A = current 2D skyline area of inserted points.
    # Between this point's x and the next (larger) x, area is A_before. Hmm.
    # Standard: HV = sum_i (x_{i-1} - x_i) * A_i where A_i is 2D area of points among {0..i} (those with x >= x_i).
    # Let me just do O(n^3) direct for safety given n<=~60.
    # Direct: for each point i, its contribution = (ref_x - x_i) * (2D area of (y,z) region dominated by i but not by any point with x<x_i that also covers it).
    # This is getting complex; n is small, use inclusion via sorting.
    # Use the well-known O(n^2) for 3D: sort by x, then for each point compute 2D dominated area minus overlap.
    # Given small n, do the clean incremental skyline.
    # Re-derive cleanly:
    # Sort points by x ascending. Process from largest x to smallest.
    # Let x[0] >= x[1] >= ... (descending). Define x_{-1} = ref_x.
    # Let S_k = set of points {0..k}. A_k = 2D skyline area of S_k (in yz, against ref).
    # HV = sum_{k=0}^{n-1} (x_k - x_{k+1}) * A_k, where x_n = ref_x.
    # Wait that double counts. Correct formula:
    # HV = sum_{k=0}^{n-1} (x_k - x_{k+1}) * A_{k}, x_{n}=ref_x, where A_k is area using points 0..k.
    # Hmm, let me verify with 1 point: A_0 = (ref_y-y0)(ref_z-z0). x_0 - x_1 = x_0 - ref_x. HV=(ref_x-x0)(ref_y-y0)(ref_z-z0). Correct.
    # With 2 points x0>x1: HV=(x0-x1)*A_0 + (x1-ref_x)*A_1. A_1 includes both points' skyline. Correct.
    def skyline_area(pairs):
        # pairs: list of (y,z); return 2D dominated area vs ref
        if not pairs:
            return 0.0
        # keep only skyline: sort by y asc, keep decreasing z
        s = sorted(pairs, key=lambda p: p[0])
        sky = []
        best_z = None
        # skyline: as y increases, z must strictly decrease to be non-dominated
        for y, z in s:
            if best_z is None or z < best_z:
                sky.append((y, z))
                best_z = z
        # area: sum over sky of (y_{i+1} - y_i) * (z_i - ref_z) where y_{last+1}=ref_y
        area = 0.0
        for i, (y, z) in enumerate(sky):
            y_next = sky[i+1][0] if i+1 < len(sky) else ref[1]
            area += (y_next - y) * (z - ref[2])
        return area

    # descending x
    xs = [p[0] for p in pts]
    yz = [(p[1], p[2]) for p in pts]
    n = len(pts)
    A = 0.0
    hv = 0.0
    # build incrementally: maintain sorted structure? n small, recompute skyline area each step O(n^2 log n)
    acc = []
    for k in range(n):
        acc.append(yz[k])
        A = skyline_area(acc)
        x_next = xs[k+1] if k+1 < n else ref[0]
        hv += (xs[k] - x_next) * A
    return hv


class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget, self.dim, self.n_obj = budget, dim, n_obj

    def __call__(self, problem, bounds):
        t0 = time.time()
        DEADLINE = 300.0
        MARGIN = 15.0
        dim = self.dim
        lo = np.array([b[0] for b in bounds], dtype=float)
        hi = np.array([b[1] for b in bounds], dtype=float)
        rng = np.random.default_rng(0)
        used = 0

        def eval_x(x):
            nonlocal used
            f = np.asarray(problem(x), dtype=float)
            used += 1
            return f

        def deadline():
            return (time.time() - t0) > (DEADLINE - MARGIN)

        # ---- Phase 1: LHS global coverage ----
        n_lhs = min(40, self.budget)
        # LHS: for each dimension, partition [0,1] into n_lhs equal intervals, assign one
        # sample per interval, permute per dimension.
        perm = [rng.permutation(n_lhs) for _ in range(dim)]
        lhs_u = np.empty((n_lhs, dim))
        for d in range(dim):
            idx = perm[d]
            lhs_u[idx, d] = (np.arange(n_lhs) + rng.random(n_lhs)) / n_lhs
        X = lo + (hi - lo) * lhs_u
        F = np.array([eval_x(x) for x in X])

        def nondominated_idx(F):
            n = len(F)
            keep = np.ones(n, dtype=bool)
            for i in range(n):
                if not keep[i]:
                    continue
                for j in range(n):
                    if i == j or not keep[j]:
                        continue
                    if np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                        keep[i] = False
                        break
            return np.where(keep)[0]

        # ---- Phase 2: Nelder-Mead on non-dominated points ----
        def nelder_mead(x0, f0, max_iter=25, tol=1e-6):
            # minimize a scalar aggregate of objectives (sum) as local guide
            def obj(x):
                return float(np.sum(problem(x)))
            nonlocal used
            n = len(x0)
            simplex = [np.array(x0, dtype=float)]
            fvals = [float(np.sum(f0))]
            used += 0  # f0 already counted
            for i in range(n):
                v = simplex[0].copy()
                v[i] += 0.05 * (hi[i] - lo[i]) if (hi[i]-lo[i]) != 0 else 0.05
                v = np.clip(v, lo, hi)
                simplex.append(v)
                fvals.append(obj(v))
                used += 1
            best = 0
            for _ in range(max_iter):
                if deadline():
                    break
                order = np.argsort(fvals)
                simplex = [simplex[i] for i in order]
                fvals = [fvals[i] for i in order]
                centroid = np.mean(simplex[:-1], axis=0)
                xr = 2 * centroid - simplex[-1]
                xr = np.clip(xr, lo, hi)
                fr = obj(xr); used += 1
                if fr < fvals[0]:
                    xe = centroid + 2 * (xr - centroid)
                    xe = np.clip(xe, lo, hi)
                    fe = obj(xe); used += 1
                    if fe < fr:
                        simplex[-1] = xe; fvals[-1] = fe
                    else:
                        simplex[-1] = xr; fvals[-1] = fr
                elif fr < fvals[-2]:
                    simplex[-1] = xr; fvals[-1] = fr
                else:
                    xc = 0.5 * (centroid + simplex[-1])
                    xc = np.clip(xc, lo, hi)
                    fc = obj(xc); used += 1
                    if fc < fvals[-1]:
                        simplex[-1] = xc; fvals[-1] = fc
                    else:
                        # contract all toward best
                        for i in range(1, n+1):
                            simplex[i] = 0.5 * (simplex[0] + simplex[i])
                            simplex[i] = np.clip(simplex[i], lo, hi)
                            fvals[i] = obj(simplex[i]); used += 1
                        if deadline():
                            break
                if np.max(np.abs(np.array(fvals) - fvals[0])) < tol:
                    break
            order = np.argsort(fvals)
            return simplex[order[0]], fvals[order[0]]

        nd = nondominated_idx(F)
        # refine the top non-dominated points (by sum) within budget & time
        order_nd = nd[np.argsort(np.sum(F[nd], axis=1))]
        refined = []
        for idx in order_nd:
            if used >= self.budget or deadline():
                break
            x_best, _ = nelder_mead(X[idx], F[idx])
            if used >= self.budget:
                break
            refined.append(x_best)

        # ---- Phase 3: gather all points, evaluate refined, final non-dominated ----
        allX = list(X) + refined
        allF = list(F) + [np.asarray(problem(x), dtype=float) for x in refined]
        used += len(refined)
        finalF = np.array(allF)
        nd2 = nondominated_idx(finalF)
        return allX[nd2]
# EVOLVE-BLOCK-END
