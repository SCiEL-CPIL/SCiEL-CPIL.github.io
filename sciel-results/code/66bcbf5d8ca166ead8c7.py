# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """DE + NSGA-II style multi-objective optimizer with deadline guard."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        t_start = time.time()
        deadline = t_start + 280.0  # safety margin before 330s kill

        rng = np.random.default_rng(12345)
        lo = np.array([b[0] for b in bounds], dtype=float)
        hi = np.array([b[1] for b in bounds], dtype=float)
        dim = self.dim
        n_obj = self.n_obj
        ref = np.array([1705.0, 12.0, 0.3], dtype=float)

        def _valid_mask(F):
            return np.all(F < ref, axis=1)

        def _non_dominated_mask(F):
            """Vectorized: returns bool mask of non-dominated points in F (n x m)."""
            n = F.shape[0]
            if n == 0:
                return np.zeros(0, dtype=bool)
            # A dominates B if A<=B in all and A<B in at least one
            # For each i, check if any j dominates i
            # diff[j,i] = F[j] - F[i]; dominates if all<=0 and any<0
            # We can compute: for each i, is there j != i with F[j]<=F[i] and F[j]<F[i]
            # Vectorized over i using broadcasting is O(n^2*m) memory; n is small (~100)
            mask = np.ones(n, dtype=bool)
            # For each i, count dominators
            # Use: F[j] <= F[i] for all dims, and F[j] < F[i] for at least one dim
            for i in range(n):
                diff = F - F[i]  # (n, m)
                le = np.all(diff <= 0, axis=1)
                lt = np.any(diff < 0, axis=1)
                dom = le & lt
                dom[i] = False
                if np.any(dom):
                    mask[i] = False
            return mask

        def _hv_3d(F, ref):
            """Exact 3D hypervolume. F: (n,3) with all points strictly < ref.
            Returns float. O(n^2 log n) via sweep."""
            n = F.shape[0]
            if n == 0:
                return 0.0
            # Filter strictly better
            m = np.all(F < ref, axis=1)
            F = F[m]
            n = F.shape[0]
            if n == 0:
                return 0.0
            if n == 1:
                f = F[0]
                if np.any(f >= ref):
                    return 0.0
                return float((ref[0]-f[0]) * (ref[1]-f[1]) * (ref[2]-f[2]))
            # Sort by dim0 ascending
            order = np.argsort(F[:, 0], kind='mergesort')
            F = F[order]
            # Sweep: for each i, the unique region contributed is
            # x in [F[i,0], F[i+1,0]) (or ref[0] for last)
            # and the 2D hypervolume of points i..n-1 in dims 1,2 with ref (ref1, ref2)
            # But we must only count the region NOT already covered by earlier points.
            # Standard: sort by x0. For i from 0..n-1:
            #   x0_lo = F[i,0], x0_hi = F[i+1,0] if i+1<n else ref[0]
            #   if x0_hi <= x0_lo: continue
            #   compute 2D HV of points [i..n-1] projected to (x1,x2) with ref (ref1,ref2)
            #   but only the portion with x1 >= F[i,1]?? No.
            # Actually the correct 3D HV via sweep:
            # The union of boxes [F[i], ref]. Sort by x0.
            # For each i, the box [F[i], ref] contributes the part where x0 in [F[i,0], F[i+1,0])
            #   and (x1,x2) in the 2D region dominated by F[i] but not by any earlier box.
            # This is complex. Instead use the standard O(n^2) for 3D:
            # HV = sum over i of (ref0 - F[i,0]) * HV_2D({(F[j,1],F[j,2]) for j: F[j,0]>=F[i,0] and F[j,1]>=F[i,1] and F[j,2]>=F[i,2]} ... )
            # That's not right either.
            #
            # Let me use the clean recursive / inclusion approach for 3D which is O(n^2):
            # Sort by x0. Maintain a list of "active" 2D points.
            # Actually, simplest correct O(n^2) 3D HV:
            #   Sort by x0 ascending.
            #   hv = 0
            #   For i in range(n):
            #       x0_lo = F[i,0]
            #       x0_hi = F[i+1,0] if i+1 < n else ref[0]
            #       if x0_hi <= x0_lo: continue
            #       # 2D HV of the set of points j in [i..n-1] with (x1,x2), ref (ref1,ref2)
            #       # BUT we only want the 2D region that is NOT covered by points with smaller x0.
            #       # Since all points in [i..n-1] have x0 >= F[i,0], and earlier points have x0 < F[i,0],
            #       # the earlier boxes cover x0 in [F[k,0], F[k+1,0]) for k < i, which is disjoint from
            #       # [F[i,0], F[i+1,0]). So within the slab x0 in [F[i,0], F[i+1,0]),
            #       # the covered region in (x1,x2) is exactly the union of boxes [F[j,1:], ref[1:]]
            #       # for j in [i..n-1]. So we need the 2D HV of points [i..n-1] in dims 1,2.
            #       pts_2d = F[i:, 1:3]
            #       hv_2d = _hv_2d(pts_2d, ref[1:3])
            #       hv += (x0_hi - x0_lo) * hv_2d
            # This is O(n^2 * cost_2d). With n~100 and 2d O(m log m), total O(n^2 log n). Fine.
            def _hv_2d(P, ref2):
                # P: (m,2), all < ref2. Return 2D HV (area of union of boxes [p, ref2]).
                m = P.shape[0]
                if m == 0:
                    return 0.0
                # Sort by x1 ascending
                order = np.argsort(P[:, 0], kind='mergesort')
                P = P[order]
                # 2D HV: sweep by x1. For each i, x1 in [P[i,0], P[i+1,0] or ref2[0])
                #   covered x2 is [min of P[j,1] for j>=i, ref2[1])  -> but only if that min < ref2[1]
                # Actually: the union of boxes [P[i], ref2]. Sort by x1.
                # For slab x1 in [P[i,0], P[i+1,0]), the covered x2 range is
                # [min_{j>=i} P[j,1], ref2[1]) intersected with [0,ref2[1]) but we need the union.
                # The union of intervals [P[j,1], ref2[1]) for j>=i is [min_{j>=i} P[j,1], ref2[1])
                # because all intervals share the right endpoint ref2[1]. So covered length = ref2[1] - min_{j>=i} P[j,1] (if positive).
                area = 0.0
                # Compute suffix minimum of P[:,1]
                suf_min = np.minimum.accumulate(P[::-1, 1])[::-1]
                for i in range(m):
                    x1_lo = P[i, 0]
                    x1_hi = P[i+1, 0] if i+1 < m else ref2[0]
                    if x1_hi <= x1_lo:
                        continue
                    min_x2 = suf_min[i]
                    if min_x2 >= ref2[1]:
                        continue
                    area += (x1_hi - x1_lo) * (ref2[1] - min_x2)
                return area
            hv = 0.0
            # suffix structure not needed; just loop
            for i in range(n):
                x0_lo = F[i, 0]
                x0_hi = F[i+1, 0] if i+1 < n else ref[0]
                if x0_hi <= x0_lo:
                    continue
                pts_2d = F[i:, 1:3]
                # filter pts strictly < ref[1:3]
                m2 = np.all(pts_2d < ref[1:3], axis=1)
                pts_2d = pts_2d[m2]
                hv_2d = _hv_2d(pts_2d, ref[1:3])
                hv += (x0_hi - x0_lo) * hv_2d
            return float(hv)

        def _crowding_distance(F_nd, X_nd):
            """Crowding distance for a non-dominated front. Returns array of distances."""
            n = F_nd.shape[0]
            if n <= 2:
                return np.full(n, np.inf)
            dist = np.zeros(n)
            for k in range(n_obj):
            # sort by objective k
                order = np.argsort(F_nd[:, k], kind='mergesort')
                dist[order[0]] = np.inf
                dist[order[-1]] = np.inf
                f_range = F_nd[order[-1], k] - F_nd[order[0], k]
                if f_range == 0:
                    continue
                for i in range(1, n-1):
                    dist[order[i]] += (F_nd[order[i+1], k] - F_nd[order[i-1], k]) / f_range
            return dist

        # ---- Initial population ----
        # Use a mix: Latin hypercube-ish random + a few corner points
        n_init = 80
        # corners (2^5=32)
        corners = np.array(np.meshgrid(*[np.array([lo[i], hi[i]]) for i in range(dim)])).T.reshape(-1, dim)
        n_c = min(32, n_init)
        n_r = n_init - n_c
        random_pts = lo + (hi - lo) * rng.random((n_r, dim))
        X = np.vstack([corners[:n_c], random_pts])
        F = np.array([problem(x) for x in X])
        used = len(X)

        # DE parameters
        CR = 0.8
        F_DE = 0.85
        pop_size = 60  # candidate pool size per generation

        # Keep a growing archive of all evaluated points
        all_X = [X.copy()]
        all_F = [F.copy()]

        def _combine_XF():
            Xc = np.vstack(all_X)
            Fc = np.vstack(all_F)
            return Xc, Fc

        # Main loop: generations
        while used < self.budget:
            if time.time() > deadline:
                break

            Xc, Fc = _combine_XF()
            n = Xc.shape[0]

            # Non-dominated mask on the whole set
            nd_mask = _non_dominated_mask(Fc)
            nd_idx = np.where(nd_mask)[0]

            if len(nd_idx) < 3:
                # fallback: pick random parents
                r1 = rng.integers(n)
                r2 = rng.integers(n)
                r3 = rng.integers(n)
                while r2 == r1:
                    r2 = rng.integers(n)
                while r3 == r1 or r3 == r2:
                    r3 = rng.integers(n)
            else:
                # Weighted selection from non-dominated set by HV contribution
                # Compute individual HV contribution (box volume) as a quick weight
                nd_F = Fc[nd_idx]
                # box volume
                vol = np.prod(np.maximum(ref - nd_F, 0), axis=1)
                vol = np.maximum(vol, 1e-12)
                w = vol / vol.sum()
                # sample 3 distinct
                picks = rng.choice(len(nd_idx), size=3, replace=False, p=w)
                r1 = nd_idx[picks[0]]
                r2 = nd_idx[picks[1]]
                r3 = nd_idx[picks[2]]

            # DE mutation
            v = Xc[r1] + F_DE * (Xc[r2] - Xc[r3])
            # binomial crossover
            j_rand = rng.integers(dim)
            trial = np.empty(dim)
            mask = (rng.random(dim) < CR) | (np.arange(dim) == j_rand)
            trial[mask] = v[mask]
            trial[~mask] = Xc[r1][~mask]
            trial = np.clip(trial, lo, hi)

            fc = problem(trial)
            used += 1

            # Add to archive
            all_X.append(trial.reshape(1, -1))
            all_F.append(fc.reshape(1, -1))

            # Periodic pruning: if archive gets too large, keep a diverse subset
            # to keep _non_dominated_mask fast. Threshold ~ 400.
            if len(all_X) > 400:
                Xc, Fc = _combine_XF()
                nd_mask = _non_dominated_mask(Fc)
                nd_idx = np.where(nd_mask)[0]
                if len(nd_idx) > 200:
                    # crowding distance on the front, keep top 200
                    cd = _crowding_distance(Fc[nd_idx], Xc[nd_idx])
                    keep_local = np.argsort(cd)[::-1][:200]
                    keep_global = nd_idx[keep_local]
                    # Also keep a few dominated points for diversity? Keep only front + random
                    # Build new archive: kept front points + 20 random dominated
                    new_X = [Xc[i].reshape(1, -1) for i in keep_global]
                    new_F = [Fc[i].reshape(1, -1) for i in keep_global]
                    # add some random from the rest
                    rest = np.setdiff1d(np.arange(len(Xc)), keep_global)
                    if len(rest) > 0:
                        add = rng.choice(rest, size=min(20, len(rest)), replace=False)
                        for i in add:
                            new_X.append(Xc[i].reshape(1, -1))
                            new_F.append(Fc[i].reshape(1, -1))
                    all_X = new_X
                    all_F = new_F

        # Return the full set of evaluated points (X). The evaluator records objective vectors
        # from problem calls, so we just need to return something; return combined X.
        Xc, Fc = _combine_XF()
        return Xc
# EVOLVE-BLOCK-END
