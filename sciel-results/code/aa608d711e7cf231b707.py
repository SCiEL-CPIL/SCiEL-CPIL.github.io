# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


class Optimizer:
    """DE-based multi-objective search with deadline guard."""

    def __init__(self, budget, dim, n_obj):
        self.budget, self.dim, self.n_obj = budget, dim, n_obj

    def __call__(self, problem, bounds):
        t_start = time.time()
        deadline = t_start + 290.0  # safety margin before 330s kill

        rng = np.random.default_rng(0)
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        dim = self.dim
        n_obj = self.n_obj
        ref = np.array([1705.0, 12.0, 0.3])

        def _non_dominated_mask(F):
            """Returns bool mask of non-dominated points in F (n x m)."""
            n = F.shape[0]
            if n == 0:
                return np.zeros(0, dtype=bool)
            mask = np.ones(n, dtype=bool)
            for i in range(n):
                diff = F - F[i]
                le = np.all(diff <= 0, axis=1)
                lt = np.any(diff < 0, axis=1)
                dom = le & lt
                dom[i] = False
                if np.any(dom):
                    mask[i] = False
            return mask

        def _hv_3d(F, ref_pt):
            """Exact 3D hypervolume. F: (n,3) with all points strictly < ref_pt."""
            n = F.shape[0]
            if n == 0:
                return 0.0
            m = np.all(F < ref_pt, axis=1)
            F = F[m]
            n = F.shape[0]
            if n == 0:
                return 0.0
            if n == 1:
                f = F[0]
                return float((ref_pt[0]-f[0]) * (ref_pt[1]-f[1]) * (ref_pt[2]-f[2]))
            order = np.argsort(F[:, 0], kind='mergesort')
            F = F[order]
            
            def _hv_2d(P, ref2):
                m = P.shape[0]
                if m == 0:
                    return 0.0
                order = np.argsort(P[:, 0], kind='mergesort')
                P = P[order]
                area = 0.0
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
            for i in range(n):
                x0_lo = F[i, 0]
                x0_hi = F[i+1, 0] if i+1 < n else ref_pt[0]
                if x0_hi <= x0_lo:
                    continue
                pts_2d = F[i:, 1:3]
                m2 = np.all(pts_2d < ref_pt[1:3], axis=1)
                pts_2d = pts_2d[m2]
                hv_2d = _hv_2d(pts_2d, ref_pt[1:3])
                hv += (x0_hi - x0_lo) * hv_2d
            return float(hv)

        def _hv_contribution_3d(F, ref_pt, idx):
            """Compute the hypervolume contribution of point idx in set F."""
            F_full = F
            F_without = np.delete(F_full, idx, axis=0)
            hv_full = _hv_3d(F_full, ref_pt)
            hv_without = _hv_3d(F_without, ref_pt)
            return hv_full - hv_without

        # ---- Initial population ----
        n_init = 80
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

        all_X = [X.copy()]
        all_F = [F.copy()]

        def _combine_XF():
            Xc = np.vstack(all_X)
            Fc = np.vstack(all_F)
            return Xc, Fc

        # Main loop
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
                # For speed, we approximate HV contribution by the volume of the box
                # that would be added if the point were removed.
                # A faster approximation: use the volume of the box defined by the point
                # and the reference point, but adjusted for overlap.
                # For now, use simple box volume as a proxy for weight.
                nd_F = Fc[nd_idx]
                vol = np.prod(np.maximum(ref - nd_F, 0), axis=1)
                vol = np.maximum(vol, 1e-12)
                w = vol / vol.sum()
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
            if len(all_X) > 400:
                Xc, Fc = _combine_XF()
                nd_mask = _non_dominated_mask(Fc)
                nd_idx = np.where(nd_mask)[0]
                if len(nd_idx) > 200:
                    # Keep top 200 by crowding distance or just random subset of front
                    # For speed, just keep the first 200 non-dominated points
                    keep_global = nd_idx[:200]
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
