"""Multi-objective optimizer for RE34: hypervolume-guided search with deadline guard."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Hypervolume-guided multi-objective search with adaptive perturbation."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        t0 = time.time()
        deadline = t0 + 300.0          # evaluator kills at 330s; keep a 30s margin
        budget = int(self.budget)
        dim = self.dim
        n_obj = self.n_obj

        lo = np.array([b[0] for b in bounds], dtype=float)
        hi = np.array([b[1] for b in bounds], dtype=float)
        rng = np.random.default_rng(12345)

        # Reference point for hypervolume (larger volume == better point).
        ref = np.array([1705.0, 12.0, 0.3], dtype=float)

        # ---------- helpers ----------
        def hv_of(points):
            """Exact hypervolume of a list of points (minimization, ref)."""
            if not points:
                return 0.0
            P = np.asarray(points, dtype=float)
            # keep only points strictly better than ref in every objective
            mask = np.all(P < ref, axis=1)
            P = P[mask]
            if len(P) == 0:
                return 0.0
            # hypervolume contribution via sorting on first objective
            order = np.argsort(P[:, 0])
            P = P[order]
            total = 0.0
            prev = -np.inf
            for i in range(len(P)):
                x = P[i, 0]
                if x >= ref[0]:
                    break
                # width along obj0 from prev to x (clipped)
                hi0 = min(x, ref[0])
                w0 = hi0 - prev
                if w0 <= 0:
                    continue
                # cross-section volume of points with index >= i (obj0 >= x)
                sub = P[i:, 1:]
                sub = sub[np.all(sub < ref[1:], axis=1)]
                if len(sub) == 0:
                    continue
                # 2D hypervolume of sub w.r.t. ref[1:]
                total += w0 * hv2(sub, ref[1:])
                prev = x
            return total

        def hv2(sub, ref2):
            """2D hypervolume (minimization)."""
            # sub: (n,2), ref2: (2,)
            if len(sub) == 0:
                return 0.0
            order = np.argsort(sub[:, 0])
            sub = sub[order]
            total = 0.0
            prev = -np.inf
            for i in range(len(sub)):
                x = sub[i, 0]
                if x >= ref2[0]:
                    break
                hi0 = min(x, ref2[0])
                w0 = hi0 - prev
                if w0 <= 0:
                    continue
                # max (best) y among points with index >= i
                ybest = np.max(sub[i:, 1])
                ybest = min(ybest, ref2[1])
                w1 = ybest - prev_y if False else 0.0
                # correct: height along y from -inf is ref2[1] - ybest
                total += w0 * (ref2[1] - ybest)
                prev = x
            return total

        def is_dominated(f, all_f):
            """True if any other point strictly dominates f."""
            # all_f includes f itself; exclude exact self by index not needed since we compare vectors
            better = np.all(all_f <= f, axis=1) & np.any(all_f < f, axis=1)
            return bool(np.any(better))

        def dominated_mask(F):
            """Boolean mask: True where point is dominated by another."""
            n = len(F)
            mask = np.zeros(n, dtype=bool)
            for i in range(n):
                fi = F[i]
                better = np.all(F <= fi, axis=1) & np.any(F < fi, axis=1)
                better[i] = False
                if np.any(better):
                    mask[i] = True
            return mask

        def pareto(F):
            """Return indices of non-dominated points."""
            n = len(F)
            keep = []
            for i in range(n):
                fi = F[i]
                better = np.all(F <= fi, axis=1) & np.any(F < fi, axis=1)
                better[i] = False
                if not np.any(better):
                    keep.append(i)
            return keep

        # ---------- main loop ----------
        # Start with a random spread.
        X = []
        F = []

        def add(x, f):
            X.append(np.asarray(x, dtype=float))
            F.append(np.asarray(f, dtype=float))

        # seed points
        for _ in range(8):
            x = lo + (hi - lo) * rng.random(dim)
            add(x, problem(x))

        used = len(F)
        # adaptive perturbation scale
        scale = 0.25

        while used < budget:
            # deadline guard
            if time.time() > deadline - 5.0:
                break

            # choose parents: prefer non-dominated
            keep = pareto(F)
            if len(keep) == 0:
                keep = list(range(len(F)))
            F_arr = np.array(F)

            # select parents weighted toward low objective sum among non-dominated
            cand = [F_arr[i] for i in keep]
            sums = np.sum(cand, axis=1)
            # softmax-ish selection (lower sum = higher weight)
            w = np.exp(-1.5 * (sums - sums.min()) / (sums.max() - sums.min() + 1e-9))
            w /= w.sum()
            pidx = keep[int(rng.choice(len(keep), p=w))]

            # mutation: adaptive step, occasionally larger
            if rng.random() < 0.15:
                step = scale * 2.0
            else:
                step = scale
            delta = step * (hi - lo) * rng.standard_normal(dim)
            child = X[pidx] + delta
            child = np.clip(child, lo, hi)
            fc = problem(child)
            add(child, fc)
            used += 1

            # update scale: shrink slowly to refine, but keep some exploration
            scale = max(0.05, scale * 0.99)

        return X
# EVOLVE-BLOCK-END
