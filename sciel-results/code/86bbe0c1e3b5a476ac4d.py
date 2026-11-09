import numpy as np
import time

class Optimizer:
    """Multi-objective optimizer for RE34: hypervolume-guided search with deadline guard."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        t0 = time.time()
        deadline = t0 + 280.0  # safety margin before 300s kill
        budget = int(self.budget)
        dim = self.dim
        n_obj = self.n_obj

        lo = np.array([b[0] for b in bounds], dtype=float)
        hi = np.array([b[1] for b in bounds], dtype=float)
        rng = np.random.default_rng(42)

        # Reference point for hypervolume
        ref = np.array([1705.0, 12.0, 0.3], dtype=float)

        # Store ALL evaluated points - every evaluation counts
        X_all = []
        F_all = []

        def add_point(x, f):
            X_all.append(np.asarray(x, dtype=float))
            F_all.append(np.asarray(f, dtype=float))

        # ---------- Hypervolume helpers ----------
        def hv2(sub, ref2):
            """2D hypervolume (minimization)."""
            if len(sub) == 0:
                return 0.0
            sub = sub[np.all(sub < ref2, axis=1)]
            if len(sub) == 0:
                return 0.0
            order = np.argsort(sub[:, 0])
            sub = sub[order]
            total = 0.0
            prev_x = -np.inf
            for i in range(len(sub)):
                x = sub[i, 0]
                if x >= ref2[0]:
                    break
                w0 = min(x, ref2[0]) - prev_x
                if w0 <= 0:
                    continue
                ybest = np.min(sub[i:, 1])
                ybest = min(ybest, ref2[1])
                if ybest >= ref2[1]:
                    prev_x = x
                    continue
                total += w0 * (ref2[1] - ybest)
                prev_x = x
            return total

        def hv3(P, ref3):
            """3D hypervolume (minimization)."""
            if len(P) == 0:
                return 0.0
            mask = np.all(P < ref3, axis=1)
            P = P[mask]
            if len(P) == 0:
                return 0.0
            order = np.argsort(P[:, 0])
            P = P[order]
            total = 0.0
            prev_x = -np.inf
            for i in range(len(P)):
                x = P[i, 0]
                if x >= ref3[0]:
                    break
                w0 = min(x, ref3[0]) - prev_x
                if w0 <= 0:
                    continue
                sub = P[i:, 1:]
                total += w0 * hv2(sub, ref3[1:])
                prev_x = x
            return total

        def pareto_indices(F):
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

        # ---------- Initial structured sampling ----------
        # Start with the known good corner [1,1,1,1,1]
        x_corner = lo.copy()
        add_point(x_corner, problem(x_corner))

        # Add a structured spread along the likely Pareto front
        # RE34: f1 dominated by x1, so vary x1 while keeping others at lower bound
        n_structured = 20
        for i in range(n_structured):
            if len(X_all) >= budget:
                break
            if time.time() > deadline - 5.0:
                break
            x1 = lo[0] + (hi[0] - lo[0]) * i / max(n_structured - 1, 1)
            x = np.array([x1, lo[1], lo[2], lo[3], lo[4]])
            x = np.clip(x, lo, hi)
            add_point(x, problem(x))

        # Add a few random points for diversity
        n_random_init = 10
        for _ in range(n_random_init):
            if len(X_all) >= budget:
                break
            if time.time() > deadline - 5.0:
                break
            x = lo + (hi - lo) * rng.random(dim)
            add_point(x, problem(x))

        used = len(X_all)
        scale = 0.2
        last_hv = 0.0
        hv_stagnation = 0

        # ---------- Main search loop ----------
        while used < budget:
            if time.time() > deadline - 5.0:
                break

            # Compute current hypervolume (only for guidance, not returned)
            F_arr = np.array(F_all)
            current_hv = hv3(F_arr, ref)

            if current_hv > last_hv + 1e-6:
                last_hv = current_hv
                hv_stagnation = 0
            else:
                hv_stagnation += 1

            # Choose strategy based on stagnation
            if hv_stagnation > 50:
                if rng.random() < 0.3:
                    x = lo + (hi - lo) * rng.random(dim)
                    fc = problem(x)
                    add_point(x, fc)
                    used += 1
                    scale = max(0.3, scale * 1.5)
                    continue
                step_mult = 2.0
            elif hv_stagnation > 20:
                step_mult = 1.5
            else:
                step_mult = 1.0

            # Select parent from non-dominated set, weighted toward lower objective sum
            keep = pareto_indices(F_arr)
            if len(keep) == 0:
                keep = list(range(len(F_arr)))

            cand = F_arr[keep]
            sums = np.sum(cand, axis=1)
            range_sums = sums.max() - sums.min() + 1e-9
            w = np.exp(-2.0 * (sums - sums.min()) / range_sums)
            w /= w.sum()
            pidx = keep[int(rng.choice(len(keep), p=w))]

            # Adaptive mutation
            step = scale * step_mult
            delta = step * (hi - lo) * rng.standard_normal(dim)
            child = X_all[pidx] + delta
            child = np.clip(child, lo, hi)
            fc = problem(child)
            add_point(child, fc)
            used += 1

            # Slowly shrink step size
            scale = max(0.03, scale * 0.995)

        return X_all
