"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-objective optimizer: LHS init, HV-guided selection, adaptive mutation."""

    def __init__(self, budget, dim, n_obj):
        self.budget, self.dim, self.n_obj = budget, dim, n_obj

    # ---- exact 3D hypervolume (minimization, ref point) ----
    def _hv3(self, P, ref):
        if len(P) == 0:
            return 0.0
        P = P[P[:, 0] < ref[0]]
        P = P[P[:, 1] < ref[1]]
        P = P[P[:, 2] < ref[2]]
        if len(P) == 0:
            return 0.0
        P = np.unique(P, axis=0)
        if len(P) == 0:
            return 0.0
        P = P[np.lexsort((P[:, 0], P[:, 1]))]
        x0, y0, z0 = P[:, 0], P[:, 1], P[:, 2]
        hv = 0.0
        prev_y = ref[1]
        for i in range(len(P)):
            y = y0[i]
            if y >= prev_y:
                continue
            band = prev_y - y
            x = x0[i]
            prev_z = ref[2]
            for j in range(i, len(P)):
                if x0[j] >= x:
                    break
                z = z0[j]
                if z >= prev_z:
                    continue
                hv += (x - x0[j]) * (prev_z - z) * band
                prev_z = z
            prev_y = y
        return hv

    def __call__(self, problem, bounds):
        t0 = time.time()
        deadline = 320.0  # safety margin before 330s kill
        rng = np.random.default_rng(0)
        lo = np.array([b[0] for b in bounds], dtype=float)
        hi = np.array([b[1] for b in bounds], dtype=float)
        span = hi - lo
        d, nobj, budget = self.dim, self.n_obj, self.budget
        ref = np.array([1705.0, 12.0, 0.3])

        used = 0

        # ---- LHS initialization ----
        pop = int(min(24, max(8, budget // 8)))
        X = np.empty((pop, d))
        for j in range(d):
            perm = rng.permutation(pop)
            X[np.arange(pop), j] = lo[j] + (hi[j] - lo[j]) * ((perm + 0.5) / pop)
        F = np.empty((pop, nobj))
        for i in range(pop):
            F[i] = np.asarray(problem(X[i]), dtype=float)
            used += 1
            if used >= budget:
                return X

        def is_dominated(f, Fm):
            return np.any(np.all(Fm <= f, axis=1) & np.any(Fm < f, axis=1))

        def update(i, x, f):
            nonlocal used
            X[i] = x
            F[i] = f
            used += 1

        # ---- main loop ----
        while used < budget:
            if time.time() > deadline:
                break

            # non-dominated set
            nd_idx = [i for i in range(pop) if not is_dominated(F[i], F)]
            if not nd_idx:
                nd_idx = list(range(pop))

            # HV contribution of each non-dominated point
            hv_contrib = {}
            for i in nd_idx:
                rest = [F[k] for k in nd_idx if k != i]
                hv_contrib[i] = self._hv3(np.array([F[i]] + rest), ref) - self._hv3(np.array(rest), ref)

            # parent selection: prefer high HV contribution
            if len(nd_idx) == 1:
                pi = nd_idx[0]
            else:
                weights = np.array([max(hv_contrib[i], 1e-9) for i in nd_idx])
                weights = weights / weights.sum()
                pi = nd_idx[rng.choice(len(nd_idx), p=weights)]

            # adaptive step: shrink with progress
            progress = used / max(1, budget)
            sigma = span * (0.25 * (1.0 - 0.7 * progress) + 0.03)
            child = X[pi] + sigma * rng.standard_normal(d)
            child = np.clip(child, lo, hi)
            fc = np.asarray(problem(child), dtype=float)
            used += 1

            # replace worst HV contributor (or worst sum if none)
            if hv_contrib:
                worst = min(hv_contrib, key=hv_contrib.get)
            else:
                worst = int(np.argmax(np.sum(F, axis=1)))
            update(worst, child, fc)

        return X
# EVOLVE-BLOCK-END
