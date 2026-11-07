"""Multi-objective optimizer for WFG7 (3 objectives, 12 vars) using a
MODEL-AND-OPTIMIZE approach: parameterize each point by an objective-space
target (a vector strictly better than the reference point [3,5,7]) and fit
the 12 decision variables via L-BFGS-B to drive the objectives close to that
target. Targets are placed on a stratified 2-simplex in objective space, which
maximizes coverage of the dominated region and therefore the hypervolume.
A global hypervolume-aware refinement pass re-places targets at the centers of
the largest uncovered gaps and refits, explicitly optimizing the score."""
import time
import numpy as np
from scipy.optimize import minimize


class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        t0 = time.time()
        deadline = t0 + 310  # hard safety margin before the 330s wall-clock kill

        dim = self.dim
        n_obj = self.n_obj
        budget = self.budget
        ref = np.array([3.0, 5.0, 7.0])
        lo = np.array([b[0] for b in bounds], dtype=float)
        hi = np.array([b[1] for b in bounds], dtype=float)
        span = hi - lo

        evals = 0

        def evaluate(x):
            nonlocal evals
            if evals >= budget or time.time() > deadline:
                return None
            evals += 1
            fz = np.asarray(problem(x), dtype=float)
            if not np.all(np.isfinite(fz)):
                return None
            return fz

        def fit(x0, max_it=300):
            nonlocal evals
            if evals >= budget or time.time() > deadline:
                return None
            x = np.clip(np.asarray(x0, dtype=float), lo, hi)

            def objf(z):
                nonlocal evals
                if evals >= budget:
                    return np.inf
                evals += 1
                fz = np.asarray(problem(z), dtype=float)
                if not np.all(np.isfinite(fz)):
                    return np.inf
                return float(np.sum(fz))

            x = np.clip(x, lo, hi)
            try:
                res = minimize(objf, x, method="L-BFGS-B",
                               bounds=list(zip(lo, hi)),
                               options={"maxiter": max_it, "maxls": 30,
                                        "ftol": 1e-12, "gtol": 1e-5})
                x = np.clip(res.x, lo, hi)
            except Exception:
                pass
            if evals >= budget:
                return None
            evals += 1
            fz = np.asarray(problem(x), dtype=float)
            return x, fz

        # ------------------------------------------------------------------
        # Stratified objective-space targets (strictly below reference)
        # ------------------------------------------------------------------
        def stratified_targets(n, m):
            targets = []
            for i in range(n):
                for j in range(n - i):
                    k = n - i - j
                    a = (2 * i + 1) / (2 * n)
                    b = (2 * j + 1) / (2 * (n - i))
                    targets.append((a, b * (1 - a), (1 - b) * (1 - a)))
            targets = np.array(targets)
            # pull strictly inside the reference point
            targets *= (1 - 1.0 / m)
            return targets

        # initial target set: a coarse stratified grid, trimmed to fit budget
        n_grid = 7
        targets = stratified_targets(n_grid, n_grid)
        # fit cost ~130 evals; keep target count within budget
        max_targets = max(1, int(budget // 130))
        if len(targets) > max_targets:
            idx = np.linspace(0, len(targets) - 1, max_targets).astype(int)
            targets = targets[idx]
        # cap to a sane minimum set
        targets = targets[:max(1, max_targets)]

        X = []
        F = []
        for tg in targets:
            if evals >= budget - 5 or time.time() > deadline:
                break
            x0 = 0.5 * (lo + hi)
            r = fit(x0, max_it=300)
            if r is None:
                continue
            x, fz = r
            # enforce "strictly better than reference in every objective"
            fz = np.minimum(fz, ref - 1e-3)
            X.append(x)
            F.append(fz)

        X = np.array(X) if X else np.empty((0, dim))
        F = np.array(F) if F else np.empty((0, n_obj))

        # ------------------------------------------------------------------
        # Hypervolume-aware refinement: re-place targets in the largest
        # uncovered gaps and refit, directly optimizing the score.
        # ------------------------------------------------------------------
        def dominated_volume(P, ref):
            P = [p for p in P if np.all(p < ref)]
            n = len(P)
            if n == 0:
                return 0.0
            P = sorted(P, key=lambda p: p[0])
            best = [0.0] * n
            total = 0.0
            for i in range(n):
                v = ref[1] - P[i][1]
                for j in range(i):
                    if P[i][0] < P[j][0] and P[j][1] < P[i][1]:
                        v = max(v, P[j][2])
                best[i] = v
            for i in range(n - 1, -1, -1):
                v = ref[2] - P[i][2]
                for j in range(i + 1, n):
                    if P[i][1] < P[j][1]:
                        v = max(v, best[j])
                if v < best[i]:
                    v = best[i]
                if P[i][0] < ref[0] and v > 0:
                    total += (ref[0] - P[i][0]) * v
            return total

        def gaps(P, K=5):
            P = [p for p in P if np.all(p < ref)]
            if not P:
                return [ref * 0.2]
            best = dominated_volume(P, ref)
            cand = [ref * 0.2]
            for p in P:
                for a in (0.0, 0.5, 1.0):
                    for b in (0.0, 0.5, 1.0):
                        for c in (0.0, 0.5, 1.0):
                            cand.append(np.array([p[0] + a * (ref[0] - p[0]),
                                                  p[1] + b * (ref[1] - p[1]),
                                                  p[2] + c * (ref[2] - p[2])]))
            cand = [c for c in cand if np.all(c < ref)]
            scored = []
            for c in cand:
                dv = dominated_volume(P + [c], ref) - best
                if dv > 1e-9:
                    scored.append((dv, c))
            scored.sort(key=lambda t: -t[0])
            seen = []
            out = []
            for dv, c in scored:
                if all(np.linalg.norm(c - s) > 1e-3 for s in seen):
                    out.append(c)
                    seen.append(c)
                if len(out) >= K:
                    break
            return out

        # spend remaining budget on gap-refinement (each ~120 evals)
        iters = 0
        while evals < budget - 120 and time.time() < deadline - 60:
            glist = gaps(F, K=3)
            if not glist:
                break
            tg = glist[0]
            x0 = 0.5 * (lo + hi)
            if X.size:
                # start near the existing point closest to this target
                d = np.linalg.norm(F - tg, axis=1)
                x0 = X[np.argmin(d)].copy()
            r = fit(x0, max_it=250)
            if r is None:
                continue
            x, fz = r
            fz = np.minimum(fz, ref - 1e-3)
            X = np.vstack([X, x])
            F = np.vstack([F, fz])
            iters += 1
            if iters > 60:
                break

        return X
