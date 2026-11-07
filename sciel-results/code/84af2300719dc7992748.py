import numpy as np
import time


class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 285.0  # safety margin before 300s kill
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        span = hi - lo
        rng = np.random.default_rng(0)

        evals_used = 0
        max_evals = self.budget

        def evaluate(x):
            nonlocal evals_used
            if evals_used >= max_evals:
                return None
            x_clipped = np.clip(x, lo, hi)
            f = np.array(problem(x_clipped), dtype=float)
            evals_used += 1
            return x_clipped, f

        def dominated_by(f, others):
            """True if any point in others strictly dominates f."""
            for af in others:
                if np.all(af <= f) and np.any(af < f):
                    return True
            return False

        def nondominated(front):
            """Return indices of non-dominated points in list front (each len n_obj)."""
            n = len(front)
            keep = [True] * n
            for i in range(n):
                if not keep[i]:
                    continue
                fi = front[i]
                for j in range(n):
                    if i == j or not keep[j]:
                        continue
                    fj = front[j]
                    if np.all(fj <= fi) and np.any(fj < fi):
                        keep[i] = False
                        break
            return [i for i in range(n) if keep[i]]

        # ---------------- Phase 1: evolutionary population (structural core) ----
        pop = int(min(20, max(5, self.budget // 10)))
        X = lo + span * rng.random((pop, self.dim))
        F = np.empty((pop, self.n_obj))
        for i in range(pop):
            res = evaluate(X[i])
            if res is None:
                break
            X[i] = res[0]
            F[i] = res[1]
        alive = min(pop, evals_used)

        while evals_used < max_evals and time.time() < deadline:
            # Non-dominated set of current population
            nd_idx = nondominated([F[i] for i in range(alive)])
            if not nd_idx:
                nd_idx = list(range(alive))
            parent = X[nd_idx[rng.integers(len(nd_idx))]]
            # Adaptive mutation: shrink as search progresses (less disruption near convergence)
            progress = evals_used / max(1, max_evals)
            scale = 0.15 * (1.0 - 0.6 * progress)
            child = parent + scale * span * rng.standard_normal(self.dim)
            child = np.clip(child, lo, hi)
            res = evaluate(child)
            if res is None:
                break
            x_new, f_new = res
            # Replace the worst member (largest sum of objectives)
            worst = int(np.argmax(np.sum(F[:alive], axis=1)))
            X[worst] = x_new
            F[worst] = f_new

        # ---------------- Phase 2: extract non-dominated front ------------------
        front_idx = nondominated([F[i] for i in range(alive)])
        if not front_idx:
            front_idx = [int(np.argmin(np.sum(F[:alive], axis=1)))]
        front = [(X[i].copy(), F[i].copy()) for i in front_idx]

        # ---------------- Phase 3: adaptive multi-objective hill polish --------
        reserved = 30
        def polish(x0, f0):
            """Multi-objective hill climb: accept strictly-better OR non-dominated moves."""
            nonlocal evals_used
            cx = x0.copy()
            cf = f0.copy()
            step = 0.12
            for _ in range(60):
                if evals_used >= max_evals - reserved:
                    break
                if time.time() > deadline:
                    break
                improved = False
                for _ in range(6):
                    if evals_used >= max_evals - reserved:
                        break
                    if time.time() > deadline:
                        break
                    dx = rng.standard_normal(self.dim) * step
                    xn = np.clip(cx + dx, lo, hi)
                    res = evaluate(xn)
                    if res is None:
                        break
                    xn, fn = res
                    # Accept if it dominates current (strict) OR is non-dominated vs current
                    dominates = np.all(fn <= cf) and np.any(fn < cf)
                    non_dom = (not (np.all(cf <= fn) and np.any(cf < fn)))
                    if dominates or non_dom:
                        cx, cf = xn, fn
                        improved = True
                        step *= 0.85
                        break
                if not improved:
                    step *= 0.6
                    if step < 0.01:
                        break
            return cx, cf

        for x0, f0 in front:
            if evals_used >= max_evals - reserved:
                break
            if time.time() > deadline:
                break
            cx, cf = polish(x0, f0)
            front.append((cx, cf))

        # ---------------- Phase 4: micro-refinement pass -----------------------
        # Re-extract front and do one more short polish on the current best per region
        front2_idx = nondominated([f for _, f in front])
        if not front2_idx:
            front2_idx = [0]
        for idx in front2_idx:
            if evals_used >= max_evals:
                break
            if time.time() > deadline:
                break
            x0, f0 = front[idx]
            cx, cf = polish(x0, f0)
            front.append((cx, cf))

        # Final non-dominated extraction for return
        final_idx = nondominated([f for _, f in front])
        if not final_idx:
            final_idx = [0]
        return [front[i][0] for i in final_idx]
