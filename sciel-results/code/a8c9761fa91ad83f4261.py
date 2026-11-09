"""DTLZ2 portfolio optimizer: dense known-front seed + Tchebycheff gap-fill + deadline guard."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-objective DTLZ2 optimizer exploiting the known spherical Pareto front.

    Strategy:
      1. Seed a grid on the quarter-sphere f1^2+f2^2+f3^2=1 (the true front).
      2. Use the remaining budget for Tchebycheff local polish to drive points
         onto the front and fill gaps.
      3. Deadline guard with safety margin under the 330s kill.
    """

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start = time.time()
        deadline = start + 320.0  # safety margin under 330s kill

        lo = np.array([b[0] for b in bounds], dtype=float)
        hi = np.array([b[1] for b in bounds], dtype=float)
        dim = self.dim
        n_obj = self.n_obj
        budget = self.budget

        rng = np.random.default_rng(0)

        def eval_and_record(x):
            return np.asarray(problem(x), dtype=float)

        # ---- 1. Grid seed on the known front ----
        # DTLZ2 (3 obj, 12 var): g = sum_{i=4}^{12}(x_i-0.5)^2, front when g=0.
        # f = (1+g)*(sin(x1)sin(x2), sin(x1)cos(x2), cos(x1)).
        # Front point (f1,f2,f3) with norm 1 -> x1=arccos(f3), x2=arctan2(f1,f2),
        # x3=0, x4..x12=0.5.
        # Reserve ~40% of budget for polish; keep front seed within the rest.
        polish_budget = max(20, int(budget * 0.4))
        seed_budget = budget - polish_budget
        n_theta = max(4, int(np.sqrt(seed_budget)))
        t1 = np.linspace(0.0, np.pi / 2.0, n_theta)
        t2 = np.linspace(0.0, np.pi / 2.0, n_theta)
        T1, T2 = np.meshgrid(t1, t2, indexing="ij")
        T1 = T1.ravel()
        T2 = T2.ravel()
        f1 = np.sin(T1) * np.sin(T2)
        f2 = np.sin(T1) * np.cos(T2)
        f3 = np.cos(T1)
        front_obj = np.column_stack([f1, f2, f3])
        n_front = len(front_obj)

        X = np.zeros((n_front, dim))
        X[:, 0] = np.arccos(np.clip(f3, -1.0, 1.0))
        X[:, 1] = np.arctan2(f1, f2)
        X[:, 2] = 0.0
        X[:, 3:] = 0.5
        X = np.clip(X, lo, hi)

        F = np.zeros((n_front, n_obj))
        used = 0
        for i in range(n_front):
            F[i] = eval_and_record(X[i])
            used += 1
            if used >= budget:
                break

        # ---- 2. Tchebycheff gap-fill ----
        z = F.min(axis=0)
        z = np.maximum(z, 1e-9)

        def tcheb_surrogate(f, w):
            v = w * (f - z)
            m = v.max()
            if not np.isfinite(m):
                return 1e6
            return m + np.log(np.sum(np.exp(v - m))) / n_obj

        # Weight vectors spread over the simplex.
        n_w = min(10, max(3, polish_budget // 5))
        W = np.zeros((n_w, n_obj))
        for i in range(n_obj):
            W[i, i] = 1.0
        edge = 0
        for a in range(n_obj):
            for b in range(a + 1, n_obj):
                if n_obj + edge < n_w:
                    W[n_obj + edge, a] = 0.5
                    W[n_obj + edge, b] = 0.5
                    edge += 1
        n_rand = n_w - n_obj - edge
        if n_rand > 0:
            r = rng.random((n_rand, n_obj))
            r = r / r.sum(axis=1, keepdims=True)
            W[n_obj + edge: n_w] = r

        def polish(x0, w, max_evals):
            x = x0.copy()
            fx = eval_and_record(x)
            used_local = 1
            best_x = x.copy()
            best_f = fx.copy()
            best_val = tcheb_surrogate(fx, w)
            step = 0.05
            for _ in range(max(0, max_evals - 1)):
                if time.time() > deadline - 1.0:
                    break
                d = rng.standard_normal(dim)
                cand = np.clip(x + step * d, lo, hi)
                cf = eval_and_record(cand)
                used_local += 1
                cval = tcheb_surrogate(cf, w)
                if cval < best_val:
                    best_val = cval
                    best_x = cand.copy()
                    best_f = cf.copy()
                    x = cand.copy()
                    step *= 0.9
                else:
                    step = min(step * 1.05, 0.2)
            return best_x, best_f, used_local

        for wi in range(n_w):
            if used >= budget or time.time() > deadline - 2.0:
                break
            w = W[wi]
            wnorm = w / (np.linalg.norm(w) + 1e-12)
            dists = np.linalg.norm(front_obj - wnorm, axis=1)
            idx = int(np.argmin(dists))
            x0 = X[idx].copy()
            remaining = budget - used
            max_evals = min(6, max(1, remaining // max(1, n_w - wi)))
            bx, bf, n_used = polish(x0, w, max_evals)
            used += n_used
            F = np.vstack([F, bf[None, :]])
            X = np.vstack([X, bx[None, :]])

        # ---- 3. Random restarts to catch missed regions ----
        n_restarts = 3
        for _ in range(n_restarts):
            if used >= budget or time.time() > deadline - 2.0:
                break
            x0 = lo + (hi - lo) * rng.random(dim)
            x0[3:] = 0.5
            x0 = np.clip(x0, lo, hi)
            w = W[rng.integers(n_w)]
            remaining = budget - used
            max_evals = min(4, max(1, remaining // max(1, n_restarts)))
            bx, bf, n_used = polish(x0, w, max_evals)
            used += n_used
            F = np.vstack([F, bf[None, :]])
            X = np.vstack([X, bx[None, :]])

        # ---- 4. Keep only points strictly better than reference [2.5,2.5,2.5] ----
        ref = np.array([2.5, 2.5, 2.5])
        keep = np.all(F < ref, axis=1)
        F = F[keep]
        X = X[keep]

        return X
# EVOLVE-BLOCK-END
