"""Multi-objective optimizer for RE21: HV-driven evolutionary search + L-BFGS-B polish."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import minimize


class Optimizer:
    """Hypervolume-driven multi-objective search with local polish."""

    def __init__(self, budget, dim, n_obj):
        self.budget, self.dim, self.n_obj = budget, dim, n_obj

    def __call__(self, problem, bounds):
        t0 = time.time()
        deadline = t0 + 290.0  # safety margin under the 300 s limit
        lo = np.array([b[0] for b in bounds], dtype=float)
        hi = np.array([b[1] for b in bounds], dtype=float)
        span = hi - lo
        dim = self.dim

        used = 0
        X = []
        F = []

        def eval_pt(x):
            nonlocal used
            used += 1
            return np.asarray(problem(x), dtype=float)

        def dominated_mask(Fv):
            """mask[i] = True if Fv[i] is dominated by some other row."""
            n = len(Fv)
            dom = np.zeros(n, dtype=bool)
            for i in range(n):
                if dom[i]:
                    continue
                d = (Fv <= Fv[i]) & (Fv < Fv[i])
                if d.sum() > 1:  # at least one other row dominates i
                    dom[i] = True
            return dom

        # 1) Random initialization
        n_init = min(30, self.budget)
        rng = np.random.default_rng(12345)
        for _ in range(n_init):
            x = lo + span * rng.random(dim)
            X.append(x)
            F.append(eval_pt(x))
        if used >= self.budget:
            return np.array(X)

        # 2) Evolutionary loop: HV-contribution driven parent selection
        while used < self.budget:
            if time.time() > deadline:
                break
            Fv = np.array(F)
            Xv = np.array(X)
            dom = dominated_mask(Fv)
            nd_idx = np.where(~dom)[0]
            if len(nd_idx) == 0:
                nd_idx = np.arange(len(Fv))

            # Compute HV contribution (2D) for each non-dominated point
            nd = Fv[nd_idx]
            n_nd = len(nd)
            hv_contrib = np.zeros(n_nd)
            # Sort by first objective ascending
            order = np.argsort(nd[:, 0])
            nd_sorted = nd[order]
            # For 2D: contribution of point i = (ref0 - f0_i) * (f1_prev - f1_i)
            # where f1_prev is the f1 of the point with the next-larger f0 (or ref1)
            ref = np.array([3000.0, 0.06])
            prev_f1 = ref[1]
            for k in range(n_nd - 1, -1, -1):
                f0, f1 = nd_sorted[k]
                # Only count if strictly better than ref
                if f0 < ref[0] and f1 < ref[1]:
                    contrib = (ref[0] - f0) * (prev_f1 - f1)
                    hv_contrib[order[k]] = max(contrib, 0.0)
                else:
                    hv_contrib[order[k]] = 0.0
                prev_f1 = f1

            total_hv = hv_contrib.sum()
            if total_hv <= 0:
                # Fallback: pick random non-dominated
                p = rng.integers(len(nd_idx))
            else:
                probs = hv_contrib / total_hv
                p = rng.choice(len(nd_idx), p=probs)

            parent = Xv[nd_idx[p]]

            # Mutation: adaptive step
            step = 0.15 * span * rng.standard_normal(dim)
            child = np.clip(parent + step, lo, hi)
            fc = eval_pt(child)

            # Replace the point with the lowest HV contribution
            worst = int(np.argmin(hv_contrib))
            X[nd_idx[worst]] = child
            F[nd_idx[worst]] = fc

        # 3) Local polish: L-BFGS-B on non-dominated points with weighted sums
        if time.time() < deadline - 10.0 and used < self.budget:
            Fv = np.array(F)
            Xv = np.array(X)
            dom = dominated_mask(Fv)
            nd_idx = np.where(~dom)[0]
            if len(nd_idx) == 0:
                nd_idx = np.arange(len(Fv))

            # Weight vectors for weighted-sum (2 objectives)
            n_w = min(8, self.budget - used)
            weights = np.array([[1.0 - i / max(n_w - 1, 1), i / max(n_w - 1, 1)]
                                for i in range(n_w)])

            for w in weights:
                if used >= self.budget or time.time() > deadline:
                    break

                def obj_func(x, w=w):
                    f = np.asarray(problem(x), dtype=float)
                    return float(np.dot(w, f))

                # Start from the best point for this weight
                Fv2 = np.array(F)
                scores = Fv2[nd_idx] @ w
                best_local = nd_idx[int(np.argmin(scores))]
                x0 = Xv[best_local].copy()

                res = minimize(obj_func, x0, method='L-BFGS-B',
                               bounds=list(zip(lo, hi)),
                               options={'maxiter': 50, 'gtol': 1e-6})
                if res.success and used < self.budget:
                    xf = np.clip(res.x, lo, hi)
                    ff = eval_pt(xf)
                    # Replace worst HV point
                    Fv3 = np.array(F)
                    dom3 = dominated_mask(Fv3)
                    nd3 = np.where(~dom3)[0]
                    if len(nd3) > 0:
                        nd3_f = Fv3[nd3]
                        ref = np.array([3000.0, 0.06])
                        # Simple: replace the last non-dominated
                        X[nd3[-1]] = xf
                        F[nd3[-1]] = ff

        return np.array(X)
# EVOLVE-BLOCK-END
