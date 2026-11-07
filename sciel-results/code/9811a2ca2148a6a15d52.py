# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize, LinearConstraint


class Optimizer:
    """Multi-objective optimizer for RE37: Sobol coverage + weighted-sum Nelder-Mead refinement."""

    def __init__(self, dim, budget, n_obj):
        self.dim = dim
        self.budget = budget
        self.n_obj = n_obj
        self._ref = np.array([1.1, 1.3, 1.1])

    def __call__(self, problem, bounds):
        t0 = time.time()
        deadline = t0 + 280.0  # 20s safety margin before 300s kill
        dim = self.dim
        n_obj = self.n_obj

        lo = np.array([b[0] for b in bounds], dtype=float)
        hi = np.array([b[1] for b in bounds], dtype=float)
        span = hi - lo
        ref = self._ref

        X = []
        F = []
        used = 0

        def eval_point(x):
            nonlocal used
            fx = np.asarray(problem(x), dtype=float)
            used += 1
            X.append(np.asarray(x, dtype=float))
            F.append(fx)
            return fx

        def dominated_mask(Fm):
            # returns boolean array: True if point i is dominated by some other point
            n = Fm.shape[0]
            m = np.zeros(n, dtype=bool)
            for i in range(n):
                better = (Fm <= Fm[i]).all(axis=1) & (Fm < Fm[i]).any(axis=1)
                better[i] = False
                if better.any():
                    m[i] = True
            return m

        # --- Phase 1: Sobol quasi-random coverage ---
        n_init = 60
        try:
            from scipy.stats import qmc
            sampler = qmc.Sobol(d=dim, scramble=True, seed=12345)
            u = sampler.random(n_init)
            init_pts = lo + span * u
        except Exception:
            rng = np.random.default_rng(7)
            init_pts = lo + span * rng.random((n_init, dim))

        for x in init_pts:
            if used >= self.budget:
                break
            if time.time() > deadline:
                break
            eval_point(np.clip(x, lo, hi))

        # --- Phase 2: weighted-sum Nelder-Mead refinement on non-dominated points ---
        # Build a set of weight vectors to explore the front
        weights = []
        # Simplex (extreme) weights
        for k in range(n_obj):
            w = np.zeros(n_obj)
            w[k] = 1.0
            weights.append(w)
        # Mid weights
        for a in [0.25, 0.5, 0.75]:
            for b in [0.25, 0.5, 0.75]:
                w = np.array([a, b, 1.0 - a - b])
                if w.min() >= 0.01:
                    weights.append(w)
        # A few random weights
        rng = np.random.default_rng(99)
        for _ in range(6):
            w = rng.random(n_obj)
            w = w / w.sum()
            weights.append(w)

        # Local refinement: Nelder-Mead on scalar weighted sum
        def local_refine(start, w, remaining_budget):
            nonlocal used
            # objective: weighted sum of objectives
            def obj(x):
                if used >= self.budget:
                    return 1e9
                fx = np.asarray(problem(x), dtype=float)
                used += 1
                X.append(np.asarray(x, dtype=float))
                F.append(fx)
                return float(np.dot(w, fx))
            try:
                res = minimize(obj, start, method='Nelder-Mead',
                               options={'maxiter': 12, 'xatol': 1e-4, 'fatol': 1e-5})
            except Exception:
                pass

        # Iterate: refine non-dominated points
        for _ in range(4):  # rounds
            if used >= self.budget or time.time() > deadline:
                break
            Fm = np.array(F)
            Xa = np.array(X)
            dm = dominated_mask(Fm)
            nd_idx = np.where(~dm)[0]
            if len(nd_idx) == 0:
                nd_idx = np.array([int(np.argmin(np.sum(Fm, axis=1)))])
            # Pick a few non-dominated points to refine
            for idx in nd_idx[:6]:
                if used >= self.budget or time.time() > deadline:
                    break
                # choose a weight roughly opposite to this point's objective profile
                fp = Fm[idx]
                w = (ref - fp + 1e-3)
                w = np.clip(w, 0.01, None)
                w = w / w.sum()
                local_refine(Xa[idx], w, self.budget)

        # --- Phase 3: fill remaining budget with targeted random + refinement ---
        rng2 = np.random.default_rng(2024)
        while used < self.budget and time.time() < deadline:
            # random point near a good region
            Fm = np.array(F)
            dm = dominated_mask(Fm)
            nd_idx = np.where(~dm)[0]
            if len(nd_idx) > 0:
                base = Fm[nd_idx[rng2.integers(len(nd_idx))]]
                # sample in a box around best-known objective region mapped back via random x
                x = X[np.where(~dominated_mask(Fm))[0][rng2.integers(len(np.where(~dominated_mask(Fm))[0]))]]
            else:
                x = lo + span * rng2.random(dim)
            x = x + 0.08 * span * rng2.standard_normal(dim)
            x = np.clip(x, lo, hi)
            eval_point(x)

        return X
# EVOLVE-BLOCK-END
