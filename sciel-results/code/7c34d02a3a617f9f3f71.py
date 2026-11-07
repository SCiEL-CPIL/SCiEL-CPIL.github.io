"""Surrogate-assisted multi-objective optimizer on RE34 (GP + EHVI)."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.stats import norm


class Optimizer:
    """Gaussian-process surrogate with Expected-Hypervolume-Improvement acquisition."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        t0 = time.time()
        deadline = t0 + 300.0  # 300 s safety margin under the 330 s kill
        dim, n_obj = self.dim, self.n_obj
        lo = np.array([b[0] for b in bounds], dtype=float)
        hi = np.array([b[1] for b in bounds], dtype=float)
        ref = np.array([1705.0, 12.0, 0.3])
        rng = np.random.default_rng(0)

        # ---- initialization: LHS spread to seed the surrogate ----
        n_init = min(self.budget, max(10, dim * 3 + 5))
        X = np.zeros((n_init, dim))
        for d in range(dim):
            X[:, d] = lo[d] + (hi[d] - lo[d]) * (np.arange(n_init) + 0.5) / n_init
        F = np.array([problem(x) for x in X])
        used = n_init
        if used >= self.budget:
            return X

        # ---- GP helpers ----
        def RBF(D):
            return np.exp(-0.5 * D)

        def fit_gp(Xtr, Ytr, Yvar):
            n = len(Xtr)
            if n == 1:
                return Ytr[0], 1.0, 1.0
            D = np.sqrt(np.maximum(0.0, np.sum((Xtr[:, None, :] - Xtr[None, :, :]) ** 2, axis=-1)))
            K = RBF(D)
            K = K + Yvar + 1e-9 * np.eye(n)
            try:
                L = np.linalg.cholesky(K)
            except np.linalg.LinAlgError:
                L = np.linalg.cholesky(K + 1e-6 * np.eye(n))
            alpha = np.linalg.solve(L.T, np.linalg.solve(L, Ytr))
            return alpha, L, 1.0

        def predict(Xq, alpha, L, Ymean, Yvar):
            n = len(alpha)
            D = np.sqrt(np.maximum(0.0, np.sum((Xq[:, None, :] - Xtr_ref[None, :, :]) ** 2, axis=-1)))
            K = RBF(D)
            v = np.linalg.solve(L, K.T)
            mu = Ymean + v @ alpha
            var = Yvar - np.sum(v ** 2, axis=1)
            var = np.maximum(var, 1e-10)
            return mu, np.sqrt(var)

        # ---- EHVI acquisition (3 objectives, exact via sorted decomposition) ----
        def ehvi(mu, sd, Fp, refp):
            m = len(Fp)
            if m == 0:
                return 0.0
            base = _hv(Fp, refp)
            mu = np.maximum(mu, refp)  # contributions below ref are zero
            sd = np.maximum(sd, 1e-8)
            z = (mu - refp) / sd
            # candidate points: current front + mu (best-case)
            pts = np.vstack([Fp, mu])
            # sorted decomposition: each point dominates the box from itself to the next-worse corner
            order = np.argsort(-pts.sum(axis=1))
            cand = pts[order]
            total = 0.0
            for i in range(len(cand)):
                lower = cand[i]
                upper = refp.copy()
                for j in range(i + 1, len(cand)):
                    for k in range(n_obj):
                        upper[k] = min(upper[k], cand[j][k])
                if np.any(upper <= lower):
                    continue
                # box [lower, upper); integral of P(all obj <= corner) over box
                # = product over dims of E[min(1, (U_k - m_k)/s_k)]-style term
                contrib = 0.0
                for k in range(n_obj):
                    a = (lower[k] - mu[k]) / sd[k]
                    b = (upper[k] - mu[k]) / sd[k]
                    term = _integral_min1(a, b)
                    contrib += term
                # product form (independence across objectives of the marginal)
                # E[prod 1{obj<=c}] approximated by product of marginals
                prod = 1.0
                for k in range(n_obj):
                    a = (lower[k] - mu[k]) / sd[k]
                    b = (upper[k] - mu[k]) / sd[k]
                    prod *= _integral_min1(a, b)
                total += prod * np.prod(np.maximum(0.0, upper - lower))
            return max(0.0, total - base)

        def _integral_min1(a, b):
            # integral over [a,b] of min(1, x) for standard-normal variable density N(0,1)
            if b <= a:
                return 0.0
            if b <= 0.0:
                return _phi(b) - _phi(a)
            if a >= 0.0:
                return _Phi(b) - _Phi(a)
            # a<0<b: split at 0
            return _phi(0.0) - _phi(a) + _Phi(b) - _Phi(0.0)

        def _phi(x):
            return norm.pdf(x)

        def _Phi(x):
            return norm.cdf(x)

        def _hv(Fp, refp):
            # exact 3-D hypervolume (inclusion-exclusion over sorted points)
            m = len(Fp)
            if m == 0:
                return 0.0
            P = [tuple(p) for p in Fp if all(p[k] < refp[k] for k in range(n_obj))]
            if not P:
                return 0.0
            # sort by first objective ascending (best first)
            P.sort(key=lambda p: p[0])
            # build union volume of boxes [p, ref)
            # recursive subdivision by 1st objective
            return _hv3(P, refp)

        def _hv3(P, refp):
            # P sorted by p[0] ascending; compute volume of union of boxes [p, ref)
            if not P:
                return 0.0
            vol = 0.0
            prev0 = refp[0]
            # iterate points from worst (largest p0) to best
            for p in reversed(P):
                # box from p0 to prev0 in dim0, and 2D union over dims 1,2 of points with p0'<=p0
                sub = [q for q in P if q[0] <= p[0]]
                area2 = _hv2(sub, (refp[1], refp[2]))
                vol += (prev0 - p[0]) * area2
                prev0 = p[0]
            return vol

        def _hv2(P, ref2):
            # 2D union area of boxes [p1,p2) -> [ref1,ref2), P sorted by p1
            if not P:
                return 0.0
            P = sorted(P, key=lambda p: p[1])
            area = 0.0
            prev1 = ref2[1]
            for p in reversed(P):
                sub = [q for q in P if q[1] <= p[1]]
                w = max(0.0, min(q[0] for q in sub) if sub else ref2[0])
                w = min(q[0] for q in sub)
                area += (prev1 - p[1]) * (ref2[0] - w)
                prev1 = p[1]
            return area

        # keep a reference to Xtr for predict closure
        global _Xtr_ref
        Xtr = X.copy()
        Ytr = F.copy()
        Yvar = 1e-6 * np.ones(n_obj)
        Ymean = np.zeros(n_obj)
        alpha, L, _ = fit_gp(Xtr, Ytr, Yvar)
        Xtr_ref = Xtr

        # ---- main loop: propose candidates, pick best EHVI, evaluate ----
        while used < self.budget:
            if time.time() > deadline - 3.0:
                break
            # refit each iteration (cheap for modest n)
            alpha, L, _ = fit_gp(Xtr, Ytr, Yvar)
            Xtr_ref = Xtr
            # build candidate pool: Latin hypercube + local jitter around current best front
            n_cand = 200
            C = np.zeros((n_cand, dim))
            for d in range(dim):
                C[:, d] = lo[d] + (hi[d] - lo[d]) * (np.arange(n_cand) + rng.random()) / n_cand
            # local candidates around non-dominated points
            dom = np.array([np.any(np.all(Ytr <= f, axis=1) & np.any(Ytr < f, axis=1)) for f in Ytr])
            nd = Xtr[~dom] if np.any(~dom) else Xtr
            n_local = min(100, n_cand)
            idx = rng.integers(0, len(nd), size=n_local)
            C = np.vstack([C[:n_cand - n_local], nd[idx] + 0.1 * (hi - lo) * rng.standard_normal((n_local, dim))])
            C = np.clip(C, lo, hi)

            mu, sd = predict(C, alpha, L, Ymean, Yvar)
            # non-dominated front of observed
            dom = np.array([np.any(np.all(Ytr <= f, axis=1) & np.any(Ytr < f, axis=1)) for f in Ytr])
            Fp = Ytr[~dom] if np.any(~dom) else Ytr
            acq = np.array([ehvi(mu[i], sd[i], Fp, ref) for i in range(len(C))])
            best = int(np.argmax(acq))
            xq = C[best]
            fq = problem(xq)
            used += 1
            # append to training set
            Xtr = np.vstack([Xtr, xq])
            Ytr = np.vstack([Ytr, fq])
        return Xtr
# EVOLVE-BLOCK-END
