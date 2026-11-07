# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


class Optimizer:
    """HV-guided multi-objective local search with LHS init and weighted Nelder-Mead polish."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def _hv_2d(self, pts, ref):
        m = len(pts)
        if m == 0:
            return 0.0
        if m == 1:
            return float((ref[0] - pts[0, 0]) * (ref[1] - pts[0, 1]))
        idx = np.argsort(pts[:, 0])
        pts = pts[idx]
        hv = 0.0
        for i in range(m):
            x_cur = pts[i, 0]
            x_next = pts[i + 1, 0] if i + 1 < m else ref[0]
            y_min = np.min(pts[:i + 1, 1])
            hv += (x_next - x_cur) * (ref[1] - y_min)
        return hv

    def _hv_3d(self, pts, ref):
        n = len(pts)
        if n == 0:
            return 0.0
        if n == 1:
            return float(np.prod(ref - pts[0]))
        idx = np.argsort(pts[:, 0])
        pts = pts[idx]
        hv = 0.0
        for i in range(n):
            x_cur = pts[i, 0]
            x_next = pts[i + 1, 0] if i + 1 < n else ref[0]
            yz = pts[:i + 1, 1:3]
            hv += (x_next - x_cur) * self._hv_2d(yz, ref[1:3])
        return hv

    def _is_pareto(self, F):
        n = len(F)
        if n == 0:
            return np.array([], dtype=bool)
        pareto = np.ones(n, dtype=bool)
        for i in range(n):
            if not pareto[i]:
                continue
            self_mask = np.ones(n, dtype=bool)
            self_mask[i] = False
            dominated = np.any(np.all(F[self_mask] <= F[i], axis=1) &
                               np.any(F[self_mask] < F[i], axis=1))
            if dominated:
                pareto[i] = False
        return pareto

    def _hv_contribs(self, pareto_F, ref):
        n = len(pareto_F)
        if n == 0:
            return np.array([])
        if n == 1:
            return np.array([float(np.prod(ref - pareto_F[0]))])
        mask = np.all(pareto_F < ref, axis=1)
        valid_F = pareto_F[mask]
        if len(valid_F) == 0:
            return np.zeros(n)
        total_hv = self._hv_3d(valid_F, ref)
        contribs = np.zeros(n)
        for i in range(n):
            if not mask[i]:
                contribs[i] = 0.0
                continue
            rem_idx = [j for j in range(n) if j != i and mask[j]]
            if len(rem_idx) == 0:
                contribs[i] = total_hv
            else:
                hv_rem = self._hv_3d(pareto_F[rem_idx], ref)
                contribs[i] = max(total_hv - hv_rem, 1e-10)
        return contribs

    def __call__(self, problem, bounds):
        t0 = time.time()
        deadline = t0 + 300
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        dim = self.dim
        n_obj = self.n_obj
        budget = self.budget
        ref = np.array([1705.0, 12.0, 0.3])

        used = 0
        all_F = []

        def call_p(x):
            nonlocal used
            fx = problem(x)
            used += 1
            return fx

        # Phase 1: LHS + random init (60 points)
        n_lhs = 30
        n_rand = 30
        lhs_X = np.zeros((n_lhs, dim))
        for d in range(dim):
            perm = np.random.permutation(n_lhs)
            lhs_X[:, d] = lo[d] + (hi[d] - lo[d]) * (perm + np.random.uniform(size=n_lhs)) / n_lhs
        lhs_X = np.clip(lhs_X, lo, hi)
        rand_X = lo + np.random.uniform(size=(n_rand, dim)) * (hi - lo)
        rand_X = np.clip(rand_X, lo, hi)
        init_X = np.vstack([lhs_X, rand_X])
        init_F = np.array([call_p(x) for x in init_X])
        all_F.extend(init_F)

        # Build Pareto archive
        valid = np.all(init_F < ref, axis=1)
        if np.any(valid):
            pF = init_F[valid]
            pX = init_X[valid]
            pmask = self._is_pareto(pF)
            pareto_X = pX[pmask].copy()
            pareto_F = pF[pmask].copy()
        else:
            pareto_X = np.empty((0, dim))
            pareto_F = np.empty((0, n_obj))

        # Phase 2: HV-guided local search (80 points)
        cached_contribs = None
        n_global = 80
        while used < 60 + n_global:
            if time.time() > deadline - 20:
                break
            progress = used / budget

            if len(pareto_F) > 0:
                if cached_contribs is None or len(cached_contribs) != len(pareto_F):
                    cached_contribs = self._hv_contribs(pareto_F, ref)
                contribs = np.clip(cached_contribs, 1e-10, None)
                temp = 1.5 * (1.0 - progress) + 0.3
                w = np.power(contribs, 1.0 / temp)
                probs = w / w.sum()
                pi = np.random.choice(len(pareto_F), p=probs)
                parent = pareto_X[pi]
            else:
                parent = lo + np.random.uniform(size=dim) * (hi - lo)

            # Adaptive step
            base = 0.6 * (1.0 - progress * 0.8) + 0.05
            step = base * (0.5 + np.random.uniform())
            n_pert = np.random.randint(1, 4)
            coords = np.random.choice(dim, size=n_pert, replace=False)
            child = parent.copy()
            for c in coords:
                child[c] += step * (hi[c] - lo[c]) * np.random.standard_normal()
            child = np.clip(child, lo, hi)

            fc = call_p(child)
            all_F.append(fc)

            # Update archive
            if np.all(fc < ref):
                dominated = False
                for pf in pareto_F:
                    if np.all(pf <= fc) and np.any(pf < fc):
                        dominated = True
                        break
                if not dominated:
                    pareto_X = np.vstack([pareto_X, [child]])
                    pareto_F = np.vstack([pareto_F, [fc]])
                    keep = np.ones(len(pareto_F), dtype=bool)
                    for i in range(len(pareto_F) - 1):
                        if np.all(pareto_F[i] >= fc) and np.any(pareto_F[i] > fc):
                            keep[i] = False
                    pareto_X = pareto_X[keep]
                    pareto_F = pareto_F[keep]
                    cached_contribs = None

            # Prune archive if too large
            if len(pareto_F) > 40:
                pmask = self._is_pareto(pareto_F)
                pareto_X = pareto_X[pmask]
                pareto_F = pareto_F[pmask]
                if len(pareto_F) > 40:
                    contribs = self._hv_contribs(pareto_F, ref)
                    top = np.argsort(contribs)[-40:]
                    pareto_X = pareto_X[top]
                    pareto_F = pareto_F[top]
                cached_contribs = None

        # Phase 3: Weighted Nelder-Mead polish on top HV contributors
        if used < budget and len(pareto_F) > 0:
            contribs = self._hv_contribs(pareto_F, ref)
            top_k = min(10, len(pareto_F))
            top_idx = np.argsort(contribs)[-top_k:]
            
            # Generate random weight vectors for diversification
            # Weights should be positive and sum to 1
            weights = np.random.dirichlet(np.ones(n_obj), size=top_k)
            
            for k, pi in enumerate(top_idx):
                if used >= budget or time.time() > deadline - 5:
                    break
                
                x0 = pareto_X[pi].copy()
                w_vec = weights[k]
                
                def obj(x, w=w_vec):
                    nonlocal used
                    if used >= budget:
                        return np.inf
                    if time.time() > deadline - 3:
                        return np.inf
                    fx = call_p(x)
                    all_F.append(fx)
                    return np.dot(w, fx)

                try:
                    # Use Nelder-Mead to minimize the weighted sum
                    # maxfev is limited to save budget for other points
                    res = minimize(obj, x0, method='Nelder-Mead',
                                   options={'maxiter': 20, 'maxfev': 6,
                                            'xatol': 1e-6, 'fatol': 1e-8})
                    # Final evaluation to record the best found
                    # The 'obj' function already records, but we ensure the final point is in the archive if it's good
                    # Actually, 'obj' records every call. We just need to update the archive with the result if it's non-dominated.
                    fx = problem(np.clip(res.x, lo, hi))
                    used += 1
                    all_F.append(fx)
                    
                    if np.all(fx < ref):
                        dominated = False
                        for pf in pareto_F:
                            if np.all(pf <= fx) and np.any(pf < fx):
                                dominated = True
                                break
                        if not dominated:
                            pareto_X = np.vstack([pareto_X, [np.clip(res.x, lo, hi)]])
                            pareto_F = np.vstack([pareto_F, [fx]])
                            keep = np.ones(len(pareto_F), dtype=bool)
                            for i in range(len(pareto_F) - 1):
                                if np.all(pareto_F[i] >= fx) and np.any(pareto_F[i] > fx):
                                    keep[i] = False
                            pareto_X = pareto_X[keep]
                            pareto_F = pareto_F[keep]
                except Exception:
                    pass

        return np.array(all_F)

# EVOLVE-BLOCK-END
