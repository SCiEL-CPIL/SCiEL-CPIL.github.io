# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import norm


class Optimizer:
    """DE + GP-surrogate multi-objective optimizer with EHVI acquisition."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        t_start = time.time()
        deadline = t_start + 290.0  # safety margin before 330s kill

        rng = np.random.default_rng(12345)
        lo = np.array([b[0] for b in bounds], dtype=float)
        hi = np.array([b[1] for b in bounds], dtype=float)
        dim = self.dim
        n_obj = self.n_obj
        ref = np.array([1705.0, 12.0, 0.3], dtype=float)

        def _hv2(P, ref2):
            """Exact 2D hypervolume. P: (n,2) all < ref2."""
            n = P.shape[0]
            if n == 0:
                return 0.0
            order = np.argsort(P[:, 0], kind='mergesort')
            P = P[order]
            # suffix min of second coord
            suf_min = np.minimum.accumulate(P[::-1, 1])[::-1]
            area = 0.0
            for i in range(n):
                x_lo = P[i, 0]
                x_hi = P[i+1, 0] if i+1 < n else ref2[0]
                if x_hi <= x_lo:
                    continue
                if suf_min[i] >= ref2[1]:
                    continue
                area += (x_hi - x_lo) * (ref2[1] - suf_min[i])
            return area

        def _hv3(F, ref):
            """Exact 3D hypervolume. F: (n,3)."""
            n = F.shape[0]
            if n == 0:
                return 0.0
            m = np.all(F < ref, axis=1)
            F = F[m]
            n = F.shape[0]
            if n == 0:
                return 0.0
            if n == 1:
                f = F[0]
                if np.any(f >= ref):
                    return 0.0
                return float(np.prod(ref - f))
            order = np.argsort(F[:, 0], kind='mergesort')
            F = F[order]
            hv = 0.0
            for i in range(n):
                x0_lo = F[i, 0]
                x0_hi = F[i+1, 0] if i+1 < n else ref[0]
                if x0_hi <= x0_lo:
                    continue
                pts_2d = F[i:, 1:3]
                m2 = np.all(pts_2d < ref[1:3], axis=1)
                pts_2d = pts_2d[m2]
                hv += (x0_hi - x0_lo) * _hv2(pts_2d, ref[1:3])
            return float(hv)

        def _non_dominated_mask(F):
            n = F.shape[0]
            if n == 0:
                return np.zeros(0, dtype=bool)
            mask = np.ones(n, dtype=bool)
            for i in range(n):
                diff = F - F[i]
                le = np.all(diff <= 0, axis=1)
                lt = np.any(diff < 0, axis=1)
                dom = le & lt
                dom[i] = False
                if np.any(dom):
                    mask[i] = False
            return mask

        # ---- Initial population ----
        n_init = 30
        corners = np.array(np.meshgrid(*[np.array([lo[i], hi[i]]) for i in range(dim)])).T.reshape(-1, dim)
        n_c = min(16, n_init)
        n_r = n_init - n_c
        random_pts = lo + (hi - lo) * rng.random((n_r, dim))
        X = np.vstack([corners[:n_c], random_pts])
        F = np.array([problem(x) for x in X])
        used = len(X)

        # ---- GP surrogate setup ----
        # We'll fit a GP to each objective. Use a simple RBF kernel with ARD.
        # Normalize inputs to [0,1] for GP stability.
        x_lo = np.zeros(dim)
        x_hi = np.ones(dim)

        def _normalize(x):
            return (x - lo) / (hi - lo)

        def _denormalize(xn):
            return lo + xn * (hi - lo)

        def _gp_predict(X_train, y_train, X_test, length_scale=0.3, noise=1e-8):
            """
            Fit a simple GP (zero mean, RBF kernel) and predict mean/std at X_test.
            X_train: (n, d), y_train: (n,), X_test: (m, d)
            Returns (mean, std) arrays of shape (m,).
            """
            n = X_train.shape[0]
            d = X_train.shape[1]

            # Standardize y
            y_mean = np.mean(y_train)
            y_std = np.std(y_train)
            if y_std < 1e-12:
                y_std = 1.0
            y_norm = (y_train - y_mean) / y_std

            # RBF kernel
            def _rbf(X1, X2, ls):
                # X1: (n1, d), X2: (n2, d)
                sq = np.sum(X1**2, axis=1, keepdims=True) + np.sum(X2**2, axis=1, keepdims=True).T - 2 * X1 @ X2.T
                sq = np.maximum(sq, 0)
                return np.exp(-sq / (2 * ls**2))

            K = _rbf(X_train, X_train, length_scale) + noise * np.eye(n)
            Kt = _rbf(X_train, X_test, length_scale)

            try:
                L = np.linalg.cholesky(K)
                alpha = np.linalg.solve(L.T, np.linalg.solve(L, y_norm))
                mu_norm = Kt.T @ alpha
                v = np.linalg.solve(L, Kt)
                var_norm = 1.0 - np.sum(v**2, axis=0)
                var_norm = np.maximum(var_norm, 1e-12)
            except np.linalg.LinAlgError:
                # Fallback: add jitter and retry
                K2 = K + 1e-4 * np.eye(n)
                L = np.linalg.cholesky(K2)
                alpha = np.linalg.solve(L.T, np.linalg.solve(L, y_norm))
                mu_norm = Kt.T @ alpha
                v = np.linalg.solve(L, Kt)
                var_norm = 1.0 - np.sum(v**2, axis=0)
                var_norm = np.maximum(var_norm, 1e-12)

            mu = mu_norm * y_std + y_mean
            std = np.sqrt(var_norm) * y_std
            return mu, std

        # ---- EHVI acquisition ----
        def _ehvi_score(X_cand, F_current):
            """
            Estimate EHVI for a candidate point X_cand.
            X_cand: (d,)
            F_current: (n, 3) current objective values
            Returns: float (higher is better)
            """
            # Predict each objective
            xn = _normalize(X_cand).reshape(1, -1)
            preds = []
            for k in range(n_obj):
                mu_k, std_k = _gp_predict(X_norm, F[:, k], xn, length_scale=0.3)
                preds.append((mu_k[0], std_k[0]))

            # Monte Carlo EHVI estimation
            n_mc = 50
            # Current HV
            hv_cur = _hv3(F_current, ref)

            total_ehvi = 0.0
            for _ in range(n_mc):
                # Sample from GP
                f_sample = np.array([preds[k][0] + preds[k][1] * rng.standard_normal() for k in range(n_obj)])
                # New HV with this point
                F_new = np.vstack([F_current, f_sample.reshape(1, -1)])
                hv_new = _hv3(F_new, ref)
                ehvi = hv_new - hv_cur
                total_ehvi += max(ehvi, 0.0)

            return total_ehvi / n_mc

        # ---- Main loop ----
        # Phase 1: DE for first ~100 evaluations to build a good initial front
        CR = 0.8
        F_DE = 0.85
        de_phase_end = 100

        # Keep track of all evaluated points
        all_X = [X.copy()]
        all_F = [F.copy()]

        def _combine_XF():
            Xc = np.vstack(all_X)
            Fc = np.vstack(all_F)
            return Xc, Fc

        # Precompute normalized X for GP
        X_norm = _normalize(np.vstack(all_X))

        while used < self.budget:
            if time.time() > deadline:
                break

            Xc, Fc = _combine_XF()
            n = Xc.shape[0]

            if used < de_phase_end:
                # DE phase
                nd_mask = _non_dominated_mask(Fc)
                nd_idx = np.where(nd_mask)[0]

                if len(nd_idx) < 3:
                    r1 = rng.integers(n)
                    r2 = rng.integers(n)
                    r3 = rng.integers(n)
                    while r2 == r1:
                        r2 = rng.integers(n)
                    while r3 == r1 or r3 == r2:
                        r3 = rng.integers(n)
                else:
                    nd_F = Fc[nd_idx]
                    vol = np.prod(np.maximum(ref - nd_F, 0), axis=1)
                    vol = np.maximum(vol, 1e-12)
                    w = vol / vol.sum()
                    picks = rng.choice(len(nd_idx), size=3, replace=False, p=w)
                    r1 = nd_idx[picks[0]]
                    r2 = nd_idx[picks[1]]
                    r3 = nd_idx[picks[2]]

                v = Xc[r1] + F_DE * (Xc[r2] - Xc[r3])
                j_rand = rng.integers(dim)
                trial = np.empty(dim)
                mask = (rng.random(dim) < CR) | (np.arange(dim) == j_rand)
                trial[mask] = v[mask]
                trial[~mask] = Xc[r1][~mask]
                trial = np.clip(trial, lo, hi)

                fc = problem(trial)
                used += 1

                all_X.append(trial.reshape(1, -1))
                all_F.append(fc.reshape(1, -1))

                # Update X_norm
                X_norm = _normalize(np.vstack(all_X))

            else:
                # GP/EHVI phase
                # Generate a batch of candidate points
                n_cand = 20
                # Mix of random and local perturbations around non-dominated points
                nd_mask = _non_dominated_mask(Fc)
                nd_idx = np.where(nd_mask)[0]

                cand_list = []
                # Random candidates
                n_rand = n_cand // 2
                for _ in range(n_rand):
                    cand_list.append(lo + (hi - lo) * rng.random(dim))

                # Local perturbations around non-dominated points
                n_local = n_cand - n_rand
                if len(nd_idx) > 0:
                    for _ in range(n_local):
                        idx = nd_idx[rng.integers(len(nd_idx))]
                        base = Xc[idx]
                        scale = 0.1 * (hi - lo)
                        cand = base + scale * rng.standard_normal(dim)
                        cand = np.clip(cand, lo, hi)
                        cand_list.append(cand)
                else:
                    for _ in range(n_local):
                        cand_list.append(lo + (hi - lo) * rng.random(dim))

                cand_arr = np.array(cand_list)

                # Evaluate EHVI for each candidate
                ehvi_scores = np.zeros(n_cand)
                for i in range(n_cand):
                    ehvi_scores[i] = _ehvi_score(cand_arr[i], Fc)

                # Pick the best candidate
                best_idx = np.argmax(ehvi_scores)
                trial = cand_arr[best_idx]

                fc = problem(trial)
                used += 1

                all_X.append(trial.reshape(1, -1))
                all_F.append(fc.reshape(1, -1))

                # Update X_norm
                X_norm = _normalize(np.vstack(all_X))

            # Periodic pruning to keep GP tractable
            if len(all_X) > 300:
                Xc, Fc = _combine_XF()
                nd_mask = _non_dominated_mask(Fc)
                nd_idx = np.where(nd_mask)[0]
                if len(nd_idx) > 150:
                    # Keep top 150 non-dominated by crowding distance
                    F_nd = Fc[nd_idx]
                    n_nd = len(nd_idx)
                    dist = np.zeros(n_nd)
                    for k in range(n_obj):
                        order = np.argsort(F_nd[:, k], kind='mergesort')
                        dist[order[0]] = np.inf
                        dist[order[-1]] = np.inf
                        f_range = F_nd[order[-1], k] - F_nd[order[0], k]
                        if f_range == 0:
                            continue
                        for i in range(1, n_nd-1):
                            dist[order[i]] += (F_nd[order[i+1], k] - F_nd[order[i-1], k]) / f_range
                    keep_local = np.argsort(dist)[::-1][:150]
                    keep_global = nd_idx[keep_local]
                    # Also keep 30 random dominated points
                    rest = np.setdiff1d(np.arange(len(Xc)), keep_global)
                    new_X = [Xc[i].reshape(1, -1) for i in keep_global]
                    new_F = [Fc[i].reshape(1, -1) for i in keep_global]
                    if len(rest) > 0:
                        add = rng.choice(rest, size=min(30, len(rest)), replace=False)
                        for i in add:
                            new_X.append(Xc[i].reshape(1, -1))
                            new_F.append(Fc[i].reshape(1, -1))
                    all_X = new_X
                    all_F = new_F
                    X_norm = _normalize(np.vstack(all_X))

        # Return the full set of evaluated points
        Xc, Fc = _combine_XF()
        return Xc
# EVOLVE-BLOCK-END
