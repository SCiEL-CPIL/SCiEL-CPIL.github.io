"""Surrogate-assisted Bayesian Optimization with local refinement for BBOB f21."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


class Optimizer:
    """Surrogate-assisted BO with GP model and local refinement for BBOB f21."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 270.0
        calls_made = 0
        dim = self.dim
        budget = self.budget
        rng = np.random.default_rng(12345)

        best_x = None
        best_f = np.inf

        def evaluate(x):
            nonlocal best_x, best_f, calls_made
            if calls_made >= budget:
                return np.inf
            x = np.clip(np.asarray(x, dtype=float), -5.0, 5.0)
            f = float(func(x))
            calls_made += 1
            if f < best_f:
                best_f = f
                best_x = x.copy()
            return f

        # Phase 1: Latin Hypercube initialization for space-filling coverage
        n_init = 40
        X_obs = []
        f_obs = []

        # Latin Hypercube sampling
        lhs = np.zeros((n_init, dim))
        for d in range(dim):
            perm = rng.permutation(n_init)
            l = (perm + rng.uniform(size=n_init)) / n_init
            l = 2.0 * l * 9.0 - 5.0  # map to [-5, 5]
            l = np.clip(l, -5.0, 5.0)
            lhs[:, d] = l

        for i in range(n_init):
            if calls_made >= budget or time.time() > deadline:
                break
            f = evaluate(lhs[i])
            X_obs.append(lhs[i])
            f_obs.append(f)

        X_obs = np.array(X_obs)
        f_obs = np.array(f_obs)

        # Phase 2: Surrogate-assisted Bayesian Optimization
        def gp_predict(X_new, X_train, f_train, length_scale=1.5, noise=1e-4):
            """Simple GP prediction with RBF kernel."""
            n = X_train.shape[0]
            m = X_new.shape[0]
            d = X_train.shape[1]

            # RBF kernel
            def rbf(K1, K2, ls):
                diff = K1[:, None, :] - K2[None, :, :]
                sq_dist = np.sum(diff ** 2, axis=2)
                return np.exp(-0.5 * sq_dist / (ls ** 2))

            K = rbf(X_train, X_train, length_scale) + noise * np.eye(n)
            K_s = rbf(X_train, X_new, length_scale)
            K_ss = rbf(X_new, X_new, length_scale)

            try:
                L = np.linalg.cholesky(K + 1e-8 * np.eye(n))
                alpha = np.linalg.solve(L.T, np.linalg.solve(L, f_train))
                mu = K_s.T @ alpha
                v = np.linalg.solve(L, K_s)
                var = np.diag(K_ss) - np.sum(v ** 2, axis=0)
                var = np.maximum(var, 1e-10)
            except np.linalg.LinAlgError:
                # Fallback: simple interpolation
                dists = np.linalg.norm(X_new[:, None, :] - X_train[None, :, :], axis=2)
                w = np.exp(-dists ** 2 / (2 * length_scale ** 2))
                w_sum = np.sum(w, axis=1) + 1e-10
                mu = (w @ f_train) / w_sum
                var = np.full(m, np.var(f_train) * 0.5)

            return mu, var

        def acquisition_function(X_cand, X_train, f_train, best_f_val, xi=0.05):
            """Expected Improvement acquisition function."""
            mu, var = gp_predict(X_cand, X_train, f_train, length_scale=1.2)
            sigma = np.sqrt(var)
            imp = (best_f_val - mu - xi)
            Z = imp / (sigma + 1e-10)
            from scipy.stats import norm
            EI = imp * norm.cdf(Z) + sigma * norm.pdf(Z)
            return EI

        def find_best_acquisition(X_train, f_train, best_f_val, n_candidates=200):
            """Find point with highest acquisition value."""
            # Generate candidate points: random + around good points
            candidates = rng.uniform(-5, 5, (n_candidates // 2, dim))

            # Add points near top performers
            top_k = min(5, len(f_train))
            idx = np.argsort(f_train)[:top_k]
            for i in idx:
                local = X_train[i] + rng.standard_normal((n_candidates // (2 * top_k) + 1, dim)) * 0.8
                local = np.clip(local, -5, 5)
                candidates = np.vstack([candidates, local])

            candidates = np.clip(candidates, -5, 5)

            EI = acquisition_function(candidates, X_train, f_train, best_f_val)
            best_idx = np.argmax(EI)
            return candidates[best_idx]

        n_bo_iter = min(budget - calls_made, 200)
        bo_no_improve = 0
        max_no_improve = 15

        for it in range(n_bo_iter):
            if time.time() > deadline or calls_made >= budget:
                break

            # Find next point using acquisition function
            x_next = find_best_acquisition(X_obs, f_obs, best_f)
            f_next = evaluate(x_next)

            X_obs = np.vstack([X_obs, x_next])
            f_obs = np.append(f_obs, f_next)

            if f_next < best_f - 1e-6:
                bo_no_improve = 0
            else:
                bo_no_improve += 1

            if bo_no_improve >= max_no_improve:
                # Adaptive exploration: inject random points to escape local optima
                for _ in range(min(5, budget - calls_made)):
                    if time.time() > deadline or calls_made >= budget:
                        break
                    x_rand = rng.uniform(-5, 5, dim)
                    f_rand = evaluate(x_rand)
                    X_obs = np.vstack([X_obs, x_rand])
                    f_obs = np.append(f_obs, f_rand)
                bo_no_improve = 0
                # Reduce GP length scale to explore more finely
                break

        # Phase 3: Local refinement around best point using adaptive (1+1)-ES
        if time.time() < deadline and calls_made < budget:
            x_curr = best_x.copy()
            f_curr = best_f
            sigma = 0.3
            es_calls = 0
            max_es_calls = min(budget - calls_made, 80)
            stagnation = 0

            while es_calls < max_es_calls:
                if time.time() > deadline or calls_made >= budget:
                    break

                y = np.clip(x_curr + sigma * rng.standard_normal(dim), -5, 5)
                fy = evaluate(y)
                es_calls += 1

                if fy < f_curr:
                    x_curr, f_curr = y, fy
                    sigma = min(sigma * 1.15, 3.0)
                    stagnation = 0
                else:
                    sigma = max(sigma * 0.85, 0.01)
                    stagnation += 1

                if stagnation >= 25:
                    break

        # Phase 4: Final Nelder-Mead polish if budget allows
        if time.time() < deadline and calls_made < budget - 10:
            x_nm = best_x.copy()
            delta = 0.15
            simplex = [x_nm.copy()]
            for i in range(dim):
                xi = x_nm.copy()
                xi[i] += delta
                simplex.append(xi)

            simplex_f = [evaluate(simplex[0])]
            for i in range(1, dim + 1):
                if time.time() > deadline or calls_made >= budget:
                    break
                simplex_f.append(evaluate(simplex[i]))

            if len(simplex_f) == dim + 1:
                alpha, gamma, rho, sigma_nm = 1.0, 2.0, 0.5, 0.5
                max_iter = min((budget - calls_made) // 3, 20)

                for _ in range(max_iter):
                    if time.time() > deadline or calls_made >= budget:
                        break

                    indices = np.argsort(simplex_f)
                    simplex = [simplex[i] for i in indices]
                    simplex_f = [simplex_f[i] for i in indices]

                    centroid = np.mean(simplex[:-1], axis=0)
                    xr = centroid + alpha * (centroid - simplex[-1])
                    fr = evaluate(xr)

                    if fr < simplex_f[0]:
                        xe = centroid + gamma * (centroid - simplex[-1])
                        fe = evaluate(xe)
                        if fe < fr:
                            simplex[-1] = xe
                            simplex_f[-1] = fe
                        else:
                            simplex[-1] = xr
                            simplex_f[-1] = fr
                    elif fr < simplex_f[-2]:
                        simplex[-1] = xr
                        simplex_f[-1] = fr
                    else:
                        xc = centroid + rho * (centroid - simplex[-1])
                        fc = evaluate(xc)
                        if fc < min(fr, simplex_f[-1]):
                            simplex[-1] = xc
                            simplex_f[-1] = fc
                        else:
                            for i in range(1, len(simplex)):
                                simplex[i] = simplex[0] + sigma_nm * (simplex[i] - simplex[0])
                                if time.time() > deadline or calls_made >= budget:
                                    break
                                simplex_f[i] = evaluate(simplex[i])
                            if fc < best_f:
                                best_x = xc.copy()
                                best_f = fc

                    # Update best from simplex
                    idx_best = np.argmin(simplex_f)
                    if simplex_f[idx_best] < best_f:
                        best_f = simplex_f[idx_best]
                        best_x = simplex[idx_best].copy()

        return best_x, best_f
# EVOLVE-BLOCK-END
