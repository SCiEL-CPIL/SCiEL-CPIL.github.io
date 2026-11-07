import numpy as np
import time
import scipy.optimize
from scipy.stats import norm

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280
        rng = np.random.default_rng(42)
        n = self.dim
        budget = self.budget
        calls = 0

        def time_ok():
            return time.time() < deadline

        def elapsed():
            return time.time() - start_time

        # Phase 1: GP-guided exploration (~40% of budget)
        gp_budget = int(budget * 0.4)

        # GP kernel parameters
        length_scales = np.ones(n) * 2.0
        noise_var = 1e-6
        signal_var = 1.0

        def gp_kernel(X1, X2):
            """Matern 3/2 kernel"""
            diff = X1[:, np.newaxis, :] - X2[np.newaxis, :, :]
            dist = np.sqrt(np.sum(diff ** 2, axis=-1))
            sqrt3 = np.sqrt(3.0)
            K = signal_var * (1 + sqrt3 * dist) * np.exp(-sqrt3 * dist)
            return K

        def gp_predict(X_new, X_obs, y_obs):
            """GP prediction at X_new given observations"""
            K = gp_kernel(X_obs, X_obs) + noise_var * np.eye(len(X_obs))
            K_s = gp_kernel(X_obs, X_new)
            K_ss = gp_kernel(X_new, X_new)

            try:
                L = np.linalg.cholesky(K)
                alpha = np.linalg.solve(L.T, np.linalg.solve(L, y_obs))
                v = np.linalg.solve(L, K_s.T)
                mu = K_s @ alpha
                var = np.diag(K_ss) - np.sum(v ** 2, axis=0)
                var = np.maximum(var, 1e-12)
            except np.linalg.LinAlgError:
                mu = np.zeros(len(X_new))
                var = np.full(len(X_new), signal_var)

            return mu, np.sqrt(var)

        def expected_improvement(mu, sigma, fbest, xi=0.01):
            """Expected improvement acquisition function"""
            sigma = np.maximum(sigma, 1e-12)
            z = (fbest - mu) / sigma
            ei = (fbest - mu) * norm.cdf(z) + sigma * norm.pdf(z)
            ei = np.maximum(ei, 0.0)
            return ei

        # Initial observations: deterministic diverse points
        X_obs = []
        y_obs = []

        # Evaluate a set of initial points
        init_points = [np.zeros(n)]
        for i in range(1, 6):
            scale = 1.0 + i * 0.5
            x_try = rng.uniform(-scale, scale, n)
            x_try = np.clip(x_try, -5, 5)
            init_points.append(x_try)

        for x in init_points:
            if calls >= gp_budget or not time_ok():
                break
            fx = func(x)
            calls += 1
            X_obs.append(x.copy())
            y_obs.append(fx)

        if not X_obs:
            return np.zeros(n), 0.0

        X_obs = np.array(X_obs)
        y_obs = np.array(y_obs)
        fbest = np.min(y_obs)
        x_best = X_obs[np.argmin(y_obs)].copy()

        # GP-guided exploration loop
        no_improve_iter = 0
        gp_iter = 0
        max_gp_iter = 50

        while calls < gp_budget and time_ok() and (deadline - time.time()) > 60 and gp_iter < max_gp_iter:
            # Generate candidate points using EI
            n_candidates = 200
            candidates = rng.uniform(-5, 5, (n_candidates, n))

            # Also add some points near current best
            for i in range(20):
                scale = 0.5 + i * 0.2
                candidates[i] = x_best + scale * rng.standard_normal(n)
                candidates[i] = np.clip(candidates[i], -5, 5)

            mu, sigma = gp_predict(candidates, X_obs, y_obs)
            ei = expected_improvement(mu, sigma, fbest)

            # Select top candidates by EI
            top_k = min(5, len(candidates))
            top_indices = np.argsort(ei)[-top_k:][::-1]

            # Evaluate the best EI point
            x_new = candidates[top_indices[0]].copy()
            fx_new = func(x_new)
            calls += 1

            X_obs = np.vstack([X_obs, x_new])
            y_obs = np.append(y_obs, fx_new)

            if fx_new < fbest:
                fbest = fx_new
                x_best = x_new.copy()
                no_improve_iter = 0
            else:
                no_improve_iter += 1

            # If stuck, add random exploration points
            if no_improve_iter > 10:
                for _ in range(3):
                    if calls >= gp_budget or not time_ok():
                        break
                    x_rand = rng.uniform(-5, 5, n)
                    fx_rand = func(x_rand)
                    calls += 1
                    X_obs = np.vstack([X_obs, x_rand])
                    y_obs = np.append(y_obs, fx_rand)
                    if fx_rand < fbest:
                        fbest = fx_rand
                        x_best = x_rand.copy()
                        no_improve_iter = 0

            # Limit GP observations to avoid slow matrix operations
            if len(X_obs) > 30:
                # Keep best 30 points
                keep_idx = np.argsort(y_obs)[:30]
                X_obs = X_obs[keep_idx]
                y_obs = y_obs[keep_idx]

            gp_iter += 1

        # Phase 2: CMA-ES for local refinement (~40% of budget)
        cma_budget = int(budget * 0.4)
        cma_calls_start = calls

        lam = 8
        mu = lam // 2
        weights = np.log(mu + 1) - np.log(np.arange(1, mu + 1))
        weights /= weights.sum()
        mu_eff = 1.0 / np.sum(weights ** 2)
        c_sigma = (mu_eff + 2) / (n + mu_eff + 5)
        d_sigma = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (n + 1)) - 1) + c_sigma
        c_c = (4 + mu_eff / n) / (n + 4 + 2 * mu_eff / n)
        c_1 = 2 / ((n + 1.3) ** 2 + mu_eff)
        c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((n + 2) ** 2 + mu_eff))
        chi_n = np.sqrt(n) * (1 - 1 / (4 * n) + 1 / (21 * n ** 2))

        sigma = 0.5  # Start smaller since GP already found a good region
        pc = np.zeros(n)
        ps = np.zeros(n)
        C = np.eye(n)
        C_sqrt = np.eye(n)
        inv_C_sqrt = np.eye(n)
        no_improve_gens = 0
        gen = 0

        x = x_best.copy()

        while calls < cma_calls_start + cma_budget and time_ok() and (deadline - time.time()) > 30:
            offspring = []
            for i in range(lam):
                if calls >= cma_calls_start + cma_budget or not time_ok():
                    break
                z = rng.standard_normal(n)
                y = C_sqrt @ z
                xi = x + sigma * y
                xi = np.clip(xi, -5, 5)
                fi = func(xi)
                calls += 1
                offspring.append((fi, xi, z))

            if len(offspring) < 2:
                break

            offspring.sort(key=lambda t: t[0])

            if offspring[0][0] < fbest:
                fbest = offspring[0][0]
                x_best = offspring[0][1].copy()
                no_improve_gens = 0
            else:
                no_improve_gens += 1

            # Restart if stuck
            if no_improve_gens > 15 and sigma < 0.1:
                sigma = 0.5
                pc = np.zeros(n)
                ps = np.zeros(n)
                C = np.eye(n)
                C_sqrt = np.eye(n)
                inv_C_sqrt = np.eye(n)
                no_improve_gens = 0
                x = x_best + 0.3 * rng.standard_normal(n)
                x = np.clip(x, -5, 5)
                continue

            # Update mean
            x_old = x.copy()
            x = np.zeros(n)
            for i in range(mu):
                x += weights[i] * offspring[i][1]

            if sigma > 1e-12:
                y_mean = (x - x_old) / sigma
            else:
                y_mean = np.zeros(n)

            ps = (1 - c_sigma) * ps + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * (inv_C_sqrt @ y_mean)
            ps_norm = np.linalg.norm(ps)
            h_sigma = 1.0 if ps_norm / np.sqrt(1 - (1 - c_sigma) ** (gen + 1)) < (1.4 + 2 / (n + 1)) * chi_n else 0.0
            if h_sigma:
                sigma *= np.exp(c_sigma / d_sigma * (ps_norm / chi_n - 1))

            pc = (1 - c_c) * pc + c_c * y_mean
            delta = (1 + 2 * max(0, mu_eff / n - 1)) * c_c * (1 - c_c)
            rank_one = np.outer(pc, pc)
            rank_mu = np.zeros((n, n))
            for i in range(mu):
                rank_mu += weights[i] * np.outer(offspring[i][2], offspring[i][2])
            C = (1 + c_1 * delta - c_1 - c_mu) * C + c_1 * rank_one + c_mu * rank_mu

            C = (C + C.T) / 2
            try:
                eigvals, eigvecs = np.linalg.eigh(C)
                eigvals = np.clip(eigvals, 1e-12, None)
                C_sqrt = eigvecs @ np.diag(np.sqrt(eigvals)) @ eigvecs.T
                inv_C_sqrt = eigvecs @ np.diag(1.0 / np.sqrt(eigvals)) @ eigvecs.T
            except np.linalg.LinAlgError:
                C = np.eye(n)
                C_sqrt = np.eye(n)
                inv_C_sqrt = np.eye(n)

            sigma = np.clip(sigma, 1e-4, 5.0)
            gen += 1

        # Phase 3: L-BFGS-B for fast local convergence
        if calls < budget and time_ok() and (deadline - time.time()) > 15:
            try:
                def wrapped_func_lbfgs(x):
                    nonlocal calls, fbest
                    if calls >= budget or not time_ok():
                        return fbest
                    x = np.clip(x, -5, 5)
                    fx = func(x)
                    calls += 1
                    if fx < fbest:
                        fbest = fx
                    return fx

                result = scipy.optimize.minimize(
                    wrapped_func_lbfgs, x_best, method='L-BFGS-B',
                    bounds=[(-5, 5)] * n,
                    options={'maxiter': 50, 'gtol': 1e-10, 'ftol': 1e-12}
                )
                if result.fun < fbest:
                    fbest = result.fun
                    x_best = result.x.copy()
            except Exception:
                pass

        # Phase 4: Nelder-Mead for further refinement
        if calls < budget and time_ok() and (deadline - time.time()) > 10:
            try:
                def wrapped_func_nm(x):
                    nonlocal calls, fbest
                    if calls >= budget or not time_ok():
                        return fbest
                    x = np.clip(x, -5, 5)
                    fx = func(x)
                    calls += 1
                    if fx < fbest:
                        fbest = fx
                    return fx

                result = scipy.optimize.minimize(
                    wrapped_func_nm, x_best, method='Nelder-Mead',
                    options={'maxiter': 100, 'xatol': 1e-10, 'fatol': 1e-12, 'adaptive': True}
                )
                if result.fun < fbest:
                    fbest = result.fun
                    x_best = result.x.copy()
            except Exception:
                pass

        # Phase 5: Final fine-tuning
        if calls < budget and time_ok() and (deadline - time.time()) > 5:
            for scale in [0.01, 0.005, 0.001, 0.0005]:
                if calls >= budget or not time_ok():
                    break
                for _ in range(3):
                    if calls >= budget or not time_ok():
                        break
                    perturbation = rng.standard_normal(n) * scale
                    x_try = x_best + perturbation
                    x_try = np.clip(x_try, -5, 5)
                    f_try = func(x_try)
                    calls += 1
                    if f_try < fbest:
                        fbest = f_try
                        x_best = x_try.copy()

        return x_best, fbest
