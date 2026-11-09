"""CMA-ES with adaptive restarts and GP-based local polish for BBOB f16 (Weierstrass)."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.spatial.distance import cdist


class Optimizer:
    """IPOP-CMA-ES with GP-based local polish for multimodal Weierstrass."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        t0 = time.time()
        deadline = t0 + 280.0  # safety margin before 300s limit
        rng = np.random.default_rng(42)

        # Collect all evaluations
        all_x = []
        all_f = []

        def call_func(x):
            nonlocal all_x, all_f
            x = np.asarray(x, dtype=np.float64).clip(-5, 5)
            f = func(x)
            all_x.append(x.copy())
            all_f.append(float(f))
            return f

        def check_time():
            return time.time() < deadline

        # --- Phase 1: IPOP-CMA-ES global search (60% of budget) ---
        global_budget = int(self.budget * 0.6)
        local_budget = self.budget - global_budget

        # CMA-ES parameters
        def cma_es(budget, x0, sigma0, max_sigma=10.0):
            n = self.dim
            lam = max(4, int(4 + 3 * np.log(n)))
            mu = lam // 2
            w = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
            w = w / w.sum()
            mueff = 1.0 / np.sum(w ** 2)
            cc = (4 + mueff / n) / (n + 4 + 2 * mueff / n)
            cs = (mueff + 2) / (n + mueff + 5)
            c1 = (2 - cc + 2 * min(mu, 2) / (n + 1) - 1) / ((n + 1.3) ** 2 + mueff)
            cmu = min(1 - c1, 2 * (mueff - 2 + 1 / mueff) / ((n + 2) ** 2 + mueff))
            damps = n * cs * (cc + 2)
            sigma = sigma0
            xmean = x0.copy()
            C = np.eye(n)
            Cinv = np.eye(n)
            Csqrt = np.eye(n)
            Cinv_sqrt = np.eye(n)
            ps = np.zeros(n)
            pc = np.zeros(n)
            chi_n = np.sqrt(n) * (1 - 1 / (4 * n) + 1 / (21 * n ** 2))

            best_x = x0.copy()
            best_f = call_func(x0)
            used = 1

            while used < budget and check_time():
                # Generate lambda offspring
                ys = []
                for i in range(lam):
                    y = xmean + sigma * Csqrt @ rng.standard_normal(n)
                    y = np.clip(y, -5, 5)
                    ys.append(y)
                    used += 1
                    if not check_time():
                        break

                if len(ys) < lam:
                    break

                fs = np.array([call_func(y) for y in ys])
                # Sort
                order = np.argsort(fs)
                ys = [ys[i] for i in order]
                fs = fs[order]

                # Update best
                if fs[0] < best_f:
                    best_f = fs[0]
                    best_x = ys[0].copy()

                # Update mean
                xnew = sum(w[i] * ys[i] for i in range(mu))
                # Update evolution paths
                ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mueff) * Cinv_sqrt @ (xnew - xmean) / sigma
                hsig = (np.linalg.norm(ps) / np.sqrt(1 - (1 - cs) ** (2 * (used + 1)))) <= (1.4 + 2 / (n + 1)) * chi_n
                pc = (1 - cc) * pc + hsig * np.sqrt(cc * (2 - cc) * mueff) * (xnew - xmean) / sigma
                # Update covariance
                deltaD = 1 + 2 * max(0, (hsig - 1) * cs * (1 - cc)) + (cmu - c1)
                rank_mu = np.outer(xnew - xmean, xnew - xmean)
                rank_one = np.outer(pc, pc)
                C = (1 + c1 * (deltaD - 1) - cmu) * C + c1 * rank_one + cmu * rank_mu
                # Ensure symmetry and positive definite
                C = (C + C.T) / 2
                # Update square root decomposition
                try:
                    eigval, eigvec = np.linalg.eigh(C)
                    eigval = np.maximum(eigval, 1e-20)
                    Csqrt = eigvec @ np.diag(np.sqrt(eigval)) @ eigvec.T
                    Cinv_sqrt = eigvec @ np.diag(1.0 / np.sqrt(eigval)) @ eigvec.T
                    Cinv = eigvec @ np.diag(1.0 / eigval) @ eigvec.T
                except np.linalg.LinAlgError:
                    C = np.eye(n)
                    Csqrt = np.eye(n)
                    Cinv_sqrt = np.eye(n)
                    Cinv = np.eye(n)

                xmean = xnew

                # Adapt sigma
                sigma *= np.exp(cs / damps * (np.linalg.norm(ps) / chi_n - 1))
                sigma = min(sigma, max_sigma)
                sigma = max(sigma, 1e-8)

                # Stagnation check: restart if sigma too small or no improvement
                if sigma < 1e-4 or np.linalg.norm(xmean - best_x) < 1e-6:
                    break

            return best_x, best_f, used

        # IPOP-CMA-ES with restarts
        best_x_global = None
        best_f_global = np.inf
        remaining = global_budget
        sigma_init = 2.0
        restart_count = 0
        max_restarts = 3

        while remaining > 0 and check_time() and restart_count <= max_restarts:
            # Determine sub-budget for this CMA-ES run
            if restart_count == 0:
                sub_budget = int(remaining * 0.5)
            else:
                sub_budget = int(remaining * 0.3)
            sub_budget = max(sub_budget, 10)

            # Start point
            if best_x_global is None:
                x0 = rng.uniform(-5, 5, self.dim)
            else:
                # Perturb best found
                x0 = np.clip(best_x_global + sigma_init * 0.5 * rng.standard_normal(self.dim), -5, 5)

            x_best, f_best, used = cma_es(sub_budget, x0, sigma_init)

            if f_best < best_f_global:
                best_f_global = f_best
                best_x_global = x_best.copy()

            remaining -= used
            restart_count += 1
            # Increase sigma for next restart
            sigma_init *= 1.5

        # --- Phase 2: GP-based local polish (40% of budget) ---
        if check_time() and local_budget > 0 and len(all_x) >= 5:
            X = np.array(all_x)
            F = np.array(all_f)
            # Use only the best 50% of points for GP to keep it lightweight
            n_points = min(len(X), 100)
            # Sort by fitness, take best
            order = np.argsort(F)[:n_points]
            X_gp = X[order]
            F_gp = F[order]

            def gp_predict(X_test):
                """Simple GP with RBF kernel."""
                n = len(X_gp)
                m = len(X_test)
                K = np.zeros((n, n))
                for i in range(n):
                    for j in range(n):
                        d2 = np.sum((X_gp[i] - X_gp[j]) ** 2)
                        K[i, j] = np.exp(-0.5 * d2 / 0.5 ** 2)
                K += 1e-6 * np.eye(n)
                Kf = np.zeros((n, m))
                for i in range(n):
                    for j in range(m):
                        d2 = np.sum((X_gp[i] - X_test[j]) ** 2)
                        Kf[i, j] = np.exp(-0.5 * d2 / 0.5 ** 2)
                try:
                    alpha = np.linalg.solve(K, F_gp)
                    mu = Kf.T @ alpha
                    v = np.linalg.solve(K, Kf)
                    var = 1.0 - np.sum(Kf * v, axis=0)
                    var = np.maximum(var, 1e-8)
                    return mu, np.sqrt(var)
                except np.linalg.LinAlgError:
                    return np.zeros(m), np.ones(m) * 0.1

            # Expected Improvement
            def ei(x, f_best):
                mu, sigma = gp_predict(x.reshape(1, -1))
                mu = mu[0]
                sigma = sigma[0]
                if sigma < 1e-8:
                    return 0.0
                z = (f_best - mu) / sigma
                from scipy.stats import norm
                ei_val = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
                return max(ei_val, 0.0)

            # Local polish: propose points using EI
            best_x_local = best_x_global.copy()
            best_f_local = best_f_global
            polish_used = 0

            while polish_used < local_budget and check_time():
                # Generate candidate points around best
                candidates = []
                n_candidates = 20
                for _ in range(n_candidates):
                    # Mix of local and slightly global perturbations
                    scale = 0.5 * (1 - polish_used / local_budget) + 0.1
                    x_cand = np.clip(best_x_local + scale * rng.standard_normal(self.dim), -5, 5)
                    candidates.append(x_cand)

                # Evaluate EI for each candidate
                ei_vals = []
                for x_c in candidates:
                    ei_vals.append(ei(x_c, best_f_local))

                # Pick best EI
                idx = np.argmax(ei_vals)
                x_new = candidates[idx]
                f_new = call_func(x_new)
                polish_used += 1

                if f_new < best_f_local:
                    best_f_local = f_new
                    best_x_local = x_new.copy()

        # Final result
        if best_f_global < best_f_local:
            final_x = best_x_global
            final_f = best_f_global
        else:
            final_x = best_x_local
            final_f = best_f_local

        return final_x, final_f
# EVOLVE-BLOCK-END
