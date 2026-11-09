# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import norm


class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        t0 = time.time()
        deadline = t0 + 295.0
        budget = self.budget
        dim = self.dim

        best_x = np.zeros(dim)
        best_f = np.inf
        calls = 0
        rng = np.random.default_rng(42)

        X_data = []
        F_data = []

        def _time_ok():
            return time.time() < deadline

        def _call(x):
            nonlocal best_x, best_f, calls
            x = np.asarray(x, dtype=float).reshape(dim)
            x = np.clip(x, -5.0, 5.0)
            fx = float(func(x))
            calls += 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            X_data.append(x.copy())
            F_data.append(fx)
            return fx

        # ---------- Phase 1: GP-Guided Exploration ----------
        # Initialize with random points
        n_init = min(20, budget)
        for _ in range(n_init):
            if not _time_ok() or calls >= budget:
                break
            x = rng.uniform(-5.0, 5.0, dim)
            _call(x)

        if best_f == np.inf:
            return best_x, best_f

        # GP Parameters
        # Matern 3/2 kernel
        def matern32_kernel(X1, X2, length_scale=1.0, sigma_f=1.0):
            X1 = np.atleast_2d(X1)
            X2 = np.atleast_2d(X2)
            d2 = np.sum(X1**2, axis=1)[:, None] + np.sum(X2**2, axis=1)[None, :] - 2 * X1 @ X2.T
            d2 = np.maximum(d2, 0.0)
            r = np.sqrt(d2) / length_scale
            K = (sigma_f**2) * (1 + r + r**2 / 3) * np.exp(-r)
            return K

        def gp_predict(X_test, X_train, y_train, length_scale=1.0, sigma_f=1.0, noise=1e-6):
            K = matern32_kernel(X_train, X_train, length_scale, sigma_f)
            K += noise * np.eye(len(X_train))
            K_star = matern32_kernel(X_test, X_train, length_scale, sigma_f)
            try:
                # Use Cholesky for stability and speed
                L = np.linalg.cholesky(K)
                alpha = np.linalg.solve(L.T, np.linalg.solve(L, y_train - np.mean(y_train)))
            except np.linalg.LinAlgError:
                # Fallback to inverse
                try:
                    K_inv = np.linalg.inv(K)
                    alpha = K_inv @ (y_train - np.mean(y_train))
                except:
                    return np.full(len(X_test), np.mean(y_train)), np.full(len(X_test), sigma_f)
            
            mu = np.mean(y_train) + K_star @ alpha
            K_ss = matern32_kernel(X_test, X_test, length_scale, sigma_f)
            var = np.diag(K_ss) - np.sum((K_star @ np.linalg.inv(K))**2, axis=1)
            # Note: Using inv again is inefficient. Better to reuse L.
            # But for simplicity and robustness in this short script, let's try to be efficient.
            # Actually, let's just use the standard formula with pinv if cholesky fails, 
            # but we already handled cholesky. Let's recompute var properly using L.
            
            # Re-do var calculation using L for efficiency
            try:
                # v = K_star.T @ L^-1
                v = np.linalg.solve(L, K_star.T)
                var = np.diag(K_ss) - np.sum(v**2, axis=0)
            except:
                var = np.full(len(X_test), sigma_f**2)
                
            var = np.maximum(var, 1e-8)
            return mu, np.sqrt(var)

        def expected_improvement(mu, sigma, best_f_current, xi=0.01):
            imp = best_f_current - mu - xi
            sigma = np.maximum(sigma, 1e-8)
            z = imp / sigma
            ei = imp * norm.cdf(z) + sigma * norm.pdf(z)
            return np.maximum(ei, 0.0)

        # Adaptive GP parameters
        # Estimate sigma_f from data
        if len(F_data) > 1:
            sigma_f = max(1.0, np.std(F_data) * 2)
        else:
            sigma_f = 1.0
        
        length_scale = 1.5

        # GP Loop
        n_gp_calls = 80
        for _ in range(n_gp_calls):
            if not _time_ok() or calls >= budget:
                break
            
            X_train = np.array(X_data)
            y_train = np.array(F_data)
            
            # Generate candidates
            n_candidates = 100
            X_cand = rng.uniform(-5.0, 5.0, (n_candidates, dim))
            
            # Add points around current best to exploit
            if best_x is not None:
                X_around_best = best_x + rng.normal(0, 0.5, (20, dim))
                X_around_best = np.clip(X_around_best, -5.0, 5.0)
                X_cand = np.vstack([X_cand, X_around_best])
            
            mu, sigma = gp_predict(X_cand, X_train, y_train, length_scale, sigma_f)
            ei = expected_improvement(mu, sigma, best_f, xi=0.005)
            
            best_idx = np.argmax(ei)
            x_next = X_cand[best_idx]
            
            _call(x_next)
            
            # Update sigma_f occasionally
            if len(F_data) % 10 == 0:
                sigma_f = max(1.0, np.std(F_data) * 2)

        # ---------- Phase 2: Multi-Start CMA-ES ----------
        remaining = budget - calls
        if remaining <= 0 or not _time_ok():
            return best_x, best_f

        # Select top starting points from Phase 1
        # Sort by function value
        sorted_indices = np.argsort(F_data)
        top_k = min(3, len(F_data))
        start_points = [np.array(X_data[i]) for i in sorted_indices[:top_k]]
        
        # Allocate budget
        # Give more budget to the best point
        weights = [0.5, 0.3, 0.2][:top_k]
        basin_budgets = [max(10, int(remaining * w)) for w in weights]
        # Adjust to fit remaining
        total_alloc = sum(basin_budgets)
        if total_alloc > remaining:
            scale = remaining / total_alloc
            basin_budgets = [int(b * scale) for b in basin_budgets]

        def cma_es(start, n_eval, sigma0=0.5):
            nonlocal best_x, best_f, calls
            if n_eval < 4:
                return

            x = start.copy()
            sigma = sigma0
            popsize = max(6, dim + 2 + int(3 * np.log(dim)))
            mu = popsize // 2

            # Weights
            w = np.log(np.arange(1, mu + 1))
            w -= w[-1]
            w /= w.sum()
            mu_eff = 1.0 / np.sum(w ** 2)

            # Parameters
            c_sigma = (mu_eff + 2) / (dim + mu_eff + 5)
            d_sigma = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (dim + 1)) - 1) + c_sigma
            c_c = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
            c_mu = (2 + (mu_eff - 2) / dim) / ((dim + 2 + 2 * mu_eff / dim) * (c_c + 1))

            C = np.eye(dim)
            C_sqrt = np.eye(dim)
            C_inv_sqrt = np.eye(dim)
            p_sigma = np.zeros(dim)
            p_c = np.zeros(dim)

            n_gen = 0
            while n_gen < n_eval and _time_ok() and calls < budget:
                if calls + popsize > budget:
                    break

                # Generate population
                offspring = []
                for _ in range(popsize):
                    z = rng.standard_normal(dim)
                    y = x + sigma * (C_sqrt @ z)
                    y = np.clip(y, -5, 5)
                    offspring.append(y)

                # Evaluate
                offspring_f = np.array([_call(y) for y in offspring])
                if not _time_ok() or calls >= budget:
                    break

                # Sort
                idx = np.argsort(offspring_f)
                offspring = np.array([offspring[i] for i in idx])
                offspring_f = offspring_f[idx]

                # Update mean
                x_new = np.sum(w[:, None] * offspring[:mu], axis=0)

                # Update paths
                diff = x_new - x
                C_inv_sqrt_diff = C_inv_sqrt @ diff / sigma
                p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * C_inv_sqrt_diff
                sigma *= np.exp(c_sigma / d_sigma * (np.linalg.norm(p_sigma) / np.sqrt(mu_eff) - 1))
                sigma = np.clip(sigma, 1e-8, 5.0)

                p_c = (1 - c_c) * p_c + np.sqrt(c_c * (2 - c_c) * mu_eff) * C_inv_sqrt_diff

                # Update covariance
                delta_heav = (1 + 2 * max(0, np.sqrt((mu_eff - 1) / (dim + 1)) - 1)) * c_c * (2 - c_c)
                rank1 = np.outer(p_c, p_c)
                rankmu = np.sum(w[:, None, None] * np.outer(offspring[:mu] - x, offspring[:mu] - x) / sigma ** 2, axis=0)
                
                # Heaviside condition
                h = 1 if np.linalg.norm(p_c) ** 2 / (c_c * (1 - c_c)) < (1 + 2 * delta_heav) * dim else 0
                
                C = (1 - c_mu) * C + c_mu * (h * rank1 + rankmu - C)

                # Periodically recompute matrix square roots
                if n_gen % 10 == 0:
                    try:
                        # Ensure C is symmetric positive definite
                        C = (C + C.T) / 2
                        C_inv = np.linalg.inv(C)
                        C_inv_sqrt = np.linalg.cholesky(C_inv).T
                        C_sqrt = np.linalg.cholesky(C)
                    except:
                        C = np.eye(dim)
                        C_inv_sqrt = np.eye(dim)
                        C_sqrt = np.eye(dim)

                x = x_new
                n_gen += 1

        # Run CMA-ES from each basin
        for i, (start, nb) in enumerate(zip(start_points, basin_budgets)):
            if not _time_ok() or calls >= budget:
                break
            # Smaller sigma for refinement
            sigma0 = 0.5 if i == 0 else 0.3
            cma_es(start, nb, sigma0)

        # ---------- Phase 3: Final Polish ----------
        if _time_ok() and calls < budget:
            remaining = budget - calls
            if remaining > 5:
                sigma = 0.1
                for _ in range(remaining):
                    if not _time_ok() or calls >= budget:
                        break
                    x = best_x + sigma * rng.standard_normal(dim)
                    x = np.clip(x, -5, 5)
                    fx = _call(x)
                    if fx < best_f:
                        sigma *= 1.1
                    else:
                        sigma *= 0.9
                    sigma = np.clip(sigma, 1e-4, 2.0)

        return best_x, best_f
# EVOLVE-BLOCK-END
