"""Surrogate-assisted Bayesian Optimization with GP and EI acquisition for BBOB f21."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.stats import norm


class Optimizer:
    """Surrogate-assisted Bayesian Optimization with Gaussian Process and EI."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def _kernel(self, X1, X2, length_scale, variance):
        """Matern 5/2 kernel."""
        d = X1.shape[1]
        # Compute squared distances with ARD
        diff = X1[:, :, None] - X2[None, :, :]  # (n1, d, n2)
        # Normalize by length scales
        diff2 = (diff ** 2) / (length_scale ** 2)  # (n1, d, n2)
        r2 = np.sum(diff2, axis=1)  # (n1, n2)
        r = np.sqrt(r2)
        sqrt5r = np.sqrt(5.0) * r
        return variance * (1.0 + sqrt5r + r2 * 5.0 / 3.0) * np.exp(-sqrt5r)

    def _gp_loglik(self, X, y, length_scale, variance, noise):
        """Compute GP log-likelihood."""
        n = len(X)
        K = self._kernel(X, X, length_scale, variance) + noise * np.eye(n)
        K += 1e-8 * np.eye(n)
        try:
            L = np.linalg.cholesky(K)
            alpha = np.linalg.solve(L.T, np.linalg.solve(L, y))
            ll = -0.5 * y @ alpha - np.sum(np.log(np.diag(L))) - 0.5 * n * np.log(2 * np.pi)
            return ll
        except:
            return -np.inf

    def _fit_gp(self, X, y):
        """Fit GP hyperparameters."""
        n = len(X)
        d = X.shape[1]
        
        # Normalize y
        y_mean = np.mean(y)
        y_std = max(np.std(y), 1e-8)
        y_norm = (y - y_mean) / y_std

        best_ll = -np.inf
        best_ls = np.ones(d)
        best_var = 1.0
        best_noise = 1e-4

        # Grid search over hyperparameters
        ls_values = [0.5, 1.0, 2.0, 3.0]
        var_values = [0.5, 1.0, 2.0]
        noise_values = [1e-5, 1e-4, 1e-3]

        for ls_val in ls_values:
            ls_try = np.ones(d) * ls_val
            for var_try in var_values:
                for noise_try in noise_values:
                    ll = self._gp_loglik(X, y_norm, ls_try, var_try, noise_try)
                    if ll > best_ll:
                        best_ll = ll
                        best_ls = ls_try.copy()
                        best_var = var_try
                        best_noise = noise_try

        # Refine with coordinate ascent
        for _ in range(3):
            for ls_mult in [0.7, 0.85, 1.0, 1.15, 1.3]:
                ls_try = best_ls * ls_mult
                ll = self._gp_loglik(X, y_norm, ls_try, best_var, best_noise)
                if ll > best_ll:
                    best_ll = ll
                    best_ls = ls_try.copy()

            for var_mult in [0.7, 0.85, 1.0, 1.15, 1.3]:
                var_try = best_var * var_mult
                ll = self._gp_loglik(X, y_norm, best_ls, var_try, best_noise)
                if ll > best_ll:
                    best_ll = ll
                    best_var = var_try

            for noise_try in [1e-5, 1e-4, 1e-3]:
                ll = self._gp_loglik(X, y_norm, best_ls, best_var, noise_try)
                if ll > best_ll:
                    best_ll = ll
                    best_noise = noise_try

        return best_ls, best_var, best_noise, y_mean, y_std

    def _gp_predict(self, X_new, X_train, y_train, length_scale, variance, noise):
        """GP prediction."""
        n = len(X_train)
        K = self._kernel(X_train, X_train, length_scale, variance) + noise * np.eye(n)
        K += 1e-8 * np.eye(n)
        K_inv = np.linalg.inv(K)
        K_star = self._kernel(X_new, X_train, length_scale, variance)
        mu = K_star @ K_inv @ y_train
        K_ss = self._kernel(X_new, X_new, length_scale, variance)
        var = np.diag(K_ss - K_star @ K_inv @ K_star.T)
        var = np.maximum(var, 1e-10)
        return mu, np.sqrt(var)

    def _ei(self, x, X_train, y_train, length_scale, variance, noise, y_mean, y_std, best_y):
        """Expected Improvement (negative for minimization)."""
        x = x.reshape(1, -1)
        mu, sigma = self._gp_predict(x, X_train, y_train, length_scale, variance, noise)
        mu = mu[0] * y_std + y_mean
        sigma = sigma[0] * y_std
        if sigma < 1e-10:
            return 0.0
        z = (mu - best_y) / sigma
        ei = (best_y - mu) * norm.cdf(z) + sigma * norm.pdf(z)
        return -ei

    def __call__(self, func):
        rng = np.random.default_rng(42)
        dim = self.dim
        budget = self.budget

        # Phase 1: Initial sampling with Latin Hypercube for better coverage
        n_init = min(25, budget // 10)
        X_list = []
        y_list = []

        # Latin Hypercube Sampling
        if n_init > 0:
            lhs_samples = []
            for i in range(n_init):
                x = (np.arange(n_init) + rng.uniform(0, 1, dim)) / n_init * 10 - 5
                lhs_samples.append(x)
            # Shuffle
            perm = rng.permutation(n_init)
            lhs_samples = [lhs_samples[i] for i in perm]

        for i in range(n_init):
            if i < len(lhs_samples):
                x = lhs_samples[i]
            else:
                x = rng.uniform(-5, 5, dim)
            y = func(x)
            X_list.append(x)
            y_list.append(y)

        X = np.array(X_list)
        y = np.array(y_list)
        best_idx = np.argmin(y)
        best_x = X[best_idx].copy()
        best_y = y[best_idx]
        calls_made = n_init

        # Phase 2: GP-based Bayesian Optimization
        n_local_search = 40
        gp_budget = budget - n_init - n_local_search

        for gp_iter in range(min(15, gp_budget)):
            if calls_made >= budget:
                break

            # Fit GP
            try:
                length_scale, variance, noise, y_mean, y_std = self._fit_gp(X, y)
            except:
                x_new = rng.uniform(-5, 5, dim)
                y_new = func(x_new)
                X_list.append(x_new)
                y_list.append(y_new)
                calls_made += 1
                if y_new < best_y:
                    best_x = x_new.copy()
                    best_y = y_new
                continue

            # Optimize EI acquisition using multiple random starts
            n_restarts = 30
            best_acq = np.inf
            best_x_new = None

            # Start points: best point, random points, and points near top candidates
            top_k = min(3, len(X_list))
            top_indices = np.argsort(y)[:top_k]

            for r in range(n_restarts):
                if r == 0:
                    x0 = best_x.copy()
                elif r == 1:
                    x0 = rng.uniform(-5, 5, dim)
                elif r == 2:
                    x0 = X[top_indices[0]].copy() + rng.normal(0, 0.3, dim)
                    x0 = np.clip(x0, -5, 5)
                elif r == 3:
                    x0 = X[top_indices[min(1, len(top_indices)-1)]].copy() + rng.normal(0, 0.5, dim)
                    x0 = np.clip(x0, -5, 5)
                elif r == 4:
                    x0 = X[top_indices[min(2, len(top_indices)-1)]].copy() + rng.normal(0, 0.7, dim)
                    x0 = np.clip(x0, -5, 5)
                else:
                    # Perturb random points
                    base = X[r % len(X_list)].copy()
                    x0 = base + rng.normal(0, 0.5, dim)
                    x0 = np.clip(x0, -5, 5)

                result = minimize(
                    self._ei,
                    x0,
                    args=(X, y, length_scale, variance, noise, y_mean, y_std, best_y),
                    method='L-BFGS-B',
                    bounds=[(-5, 5)] * dim,
                    options={'maxiter': 80, 'ftol': 1e-6}
                )
                if result.fun < best_acq:
                    best_acq = result.fun
                    best_x_new = np.clip(result.x, -5, 5)

            if best_x_new is None:
                best_x_new = rng.uniform(-5, 5, dim)

            # Evaluate new point
            y_new = func(best_x_new)
            X_list.append(best_x_new)
            y_list.append(y_new)
            calls_made += 1

            if y_new < best_y:
                best_x = best_x_new.copy()
                best_y = y_new

            X = np.array(X_list)
            y = np.array(y_list)

        # Phase 3: Local (1+1)-ES around best point with adaptive sigma
        x = best_x.copy()
        fx = best_y
        sigma = 0.15
        success_count = 0

        for i in range(n_local_search):
            if calls_made >= budget:
                break
            y = np.clip(x + sigma * rng.standard_normal(dim), -5, 5)
            fy = func(y)
            calls_made += 1
            if fy < fx:
                x, fx = y, fy
                sigma *= 1.25
                success_count += 1
            else:
                sigma *= 0.8
                success_count = 0

            # Reduce sigma if too many failures
            if success_count == 0 and i > 5:
                sigma = max(sigma * 0.9, 0.01)

            if fx < best_y:
                best_x = x.copy()
                best_y = fx

        return best_x, best_y
# EVOLVE-BLOCK-END
