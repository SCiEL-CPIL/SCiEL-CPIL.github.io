"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


class Optimizer:
    """Bayesian Optimization with Gaussian Process surrogate."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        rng = np.random.default_rng(12345)

        # Initial exploration: center + random points
        n_init = min(15, budget // 4)
        X = np.zeros((n_init, dim))
        X[0] = np.zeros(dim)  # center
        X[1:] = rng.uniform(-4, 4, (n_init - 1, dim))

        Y = np.array([func(x) for x in X])

        best_x = X[np.argmin(Y)].copy()
        best_y = Y.min()

        # GP hyperparameters - tuned for 5D Griewank-Rosenbrock
        length_scale = 1.5
        signal_var = 1.0
        noise_var = 1e-6

        def matern_kernel(X1, X2, length_scale, signal_var):
            """Matern 5/2 kernel."""
            diff = X1[:, np.newaxis, :] - X2[np.newaxis, :, :]
            sq_dist = np.sum(diff ** 2, axis=-1)
            r = np.sqrt(np.maximum(sq_dist, 0))
            sqrt3_r = np.sqrt(3) * r / length_scale
            K = signal_var * (1 + sqrt3_r + (sqrt3_r ** 2) / 3) * np.exp(-sqrt3_r)
            return K

        def gp_predict(X_new, X_obs, Y_obs, length_scale, signal_var, noise_var):
            """Predict mean and variance at X_new given observations."""
            K = matern_kernel(X_obs, X_obs, length_scale, signal_var)
            K[np.diag_indices_from(K)] += noise_var

            # Use Cholesky for numerical stability
            try:
                L = np.linalg.cholesky(K + 1e-8 * np.eye(len(X_obs)))
            except np.linalg.LinAlgError:
                try:
                    L = np.linalg.cholesky(K + 1e-6 * np.eye(len(X_obs)))
                except np.linalg.LinAlgError:
                    L = np.linalg.cholesky(K + 1e-4 * np.eye(len(X_obs)))

            K_star = matern_kernel(X_obs, X_new, length_scale, signal_var)

            # Solve L @ alpha = K_star.T
            alpha = np.linalg.solve(L.T, np.linalg.solve(L, K_star))

            mean = alpha.T @ Y_obs
            var = signal_var - np.sum(alpha ** 2, axis=0)
            var = np.maximum(var, 1e-10)
            return mean, var

        def expected_improvement(X_new, X_obs, Y_obs, length_scale, signal_var, noise_var, best_y):
            """Compute Expected Improvement acquisition."""
            mean, var = gp_predict(X_new, X_obs, Y_obs, length_scale, signal_var, noise_var)
            std = np.sqrt(var)
            eps = 1e-8
            z = (best_y - mean) / np.maximum(std, eps)
            EI = (best_y - mean) * norm.cdf(z) + std * norm.pdf(z)
            return EI

        def log_marginal_likelihood(Y_obs, X_obs, length_scale, signal_var, noise_var):
            """Compute log marginal likelihood for hyperparameter tuning."""
            K = matern_kernel(X_obs, X_obs, length_scale, signal_var)
            K[np.diag_indices_from(K)] += noise_var
            try:
                L = np.linalg.cholesky(K + 1e-8 * np.eye(len(X_obs)))
            except np.linalg.LinAlgError:
                return -1e10
            alpha = np.linalg.solve(L.T, np.linalg.solve(L, Y_obs))
            lml = -0.5 * Y_obs @ alpha - np.sum(np.log(np.diag(L))) - 0.5 * len(Y_obs) * np.log(2 * np.pi)
            return lml

        # Main optimization loop
        calls_made = n_init

        for step in range(budget - n_init):
            # Periodic hyperparameter tuning
            if step % 30 == 0 and len(Y) > 10:
                best_lml = log_marginal_likelihood(Y, X, length_scale, signal_var, noise_var)
                for _ in range(8):
                    ls_candidate = length_scale * (0.5 + rng.random() * 1.0)
                    sv_candidate = signal_var * (0.5 + rng.random() * 1.0)
                    nv_candidate = noise_var * (0.1 + rng.random() * 10)
                    lml_candidate = log_marginal_likelihood(Y, X, ls_candidate, sv_candidate, nv_candidate)
                    if lml_candidate > best_lml:
                        best_lml = lml_candidate
                        length_scale, signal_var, noise_var = ls_candidate, sv_candidate, nv_candidate

            # Generate diverse candidate points
            # Global exploration
            n_global = 100
            candidates = rng.uniform(-5, 5, (n_global, dim))

            # Local exploitation around best point with adaptive radius
            progress = step / max(1, budget - n_init)
            radius = max(0.05, 2.0 * (1 - progress) ** 1.5)
            n_local = 100
            local_pert = best_x + rng.normal(0, radius, (n_local, dim))
            local_pert = np.clip(local_pert, -5, 5)
            candidates = np.vstack([candidates, local_pert])

            # Also explore around second-best for diversity
            if len(Y) > 3:
                sorted_idx = np.argsort(Y)
                second_best = X[sorted_idx[1]]
                n_second = 50
                second_pert = second_best + rng.normal(0, radius * 0.5, (n_second, dim))
                second_pert = np.clip(second_pert, -5, 5)
                candidates = np.vstack([candidates, second_pert])

            # Compute EI for all candidates
            ei = expected_improvement(candidates, X, Y, length_scale, signal_var, noise_var, best_y)

            # Select best candidate
            idx = np.argmax(ei)
            x_new = candidates[idx]
            y_new = func(x_new)
            calls_made += 1

            # Update observations
            X = np.vstack([X, x_new])
            Y = np.append(Y, y_new)

            if y_new < best_y:
                best_y = y_new
                best_x = x_new.copy()

        return best_x, best_y
# EVOLVE-BLOCK-END
