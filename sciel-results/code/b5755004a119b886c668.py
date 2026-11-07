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
        rng = np.random.default_rng(2024)

        # Initial exploration
        n_init = min(12, budget // 4)
        X = np.zeros((n_init, dim))
        X[0] = np.zeros(dim)
        X[1:] = rng.uniform(-5, 5, (n_init - 1, dim))
        Y = np.array([func(x) for x in X])

        best_x = X[np.argmin(Y)]
        best_y = Y.min()

        # GP hyperparameters
        length_scale = 1.5
        signal_var = 1.0
        noise_var = 1e-6

        def gp_kernel(X1, X2, length_scale, signal_var):
            """Matern 5/2 kernel."""
            diff = X1[:, np.newaxis, :] - X2[np.newaxis, :, :]
            sq_dist = np.sum(diff ** 2, axis=-1)
            r = np.sqrt(sq_dist)
            sqrt3_r = np.sqrt(3) * r / length_scale
            K = signal_var * (1 + sqrt3_r + (sqrt3_r ** 2) / 3) * np.exp(-sqrt3_r)
            return K

        def gp_predict(X_new, X_obs, Y_obs, length_scale, signal_var, noise_var):
            """Predict mean and variance at X_new given observations."""
            K = gp_kernel(X_obs, X_obs, length_scale, signal_var)
            K[np.diag_indices_from(K)] += noise_var
            K_star = gp_kernel(X_obs, X_new, length_scale, signal_var)
            K_ss = gp_kernel(X_new, X_new, length_scale, signal_var)

            try:
                L = np.linalg.cholesky(K + 1e-8 * np.eye(len(K)))
            except np.linalg.LinAlgError:
                L = np.linalg.cholesky(K + 1e-4 * np.eye(len(K)))

            alpha = np.linalg.solve(L.T, np.linalg.solve(L, Y_obs))
            v = np.linalg.solve(L, K_star)
            mean = v.T @ alpha
            var = np.diag(K_ss) - np.sum(v ** 2, axis=0)
            var = np.maximum(var, 1e-10)
            return mean, var

        def expected_improvement(X_new, X_obs, Y_obs, length_scale, signal_var, noise_var, best_y):
            mean, var = gp_predict(X_new, X_obs, Y_obs, length_scale, signal_var, noise_var)
            std = np.sqrt(var)
            eps = 1e-8
            z = (best_y - mean) / np.maximum(std, eps)
            EI = (best_y - mean) * norm.cdf(z) + std * norm.pdf(z)
            return EI

        def log_marginal_likelihood(Y_obs, X_obs, length_scale, signal_var, noise_var):
            K = gp_kernel(X_obs, X_obs, length_scale, signal_var)
            K[np.diag_indices_from(K)] += noise_var
            try:
                L = np.linalg.cholesky(K + 1e-8 * np.eye(len(K)))
            except np.linalg.LinAlgError:
                return -1e10
            alpha = np.linalg.solve(L.T, np.linalg.solve(L, Y_obs))
            lml = -0.5 * Y_obs @ alpha - np.sum(np.log(np.diag(L))) - 0.5 * len(Y_obs) * np.log(2 * np.pi)
            return lml

        calls_made = n_init
        for step in range(budget - n_init):
            # Tune hyperparameters periodically
            if step % 30 == 0 and len(Y) > 10:
                best_lml = -np.inf
                best_params = (length_scale, signal_var, noise_var)
                for _ in range(8):
                    ls = length_scale * (0.5 + rng.random() * 1.5)
                    sv = signal_var * (0.5 + rng.random() * 1.5)
                    nv = noise_var * (0.1 + rng.random() * 20)
                    lml = log_marginal_likelihood(Y, X, ls, sv, nv)
                    if lml > best_lml:
                        best_lml = lml
                        best_params = (ls, sv, nv)
                length_scale, signal_var, noise_var = best_params

            # Generate candidates
            n_global = 120
            n_local = 80
            candidates = rng.uniform(-5, 5, (n_global, dim))

            # Adaptive local exploration
            if calls_made < budget * 0.25:
                local_scale = 1.2
            elif calls_made < budget * 0.5:
                local_scale = 0.6
            elif calls_made < budget * 0.75:
                local_scale = 0.3
            else:
                local_scale = 0.15

            local_pert = best_x + rng.normal(0, local_scale, (n_local, dim))
            local_pert = np.clip(local_pert, -5, 5)
            candidates = np.vstack([candidates, local_pert])

            # Also add points near other good solutions
            if len(Y) > 20:
                top_k = min(3, len(Y))
                top_idx = np.argsort(Y)[:top_k]
                for idx in top_idx:
                    extra = X[idx] + rng.normal(0, local_scale * 0.5, (10, dim))
                    extra = np.clip(extra, -5, 5)
                    candidates = np.vstack([candidates, extra])

            # Compute EI
            ei = expected_improvement(candidates, X, Y, length_scale, signal_var, noise_var, best_y)

            idx = np.argmax(ei)
            x_new = candidates[idx]
            y_new = func(x_new)
            calls_made += 1

            X = np.vstack([X, x_new])
            Y = np.append(Y, y_new)

            if y_new < best_y:
                best_y = y_new
                best_x = x_new

        return best_x, best_y
# EVOLVE-BLOCK-END
