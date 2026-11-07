"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Bayesian Optimization with Gaussian Process surrogate for BBOB-style functions."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        t0 = time.time()
        deadline = t0 + 295.0  # Safety margin before 330s kill
        n = self.dim
        budget = self.budget
        rng = np.random.default_rng(42)

        # Time check helper
        def time_ok():
            return time.time() < deadline

        # Safe evaluation wrapper
        def safe_call(x):
            try:
                fx = func(np.asarray(x, dtype=float))
                if not np.isfinite(fx):
                    return np.inf
                return float(fx)
            except Exception:
                return np.inf

        # Initial random samples
        n_init = min(15, budget)
        X = rng.uniform(-5, 5, size=(n_init, n))
        Y = np.array([safe_call(x) for x in X])
        best_idx = np.argmin(Y)
        best_x = X[best_idx].copy()
        best_y = Y[best_idx]

        calls = n_init

        # Gaussian Process hyperparameters
        # We'll use a simple RBF kernel with learned length scales
        # For simplicity and speed, we use a fixed kernel but adapt sigma_f

        # GP surrogate using scipy
        from scipy.optimize import minimize as scipy_minimize

        def gp_predict(X_train, Y_train, X_test, length_scales, sigma_f, sigma_n):
            """
            Predict mean and variance using GP with RBF kernel.
            """
            # Standardize for numerical stability
            X_train_mean = X_train.mean(axis=0)
            X_train_std = X_train.std(axis=0) + 1e-10
            X_train_norm = (X_train - X_train_mean) / X_train_std
            X_test_norm = (X_test - X_train_mean) / X_train_std

            # RBF kernel
            def rbf_kernel(X1, X2, ls):
                # ls is length scale vector
                X1_scaled = X1 / ls
                X2_scaled = X2 / ls
                # Compute squared distances
                sq_dists = np.sum(X1_scaled ** 2, axis=1)[:, None] + \
                           np.sum(X2_scaled ** 2, axis=1)[None, :] - \
                           2 * X1_scaled @ X2_scaled.T
                sq_dists = np.maximum(sq_dists, 0)
                return sigma_f ** 2 * np.exp(-0.5 * sq_dists)

            K = rbf_kernel(X_train_norm, X_train_norm, length_scales)
            K += sigma_n ** 2 * np.eye(len(Y_train))
            K_star = rbf_kernel(X_train_norm, X_test_norm, length_scales)

            try:
                L = np.linalg.cholesky(K)
                alpha = np.linalg.solve(L.T, np.linalg.solve(L, Y_train))
                mu = K_star.T @ alpha
                v = np.linalg.solve(L, K_star)
                var = sigma_f ** 2 - np.sum(v ** 2, axis=0)
                var = np.maximum(var, 1e-10)
                return mu, var
            except np.linalg.LinAlgError:
                # Fallback: use diagonal
                mu = np.mean(Y_train) * np.ones(len(X_test))
                var = np.var(Y_train) * np.ones(len(X_test))
                return mu, var

        # Expected Improvement acquisition function
        def expected_improvement(mu, var, best_y, xi=0.01):
            """Compute EI for minimization."""
            with np.errstate(divide='ignore', invalid='ignore'):
                sigma = np.sqrt(var)
                z = (best_y - mu - xi) / (sigma + 1e-10)
                ei = (best_y - mu - xi) * np.maximum(z, 0) + sigma * (
                        1 - np.exp(-0.5 * z ** 2)) / np.sqrt(2 * np.pi)
            ei = np.where(sigma > 1e-8, ei, 0.0)
            return ei

        # Optimize GP hyperparameters
        def optimize_hyperparams(X_train, Y_train):
            """Find optimal length scales and sigma_f."""
            # Standardize
            X_mean = X_train.mean(axis=0)
            X_std = X_train.std(axis=0) + 1e-10

            def neg_log_likelihood(params):
                try:
                    ls = np.exp(params[:n])
                    log_sigma_f = params[n]
                    log_sigma_n = params[n + 1]
                    sigma_f = np.exp(log_sigma_f)
                    sigma_n = np.exp(log_sigma_n)

                    X_norm = (X_train - X_mean) / X_std

                    def rbf_kernel(X1, X2, ls):
                        X1_scaled = X1 / ls
                        X2_scaled = X2 / ls
                        sq_dists = np.sum(X1_scaled ** 2, axis=1)[:, None] + \
                                   np.sum(X2_scaled ** 2, axis=1)[None, :] - \
                                   2 * X1_scaled @ X2_scaled.T
                        sq_dists = np.maximum(sq_dists, 0)
                        return sigma_f ** 2 * np.exp(-0.5 * sq_dists)

                    K = rbf_kernel(X_norm, X_norm, ls)
                    K += sigma_n ** 2 * np.eye(len(Y_train))

                    try:
                        L = np.linalg.cholesky(K)
                    except np.linalg.LinAlgError:
                        return 1e10

                    alpha = np.linalg.solve(L.T, np.linalg.solve(L, Y_train))
                    nll = 0.5 * Y_train @ alpha + \
                          np.sum(np.log(np.diag(L))) + \
                          0.5 * len(Y_train) * np.log(2 * np.pi)
                    return nll
                except Exception:
                    return 1e10

            # Initial guess
            x0 = np.zeros(n + 2)
            x0[:n] = np.log(1.0)  # length scale ~1
            x0[n] = np.log(np.std(Y_train) + 1e-6)  # sigma_f
            x0[n + 1] = np.log(1e-4)  # sigma_n

            result = scipy_minimize(neg_log_likelihood, x0, method='L-BFGS-B',
                                    options={'maxiter': 50})
            if result.success:
                ls = np.exp(result.x[:n])
                sigma_f = np.exp(result.x[n])
                sigma_n = np.exp(result.x[n + 1])
            else:
                ls = np.ones(n)
                sigma_f = np.std(Y_train) + 1e-6
                sigma_n = 1e-4
            return ls, sigma_f, sigma_n

        # Main BO loop
        while calls < budget and time_ok():
            # Update GP hyperparameters periodically
            if calls % 10 == 0 or calls == n_init:
                try:
                    length_scales, sigma_f, sigma_n = optimize_hyperparams(X, Y)
                except Exception:
                    length_scales = np.ones(n)
                    sigma_f = np.std(Y) + 1e-6
                    sigma_n = 1e-4

            # Candidate points for acquisition maximization
            n_candidates = 200
            candidates = rng.uniform(-5, 5, size=(n_candidates, n))
            # Also include points near the current best
            for _ in range(50):
                candidates = np.vstack([candidates, best_x + rng.normal(0, 0.5, n)])
            candidates = np.clip(candidates, -5, 5)

            # Predict mean and variance
            mu, var = gp_predict(X, Y, candidates, length_scales, sigma_f, sigma_n)

            # Compute EI
            ei = expected_improvement(mu, var, best_y)
            next_idx = np.argmax(ei)
            next_x = candidates[next_idx]

            # Evaluate
            next_y = safe_call(next_x)
            calls += 1

            # Update
            X = np.vstack([X, next_x])
            Y = np.append(Y, next_y)

            if next_y < best_y:
                best_x = next_x.copy()
                best_y = next_y

        # Final polish with local search if time permits
        if time_ok() and calls < budget:
            # Use Nelder-Mead from the best point
            from scipy.optimize import minimize as scipy_minimize2
            try:
                result = scipy_minimize2(safe_call, best_x, method='Nelder-Mead',
                                         options={'maxiter': 50, 'xatol': 1e-8, 'fatol': 1e-10})
                if result.fun < best_y:
                    best_x = result.x
                    best_y = result.fun
            except Exception:
                pass

        return best_x, best_y
# EVOLVE-BLOCK-END
