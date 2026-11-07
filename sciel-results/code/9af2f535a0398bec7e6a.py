"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Bayesian Optimization with Matern-5/2 GP surrogate and EI acquisition."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0  # 20s safety margin before 300s limit
        rng = np.random.default_rng(123)
        dim = self.dim
        budget = self.budget
        lb = -5.0
        ub = 5.0

        # Initialize with a small set of random points
        n_init = min(20, budget // 10)
        X = rng.uniform(lb, ub, (n_init, dim))
        Y = np.array([func(x) for x in X])
        best_idx = np.argmin(Y)
        best_x = X[best_idx].copy()
        best_y = Y[best_idx]

        # GP hyperparameters
        length_scale = np.ones(dim) * 2.0
        sigma_f = 1.0
        sigma_n = 1e-4

        # Matern 5/2 kernel
        def matern52(X1, X2, ls, sf):
            # Standardize by length scale
            X1s = X1 / ls
            X2s = X2 / ls
            # Squared distances
            sq_dists = np.sum(X1s ** 2, axis=1)[:, None] - 2 * X1s @ X2s.T + np.sum(X2s ** 2, axis=1)[None, :]
            sq_dists = np.maximum(sq_dists, 0.0)
            r = np.sqrt(sq_dists)
            kernel = (1.0 + np.sqrt(5.0) * r + 5.0 / 3.0 * r ** 2) * np.exp(-np.sqrt(5.0) * r)
            return sf ** 2 * kernel

        def compute_K(X_train, ls, sf):
            return matern52(X_train, X_train, ls, sf) + sigma_n * np.eye(len(X_train))

        def predict_gp(X_new, X_train, Y_train, ls, sf):
            K = compute_K(X_train, ls, sf)
            K_new = matern52(X_new, X_train, ls, sf)
            try:
                L = np.linalg.cholesky(K)
            except np.linalg.LinAlgError:
                L = np.linalg.cholesky(K + 1e-6 * np.eye(len(X_train)))
            alpha = np.linalg.solve(L.T, np.linalg.solve(L, Y_train))
            mu = K_new @ alpha
            v = np.linalg.solve(L, K_new.T)
            var = sf ** 2 - np.sum(v ** 2, axis=0)
            var = np.maximum(var, 1e-10)
            return mu, np.sqrt(var)

        def expected_improvement(mu, sigma, best_y, xi=0.01):
            # EI for minimization
            improvement = best_y - mu - xi
            with np.errstate(divide='ignore', invalid='ignore'):
                z = improvement / sigma
                ei = improvement * (1.0 + 0.0)  # placeholder
            # Use standard normal CDF and PDF
            from scipy.stats import norm
            z = improvement / sigma
            ei = improvement * norm.cdf(z) + sigma * norm.pdf(z)
            return np.maximum(ei, 0.0)

        # Adaptive budget for GP updates
        n_ei_candidates = 200
        n_gp_updates = 5  # Update GP every 5 evaluations

        eval_count = n_init
        for step in range(budget - n_init):
            if time.time() > deadline:
                break

            # Update hyperparameters periodically (simple adaptive scheme)
            if step > 0 and step % 20 == 0:
                # Simple adaptation: adjust length scales based on data spread
                for d in range(dim):
                    spread = np.ptp(X[:, d])
                    if spread > 1e-6:
                        length_scale[d] = max(0.1, min(5.0, length_scale[d] * 0.95))
                sigma_f = max(0.1, min(10.0, np.std(Y)))

            # Generate EI candidates
            X_cand = rng.uniform(lb, ub, (n_ei_candidates, dim))
            mu, sigma = predict_gp(X_cand, X, Y, length_scale, sigma_f)
            ei = expected_improvement(mu, sigma, best_y)
            next_idx = np.argmax(ei)
            x_next = X_cand[next_idx]

            # Evaluate
            y_next = func(x_next)
            eval_count += 1

            # Update best
            if y_next < best_y:
                best_y = y_next
                best_x = x_next.copy()

            # Add to training set
            X = np.vstack([X, x_next])
            Y = np.append(Y, y_next)

            # Optional: thin out training set if it gets too large
            if len(X) > 100:
                # Keep best 100 points
                idx = np.argsort(Y)[:100]
                X = X[idx]
                Y = Y[idx]

        return best_x, best_y
# EVOLVE-BLOCK-END
