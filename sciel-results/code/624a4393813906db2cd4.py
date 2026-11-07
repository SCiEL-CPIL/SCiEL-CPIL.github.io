"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.stats import norm


class Optimizer:
    """Global surrogate (GP) + Expected Improvement, then local L-BFGS-B polish."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        rng = np.random.default_rng(42)
        dim = self.dim
        budget = self.budget

        # Phase 1: Global exploration with GP surrogate
        n_gp_calls = min(120, budget // 4)
        n_local_calls = budget - n_gp_calls

        X = np.empty((0, dim))
        Y = np.empty((0,))
        calls = 0

        def evaluate(x):
            nonlocal calls
            x = np.clip(np.asarray(x, dtype=float), -5.0, 5.0)
            y = func(x)
            calls += 1
            return x, y

        # Initial sampling: mix of random and structured points
        # Use a small set of deterministic + random points for good coverage
        initial_points = []
        # Center
        initial_points.append(np.zeros(dim))
        # Vertices of a smaller cube
        for i in range(min(2 ** dim, 16)):
            pt = np.array([(-4.0 if (i >> j) & 1 else 4.0) for j in range(dim)])
            initial_points.append(pt)
        # Random points
        while len(initial_points) < min(10, n_gp_calls):
            initial_points.append(rng.uniform(-4.5, 4.5, dim))

        for pt in initial_points[:n_gp_calls]:
            x, y = evaluate(pt)
            X = np.vstack([X, x.reshape(1, -1)]) if X.size > 0 else x.reshape(1, -1)
            Y = np.append(Y, y)

        # GP hyperparameters
        length_scale = 1.0
        signal_var = 1.0
        noise_var = 1e-4

        def gp_predict(X_new, X_train, Y_train):
            """Predict mean and variance using a simple RBF kernel GP."""
            # Standardize Y for numerical stability
            Y_mean = Y_train.mean()
            Y_std = Y_train.std() + 1e-8
            Y_norm = (Y_train - Y_mean) / Y_std

            # Compute kernel matrix
            K = self._rbf_kernel(X_train, X_train, length_scale) + noise_var * np.eye(len(Y_train))
            Kf = self._rbf_kernel(X_train, X_new, length_scale)
            Kff = self._rbf_kernel(X_new, X_new, length_scale)

            # Solve for alpha
            try:
                alpha = np.linalg.solve(K, Y_norm)
            except np.linalg.LinAlgError:
                alpha = np.linalg.lstsq(K, Y_norm, rcond=None)[0]

            # Predictions
            mu = Y_mean + Y_std * Kf.T @ alpha
            var = np.diag(Kff) - np.sum(Kf.T * (K @ Kf), axis=0)
            var = np.maximum(var, 1e-8)
            return mu, np.sqrt(var)

        def _rbf_kernel(self, A, B, ls):
            """Compute RBF kernel between two sets of points."""
            # ||a - b||^2 = ||a||^2 + ||b||^2 - 2 a.b
            A_sq = np.sum(A ** 2, axis=1).reshape(-1, 1)
            B_sq = np.sum(B ** 2, axis=1).reshape(1, -1)
            dist_sq = A_sq + B_sq - 2 * A @ B.T
            dist_sq = np.maximum(dist_sq, 0)
            return signal_var * np.exp(-0.5 * dist_sq / (ls ** 2))

        # Optimize GP hyperparameters using a few iterations
        best_hyp = (1.0, 1.0, 1e-4)
        best_score = -np.inf

        # Simple grid search over a few hyperparameter combinations
        for ls in [0.5, 1.0, 2.0, 3.0]:
            for sv in [0.5, 1.0, 2.0]:
                for nv in [1e-5, 1e-4, 1e-3]:
                    # Compute negative log marginal likelihood approximation
                    Y_mean = Y.mean()
                    Y_std = Y.std() + 1e-8
                    Y_norm = (Y - Y_mean) / Y_std
                    K = self._rbf_kernel(X, X, ls) + nv * np.eye(len(Y))
                    try:
                        L = np.linalg.cholesky(K)
                        alpha = np.linalg.solve(L.T, np.linalg.solve(L, Y_norm))
                        nll = 0.5 * Y_norm @ alpha + np.sum(np.log(np.diag(L))) + 0.5 * len(Y) * np.log(2 * np.pi)
                        if nll < best_score:
                            best_score = nll
                            best_hyp = (ls, sv, nv)
                    except np.linalg.LinAlgError:
                        continue

        length_scale, signal_var, noise_var = best_hyp

        # Phase 1: EI-guided exploration
        best_x = X[np.argmin(Y)]
        best_y = Y.min()

        for i in range(n_gp_calls - len(X)):
            if calls >= budget:
                break

            # Generate candidate points for EI
            n_candidates = 50
            candidates = rng.uniform(-5, 5, (n_candidates, dim))
            # Also add points near current best
            for _ in range(10):
                candidates = np.vstack([candidates, best_x + rng.uniform(-1.5, 1.5, dim)])

            candidates = np.clip(candidates, -5.0, 5.0)

            # Predict mean and std
            mu, sigma = gp_predict(candidates, X, Y)

            # Expected Improvement
            best_y_current = Y.min()
            Z = (best_y_current - mu) / (sigma + 1e-10)
            EI = (best_y_current - mu) * norm.cdf(Z) + sigma * norm.pdf(Z)
            EI = np.where(sigma > 1e-8, EI, 0)

            idx = np.argmax(EI)
            x_new, y_new = evaluate(candidates[idx])

            X = np.vstack([X, x_new.reshape(1, -1)])
            Y = np.append(Y, y_new)

            if y_new < best_y:
                best_y = y_new
                best_x = x_new

        # Phase 2: Local optimization with L-BFGS-B from the best point
        if calls < budget:
            x0 = best_x.copy()
            remaining = budget - calls

            def objective(x):
                nonlocal calls
                x = np.clip(x, -5.0, 5.0)
                y = func(x)
                calls += 1
                return y

            # Use L-BFGS-B with bounds
            bounds = [(-5.0, 5.0)] * dim
            try:
                res = minimize(objective, x0, method='L-BFGS-B', bounds=bounds,
                               options={'maxiter': max(1, remaining // 2), 'maxfun': remaining})
                if res.fun < best_y:
                    best_y = res.fun
                    best_x = res.x
            except Exception:
                pass

        # Phase 3: If calls remain, do a few more local steps with smaller steps
        if calls < budget:
            sigma_local = 0.5
            for _ in range(budget - calls):
                if calls >= budget:
                    break
                y = np.clip(best_x + sigma_local * rng.standard_normal(dim), -5, 5)
                fy = func(y)
                calls += 1
                if fy < best_y:
                    best_x, best_y = y, fy
                    sigma_local *= 1.2
                else:
                    sigma_local *= 0.8

        return best_x, best_y
# EVOLVE-BLOCK-END
