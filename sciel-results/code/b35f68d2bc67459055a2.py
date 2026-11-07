import numpy as np
import time
from scipy.optimize import minimize
from scipy.stats import norm

class Optimizer:
    """Bayesian-optimization hybrid with Gaussian Process surrogate and L-BFGS-B polish."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        rng = np.random.default_rng(42)

        # Safety margin: 330s limit, stop at 320s
        deadline = time.time() + 320

        best_x = None
        best_f = np.inf
        calls = 0

        bounds = [(-5.0, 5.0) for _ in range(dim)]
        lower = np.full(dim, -5.0)
        upper = np.full(dim, 5.0)

        # Wrapper to track calls and update best
        def wrapped_func(x):
            nonlocal calls, best_f, best_x
            if calls >= budget or time.time() > deadline:
                return np.inf
            x = np.clip(np.asarray(x, dtype=float), -5, 5)
            if x.shape != (dim,):
                x = x.reshape(-1)
            if len(x) != dim:
                x = x[:dim]
            val = func(x)
            if not np.isfinite(val):
                return 1e10
            calls += 1
            if val < best_f:
                best_f = val
                best_x = x.copy()
            return val

        def polish(x_start, max_iter=30, maxfev=None):
            """Run L-BFGS-B polish from a starting point."""
            nonlocal calls, best_f, best_x
            if calls >= budget or time.time() > deadline:
                return
            if maxfev is None:
                maxfev = max(10, (budget - calls) // 3)
            try:
                res = minimize(
                    wrapped_func,
                    x_start.copy(),
                    method='L-BFGS-B',
                    bounds=bounds,
                    options={
                        'maxiter': max_iter,
                        'maxfev': maxfev,
                        'ftol': 1e-14,
                        'gtol': 1e-10,
                        'disp': False
                    }
                )
            except Exception:
                pass

        # Phase 1: Initial sampling - mix of random and structured points
        # Use a smaller initial set to leave more budget for BO
        n_samples = min(30, budget // 10)

        # Add some structured initial points (corners, center, etc.)
        structured_points = [
            np.zeros(dim),
            np.full(dim, -5.0),
            np.full(dim, 5.0),
            rng.uniform(-5, 5, dim),
            rng.uniform(-5, 5, dim),
        ]
        # Add random points
        for i in range(n_samples - len(structured_points)):
            structured_points.append(rng.uniform(-5, 5, dim))

        for x in structured_points:
            if calls >= budget or time.time() > deadline:
                break
            wrapped_func(x)

        if best_x is None:
            return np.zeros(dim), np.inf

        # Phase 2: Quick polish from current best
        polish(best_x, max_iter=20, maxfev=max(15, (budget - calls) // 5))

        if calls >= budget or time.time() > deadline:
            return best_x, best_f

        # Phase 3: Bayesian Optimization loop
        # Maintain a set of observed points
        X_obs = [best_x.copy()]
        y_obs = [best_f]
        # Add the initial samples we evaluated
        # We need to track all evaluated points
        # Let's restructure: track all points in a list
        # Actually, let's just rebuild from what we know

        # For simplicity, let's do a simpler BO approach:
        # Keep a set of candidate points, evaluate the most promising ones

        # GP-based BO
        def gp_predict(X, X_obs, y_obs, length_scale=1.0, noise=1e-4):
            """Simple GP prediction with RBF kernel."""
            n = len(X_obs)
            m = len(X)

            # Compute kernel matrix K
            K = np.zeros((n, n))
            for i in range(n):
                for j in range(n):
                    d2 = np.sum((X_obs[i] - X_obs[j]) ** 2)
                    K[i, j] = np.exp(-d2 / (2 * length_scale ** 2))
            K[np.diag_indices(n)] += noise

            # Compute kernel vector K*
            Ks = np.zeros((n, m))
            for i in range(n):
                for j in range(m):
                    d2 = np.sum((X_obs[i] - X[j]) ** 2)
                    Ks[i, j] = np.exp(-d2 / (2 * length_scale ** 2))

            try:
                L = np.linalg.cholesky(K)
            except np.linalg.LinAlgError:
                L = np.linalg.cholesky(K + 1e-6 * np.eye(n))

            alpha = np.linalg.solve(L.T, np.linalg.solve(L, y_obs))
            mu = Ks.T @ alpha
            v = np.linalg.solve(L, Ks)
            cov = np.eye(m) - v.T @ v
            std = np.sqrt(np.maximum(np.diag(cov), 1e-12))

            return mu, std

        def expected_improvement(mu, std, y_best, xi=0.01):
            """Expected Improvement acquisition function."""
            improvement = y_best - mu - xi
            z = improvement / (std + 1e-12)
            ei = improvement * norm.cdf(z) + std * norm.pdf(z)
            return np.maximum(ei, 0)

        # BO loop
        n_bo_iters = min(50, (budget - calls) // 2)

        for it in range(n_bo_iters):
            if calls >= budget or time.time() > deadline:
                break

            # Generate candidate points
            n_candidates = 50
            X_cand = np.zeros((n_candidates, dim))
            # Mix of random and local perturbations
            for j in range(n_candidates):
                if j < n_candidates // 2:
                    # Random in box
                    X_cand[j] = rng.uniform(-5, 5, dim)
                elif j < 3 * n_candidates // 4:
                    # Local perturbation around best
                    X_cand[j] = best_x + rng.normal(0, 0.5, dim)
                    X_cand[j] = np.clip(X_cand[j], -5, 5)
                else:
                    # Perturbation around other good points
                    idx = rng.integers(0, len(X_obs))
                    X_cand[j] = X_obs[idx] + rng.normal(0, 1.0, dim)
                    X_cand[j] = np.clip(X_cand[j], -5, 5)

            # GP prediction
            mu, std = gp_predict(X_cand, X_obs, y_obs, length_scale=1.0, noise=1e-4)

            # EI acquisition
            ei = expected_improvement(mu, std, best_f, xi=0.01)

            # Select top candidates
            top_k = min(3, n_candidates)
            top_idx = np.argsort(-ei)[:top_k]

            for idx in top_idx:
                if calls >= budget or time.time() > deadline:
                    break
                x_new = X_cand[idx]
                f_new = wrapped_func(x_new)
                if np.isfinite(f_new):
                    X_obs.append(x_new.copy())
                    y_obs.append(f_new)

            # Occasionally polish the current best
            if it % 5 == 0 and calls < budget - 20:
                polish(best_x, max_iter=10, maxfev=max(10, (budget - calls) // 10))

        # Phase 4: Final polish if budget remains
        if calls < budget and best_x is not None and time.time() < deadline:
            remaining = budget - calls
            if remaining > 10:
                try:
                    minimize(
                        wrapped_func,
                        best_x.copy(),
                        method='L-BFGS-B',
                        bounds=bounds,
                        options={
                            'maxiter': 50,
                            'maxfev': remaining,
                            'ftol': 1e-14,
                            'gtol': 1e-10,
                            'disp': False
                        }
                    )
                except Exception:
                    pass

        return best_x, best_f
