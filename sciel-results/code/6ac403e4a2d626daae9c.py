"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.stats import norm


class Optimizer:
    """Derivative-free optimizer using GP-based Bayesian optimization with Nelder-Mead refinement."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim
        self.calls = 0

    def __call__(self, func):
        self.calls = 0
        dim = self.dim
        budget = self.budget
        rng = np.random.default_rng(42)

        best_x = None
        best_f = np.inf

        def wrapped_func(x):
            nonlocal best_x, best_f
            x = np.asarray(x, dtype=float)
            x = np.clip(x, -5.0, 5.0)
            val = float(func(x))
            self.calls += 1
            if val < best_f:
                best_f = val
                best_x = x.copy()
            return val

        # Phase 1: Initial random sampling to build GP training data
        n_init = max(10, budget // 8)
        X_samples = []
        Y_samples = []
        for _ in range(n_init):
            if self.calls >= budget:
                break
            x = rng.uniform(-5, 5, dim)
            y = wrapped_func(x)
            X_samples.append(x)
            Y_samples.append(y)

        X_train = np.array(X_samples)
        Y_train = np.array(Y_samples)

        # Phase 2: GP-based Bayesian optimization
        # Reserve last 30% of budget for Nelder-Mead refinement
        nm_budget = int(budget * 0.3)
        gp_budget = budget - nm_budget
        gp_calls_remaining = gp_budget - self.calls

        # GP hyperparameters
        length_scale = 1.0
        signal_var = 1.0
        noise_var = 1e-6
        xi = 0.01  # Exploration parameter for EI

        def rbf_kernel(X1, X2, ls, sv):
            """RBF kernel matrix."""
            sq_dists = np.sum(X1**2, axis=1)[:, None] + np.sum(X2**2, axis=1)[None, :] - 2 * X1 @ X2.T
            sq_dists = np.maximum(sq_dists, 0)
            return sv * np.exp(-0.5 * sq_dists / (ls**2))

        def gp_predict(X_new, X_train, Y_train, ls, sv, nv):
            """Gaussian process prediction with RBF kernel."""
            K = rbf_kernel(X_train, X_train, ls, sv) + nv * np.eye(len(X_train))
            K_star = rbf_kernel(X_new, X_train, ls, sv)
            try:
                K_inv = np.linalg.inv(K)
            except np.linalg.LinAlgError:
                K_inv = np.linalg.pinv(K)
            mu = K_star @ K_inv @ Y_train
            v = K_star @ K_inv
            K_ss = rbf_kernel(X_new, X_new, ls, sv)
            var = np.diag(K_ss) - np.sum(v**2, axis=1) + nv
            var = np.maximum(var, 1e-10)
            return mu, np.sqrt(var)

        def expected_improvement(mu, sigma, best_f):
            """Compute Expected Improvement."""
            sigma = np.maximum(sigma, 1e-10)
            improvement = best_f - mu
            z = improvement / sigma
            ei = improvement * norm.cdf(z) + sigma * norm.pdf(z)
            return ei

        # Adaptive length scale based on observed spread
        if len(X_train) > 1:
            diffs = X_train[0] - X_train
            dists = np.sqrt(np.sum(diffs**2, axis=1))
            length_scale = max(0.5, np.median(dists) * 0.5)
            signal_var = max(1e-4, np.var(Y_train))
            noise_var = max(1e-8, np.var(Y_train) * 0.01)

        for i in range(gp_calls_remaining):
            if self.calls >= budget:
                break

            # Generate candidate points
            n_candidates = 200
            X_candidates = rng.uniform(-5, 5, (n_candidates, dim))

            # Also add some points near current best
            if best_x is not None:
                local_candidates = best_x + rng.normal(0, length_scale * 0.5, (n_candidates // 2, dim))
                local_candidates = np.clip(local_candidates, -5, 5)
                X_candidates = np.vstack([X_candidates[:n_candidates // 2], local_candidates])

            # GP prediction
            mu, sigma = gp_predict(X_candidates, X_train, Y_train, length_scale, signal_var, noise_var)

            # Compute EI
            ei = expected_improvement(mu, sigma, best_f)

            # Select best candidate
            idx = np.argmax(ei)
            x_next = X_candidates[idx]
            y_next = wrapped_func(x_next)

            # Update training data
            X_train = np.vstack([X_train, x_next])
            Y_train = np.append(Y_train, y_next)

            # Adaptively adjust length scale
            if i % 20 == 19 and len(X_train) > 5:
                diffs = X_train[0] - X_train
                dists = np.sqrt(np.sum(diffs**2, axis=1))
                length_scale = max(0.3, np.median(dists) * 0.4)

        # Phase 3: Nelder-Mead refinement from the best point
        if self.calls < budget:
            remaining = budget - self.calls
            nm_budget_actual = min(remaining, nm_budget)

            try:
                result = minimize(
                    wrapped_func,
                    best_x,
                    method='Nelder-Mead',
                    options={
                        'maxiter': nm_budget_actual,
                        'maxfev': nm_budget_actual,
                        'xatol': 1e-8,
                        'fatol': 1e-10
                    }
                )
            except Exception:
                pass

        # Phase 4: If we still have budget, do a few more Nelder-Mead runs from different starting points
        if self.calls < budget:
            remaining = budget - self.calls
            if remaining > 30:
                n_extra = min(3, remaining // 30)
                for _ in range(n_extra):
                    if self.calls >= budget - 5:
                        break
                    x_start = best_x + rng.normal(0, 0.3, dim)
                    x_start = np.clip(x_start, -5, 5)
                    try:
                        result = minimize(
                            wrapped_func,
                            x_start,
                            method='Nelder-Mead',
                            options={
                                'maxiter': remaining // (n_extra + 1),
                                'maxfev': remaining // (n_extra + 1),
                                'xatol': 1e-8,
                                'fatol': 1e-10
                            }
                        )
                    except Exception:
                        pass

        return best_x, best_f
# EVOLVE-BLOCK-END
