"""Optimizer for BBOB f15 (Rastrigin rotated) with surrogate-guided search and ES refinement."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import norm


class Optimizer:
    """Surrogate-assisted optimizer with RBF GP and ES refinement."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        deadline = time.time() + 280.0  # safety margin before 300s limit
        rng = np.random.default_rng(42)
        dim = self.dim

        best_x = None
        best_f = np.inf
        total_budget_used = 0

        # Storage for surrogate
        X_data = []  # list of (x, f)
        
        def record(x, fx):
            nonlocal best_x, best_f, total_budget_used
            X_data.append((x.copy(), fx))
            total_budget_used += 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()

        def rbf_kernel(X1, X2, length_scale=1.0, variance=1.0):
            """Compute RBF kernel matrix between X1 and X2."""
            # X1: (n, d), X2: (m, d)
            d2 = np.sum(X1**2, axis=1)[:, None] + np.sum(X2**2, axis=1)[None, :] - 2 * X1 @ X2.T
            d2 = np.maximum(d2, 0)
            return variance * np.exp(-d2 / (2 * length_scale**2))

        def predict_gp(X_new, X_train, y_train, length_scale=1.0, variance=1.0, noise=0.1):
            """Simple GP prediction using RBF kernel."""
            K = rbf_kernel(X_train, X_train, length_scale, variance)
            K += noise * np.eye(len(X_train))
            K_star = rbf_kernel(X_new, X_train, length_scale, variance)
            try:
                L = np.linalg.cholesky(K + 1e-8 * np.eye(len(X_train)))
                alpha = np.linalg.solve(L.T, np.linalg.solve(L, y_train))
                mu = K_star @ alpha
                v = np.linalg.solve(L, K_star.T)
                var = variance - np.sum(v**2, axis=1)
                var = np.maximum(var, 1e-10)
                return mu, var
            except np.linalg.LinAlgError:
                # Fallback: simple weighted average
                weights = rbf_kernel(X_new, X_train, length_scale, variance)
                weights = weights / (weights.sum(axis=1, keepdims=True) + 1e-8)
                mu = weights @ y_train
                var = np.full(len(X_new), 1.0)
                return mu, var

        def expected_improvement(mu, sigma, f_best, xi=0.01):
            """Compute EI acquisition."""
            sigma = np.maximum(sigma, 1e-8)
            z = (f_best - xi - mu) / sigma
            ei = (f_best - xi - mu) * norm.cdf(z) + sigma * norm.pdf(z)
            ei = np.maximum(ei, 0)
            return ei

        # Phase 1: Initial random sampling
        n_init = 40
        for i in range(n_init):
            if total_budget_used >= self.budget or time.time() > deadline:
                break
            x0 = rng.uniform(-4.5, 4.5, dim)
            fx0 = func(x0)
            record(x0, fx0)

        # Phase 2: Surrogate-guided exploration
        # Use EI to select next points, refit GP periodically
        n_gp_points = 200  # how many points to use in GP phase
        gp_batch = 10  # how many points to select per batch
        length_scale = 1.5
        gp_variance = 1.0
        noise = 0.05

        for batch in range(n_gp_points // gp_batch + 1):
            if total_budget_used >= self.budget - 50:  # reserve 50 for ES
                break
            if time.time() > deadline:
                break

            # Prepare training data (use most recent points for efficiency)
            if len(X_data) < 5:
                break
            # Use up to 80 most recent points for GP to keep it fast
            max_train = min(len(X_data), 80)
            idx = range(len(X_data) - max_train, len(X_data))
            X_train = np.array([X_data[i][0] for i in idx])
            y_train = np.array([X_data[i][1] for i in idx])

            # Generate candidate points
            n_candidates = 500
            candidates = rng.uniform(-5, 5, (n_candidates, dim))

            # Predict mean and variance
            mu, var = predict_gp(candidates, X_train, y_train, length_scale, gp_variance, noise)
            sigma = np.sqrt(var)

            # Compute EI
            ei = expected_improvement(mu, sigma, best_f, xi=0.01)

            # Select top candidates
            top_idx = np.argsort(ei)[-gp_batch:]
            
            # Evaluate selected candidates
            for idx_sel in top_idx:
                if total_budget_used >= self.budget - 50:
                    break
                if time.time() > deadline:
                    break
                x_new = candidates[idx_sel]
                fx_new = func(x_new)
                record(x_new, fx_new)

            # Adapt length_scale based on improvement
            if batch % 3 == 2 and batch > 0:
                # Slightly increase exploration
                length_scale = min(length_scale * 1.1, 3.0)

        # Phase 3: Local ES refinement on the best point
        if best_x is not None and total_budget_used < self.budget and time.time() < deadline:
            remaining = self.budget - total_budget_used
            if remaining > 10:
                x = best_x.copy()
                fx = best_f
                sigma = 0.3
                no_improve = 0
                patience = 40

                for _ in range(remaining):
                    if total_budget_used >= self.budget or time.time() > deadline:
                        break

                    y = np.clip(x + sigma * rng.standard_normal(dim), -5, 5)
                    fy = func(y)
                    record(y, fy)

                    if fy < fx:
                        x, fx = y, fy
                        sigma = min(sigma * 1.15, 2.0)
                        no_improve = 0
                    else:
                        sigma *= 0.92
                        no_improve += 1

                    if no_improve >= patience:
                        # Try a few random restarts from best
                        for _ in range(min(5, remaining - total_budget_used)):
                            if total_budget_used >= self.budget or time.time() > deadline:
                                break
                            y2 = np.clip(best_x + 0.5 * rng.standard_normal(dim), -5, 5)
                            fy2 = func(y2)
                            record(y2, fy2)
                            if fy2 < best_f:
                                x, fx = y2, fy2
                                sigma = 0.3
                                no_improve = 0
                        if no_improve >= patience:
                            break

        return best_x, best_f
# EVOLVE-BLOCK-END
