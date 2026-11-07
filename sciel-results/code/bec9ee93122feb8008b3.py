"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial.distance import cdist


class Optimizer:
    """GP surrogate with Expected Improvement acquisition for BBOB f7."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        t0 = time.time()
        deadline = t0 + 280  # safety margin before 300s limit
        rng = np.random.default_rng(12345)
        dim = self.dim
        budget = self.budget

        # Data storage
        X = np.zeros((0, dim))
        Y = np.zeros(0)

        best_x = None
        best_f = np.inf
        calls_made = 0

        def add_point(x, f):
            nonlocal X, Y, best_x, best_f, calls_made
            X = np.vstack([X, x.reshape(1, -1)])
            Y = np.append(Y, f)
            calls_made += 1
            if f < best_f:
                best_f = f
                best_x = x.copy()

        def gp_predict(Xnew, length_scale=1.0, noise=1e-6):
            """Gaussian Process prediction using RBF kernel."""
            n = X.shape[0]
            # RBF kernel
            K = np.exp(-0.5 * cdist(X, X) ** 2 / length_scale ** 2) + noise * np.eye(n)
            K_new = np.exp(-0.5 * cdist(X, Xnew) ** 2 / length_scale ** 2)
            try:
                K_inv = np.linalg.inv(K)
            except np.linalg.LinAlgError:
                K_inv = np.linalg.pinv(K)
            mu = K_new.T @ K_inv @ Y
            v = K_new.T @ K_inv @ K_new
            var = np.exp(-0.5 * 0) - np.diag(v)  # kernel at same point = 1
            var = np.maximum(var, 1e-8)
            std = np.sqrt(var)
            return mu, std

        def expected_improvement(x_candidate, length_scale=1.0, noise=1e-6):
            """EI acquisition for minimization."""
            mu, std = gp_predict(x_candidate.reshape(1, -1), length_scale, noise)
            mu = mu[0]
            std = std[0]
            if std < 1e-10:
                return 0.0
            z = (best_f - mu) / std
            from scipy.stats import norm
            ei = (best_f - mu) * norm.cdf(z) + std * norm.pdf(z)
            return -ei  # negative because we minimize

        # Phase 1: Initial random sampling to seed the GP
        n_init = min(10, budget)
        for i in range(n_init):
            if calls_made >= budget or time.time() > deadline:
                break
            x = rng.uniform(-5, 5, dim)
            f = func(x)
            add_point(x, f)

        # Adaptive length scale: start broad, narrow as we converge
        # Phase 2: GP-guided search with EI
        length_scales = [3.0, 2.0, 1.5, 1.0, 0.7, 0.5, 0.3, 0.2, 0.15, 0.1]
        ls_idx = 0

        while calls_made < budget and time.time() < deadline:
            # Update length scale periodically
            progress = calls_made / budget
            if progress > 0.7 and ls_idx < len(length_scales) - 1:
                ls_idx = min(ls_idx + 1, len(length_scales) - 1)
            ls = length_scales[ls_idx]

            # Multi-start optimization of EI
            best_ei_x = None
            best_ei_val = -np.inf

            n_starts = 5 if calls_made < 50 else 3
            for s in range(n_starts):
                if time.time() > deadline:
                    break
                # Start from best point or random
                if s == 0 and best_x is not None:
                    x0 = best_x + rng.normal(0, ls * 0.5, dim)
                elif s == 1 and best_x is not None:
                    x0 = best_x + rng.normal(0, ls, dim)
                else:
                    x0 = rng.uniform(-5, 5, dim)
                x0 = np.clip(x0, -5, 5)

                res = minimize(
                    expected_improvement,
                    x0,
                    args=(ls, 1e-6),
                    method='L-BFGS-B',
                    bounds=[(-5, 5)] * dim,
                    options={'maxiter': 50, 'ftol': 1e-6}
                )
                x_opt = np.clip(res.x, -5, 5)
                ei_val = -expected_improvement(x_opt, ls, 1e-6)
                if ei_val > best_ei_val:
                    best_ei_val = ei_val
                    best_ei_x = x_opt

            if best_ei_x is None:
                break

            # Evaluate the best EI point
            f = func(best_ei_x)
            add_point(best_ei_x, f)

        return best_x, best_f
# EVOLVE-BLOCK-END
