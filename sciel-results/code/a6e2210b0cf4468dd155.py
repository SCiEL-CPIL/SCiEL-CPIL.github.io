"""Bayesian-optimization-guided optimizer for BBOB f12 (Bent cigar)."""
import numpy as np
import time
from scipy.optimize import minimize
from scipy.stats import norm


class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        deadline = time.time() + 315.0  # safety margin before 330s limit

        rng = np.random.default_rng(42)

        # --- Initialization: evaluate a small set of starting points ---
        # Start at origin plus a few random points to capture the rotated valley
        init_points = [np.zeros(dim)]
        for _ in range(4):
            init_points.append(rng.uniform(-5, 5, dim))

        X_obs = np.array(init_points)
        F_obs = np.array([func(x) for x in init_points])
        calls = len(init_points)

        best_idx = np.argmin(F_obs)
        best_x = X_obs[best_idx].copy()
        best_f = F_obs[best_idx]

        # --- GP surrogate parameters ---
        # Matérn 3/2 kernel: K(x, y) = sigma_f^2 * (1 + sqrt(3)*r) * exp(-sqrt(3)*r)
        # where r = ||x - y|| / l
        sigma_f = 1.0  # signal variance (will be fit)
        length_scale = 2.0  # initial length scale
        noise = 1e-6  # observation noise

        def matern32_kernel(X1, X2, length_scale, sigma_f):
            """Compute Matérn 3/2 kernel matrix."""
            # Pairwise squared distances
            diff = X1[:, np.newaxis, :] - X2[np.newaxis, :, :]
            r = np.sqrt(np.sum(diff ** 2, axis=-1) + 1e-12)
            sr = np.sqrt(3.0) * r / length_scale
            K = sigma_f ** 2 * (1.0 + sr) * np.exp(-sr)
            return K

        def log_likelihood(params, X, y):
            """Negative log marginal likelihood for GP."""
            try:
                sigma_f = np.exp(params[0])
                length_scale = np.exp(params[1])
                noise = np.exp(params[2])

                K = matern32_kernel(X, X, length_scale, sigma_f)
                K += noise * np.eye(len(X))

                # Cholesky decomposition
                try:
                    L = np.linalg.cholesky(K)
                except np.linalg.LinAlgError:
                    return 1e10

                alpha = np.linalg.solve(L.T, np.linalg.solve(L, y))
                log_det = 2.0 * np.sum(np.log(np.diag(L)))

                nll = 0.5 * y @ alpha + 0.5 * log_det + 0.5 * len(y) * np.log(2 * np.pi)
                return nll
            except Exception:
                return 1e10

        def gp_predict(X_new, X_train, y_train, params):
            """Predict mean and variance at X_new given training data."""
            sigma_f = np.exp(params[0])
            length_scale = np.exp(params[1])
            noise = np.exp(params[2])

            K = matern32_kernel(X_train, X_train, length_scale, sigma_f)
            K += noise * np.eye(len(X_train))
            K_new = matern32_kernel(X_new, X_train, length_scale, sigma_f)

            try:
                L = np.linalg.cholesky(K)
            except np.linalg.LinAlgError:
                # Fallback: return high variance
                return np.zeros(len(X_new)), np.full(len(X_new), sigma_f ** 2)

            v = np.linalg.solve(L.T, np.linalg.solve(L, y_train))
            mean = K_new @ v
            var = sigma_f ** 2 - np.sum(K_new @ np.linalg.solve(L, K_new.T), axis=1)
            var = np.maximum(var, 1e-10)
            return mean, var

        def expected_improvement(X_cand, y_best, mean, var, xi=0.01):
            """Expected Improvement acquisition function."""
            var = np.maximum(var, 1e-10)
            sigma = np.sqrt(var)
            z = (y_best - mean - xi) / sigma
            EI = (y_best - mean - xi) * norm.cdf(z) + sigma * norm.pdf(z)
            return EI

        # --- Main optimization loop ---
        # Phase 1: Fit GP and use EI for a while
        # Phase 2: Switch to local refinement with CMA-ES or Nelder-Mead

        n_gp_refit = 8  # refit GP every 8 evaluations
        gp_count = 0

        # Initial GP parameter fit
        params = np.array([np.log(1.0), np.log(2.0), np.log(1e-6)])

        # Try to fit GP parameters using a few iterations of L-BFGS
        try:
            res = minimize(log_likelihood, params, args=(X_obs, F_obs),
                           method='L-BFGS-B',
                           bounds=[(-5, 5), (-3, 3), (-12, -3)],
                           options={'maxiter': 20})
            if res.success:
                params = res.x
        except Exception:
            pass

        while calls < budget and time.time() < deadline:
            # Refit GP parameters periodically
            gp_count += 1
            if gp_count % n_gp_refit == 0 and len(X_obs) >= 10:
                try:
                    res = minimize(log_likelihood, params, args=(X_obs, F_obs),
                                   method='L-BFGS-B',
                                   bounds=[(-5, 5), (-3, 3), (-12, -3)],
                                   options={'maxiter': 15})
                    if res.success:
                        params = res.x
                except Exception:
                    pass

            # Predict mean and variance at a set of candidate points
            # Use a mix of:
            # 1. Points around the current best (exploitation)
            # 2. Random points (exploration)
            # 3. Points that maximize EI

            n_cand = 30
            candidates = []

            # Exploitation: points near best_x
            for _ in range(10):
                candidates.append(best_x + 0.5 * rng.standard_normal(dim))

            # Exploration: random points in the box
            for _ in range(10):
                candidates.append(rng.uniform(-5, 5, dim))

            # EI-based: evaluate EI at a grid of points
            grid_size = 3
            grid_points = []
            for i in range(grid_size ** dim):
                point = []
                idx = i
                for d_idx in range(dim):
                    val = -5.0 + (idx % grid_size) * (10.0 / (grid_size - 1))
                    point.append(val)
                    idx //= grid_size
                grid_points.append(point)
            candidates.extend(grid_points)

            candidates = np.array(candidates)
            # Clip to box
            candidates = np.clip(candidates, -5, 5)

            # Predict at candidates
            mean, var = gp_predict(candidates, X_obs, F_obs, params)

            # Compute EI
            EI = expected_improvement(candidates, best_f, mean, var, xi=0.01)

            # Select the point with highest EI
            best_cand_idx = np.argmax(EI)
            x_new = candidates[best_cand_idx].copy()

            # Evaluate
            f_new = func(x_new)
            calls += 1

            # Update best
            if f_new < best_f:
                best_f = f_new
                best_x = x_new.copy()

            # Add to observation set
            X_obs = np.vstack([X_obs, x_new])
            F_obs = np.append(F_obs, f_new)

            # If we're close to convergence, switch to local refinement
            if calls > budget * 0.6:
                # Use Nelder-Mead to polish around best_x
                try:
                    def wrapper(x):
                        nonlocal calls
                        if calls >= budget or time.time() > deadline:
                            return best_f
                        x = np.clip(x, -5, 5)
                        fx = func(x)
                        calls += 1
                        return fx

                    res = minimize(wrapper, best_x, method='Nelder-Mead',
                                   options={'maxiter': 30, 'xatol': 1e-8, 'fatol': 1e-10})
                    if res.fun < best_f:
                        best_f = res.fun
                        best_x = np.clip(res.x, -5, 5)
                except Exception:
                    pass

        return best_x, best_f
