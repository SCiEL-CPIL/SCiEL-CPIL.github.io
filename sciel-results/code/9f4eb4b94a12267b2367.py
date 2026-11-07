"""Bayesian Optimization with Gaussian Process surrogate for BBOB f8 (Rosenbrock)."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.stats import norm


class Optimizer:
    """GP-based Bayesian Optimization with Expected Improvement acquisition."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def _kernel(self, X, Y):
        """Matern 5/2 kernel + white noise."""
        # X: (n, d), Y: (m, d)
        sq_dists = np.sum(X**2, axis=1)[:, None] + np.sum(Y**2, axis=1)[None, :] - 2 * X @ Y.T
        sq_dists = np.maximum(sq_dists, 0)
        # Use a length scale that adapts to the domain
        l = 1.5
        v = sq_dists / l**2
        K = np.sqrt(5) * v + 1
        K = np.exp(-np.sqrt(K))
        return K

    def _log_likelihood(self, X, y, lengths, noise):
        """Negative log marginal likelihood for GP hyperparameters."""
        K = self._kernel_with_params(X, X, lengths)
        K[np.diag_indices_from(K)] += noise
        try:
            L = np.linalg.cholesky(K + 1e-8 * np.eye(len(y)))
        except np.linalg.LinAlgError:
            return 1e6
        alpha = np.linalg.solve(L.T, np.linalg.solve(L, y))
        nll = 0.5 * y @ alpha + np.sum(np.log(np.diag(L))) + 0.5 * len(y) * np.log(2 * np.pi)
        return nll

    def _kernel_with_params(self, X, Y, lengths):
        """Kernel with per-dimension length scales."""
        # X: (n, d), Y: (m, d), lengths: (d,)
        Xs = X / lengths
        Ys = Y / lengths
        sq_dists = np.sum(Xs**2, axis=1)[:, None] + np.sum(Ys**2, axis=1)[None, :] - 2 * Xs @ Ys.T
        sq_dists = np.maximum(sq_dists, 0)
        v = sq_dists
        K = np.sqrt(5) * v + 1
        K = np.exp(-np.sqrt(K))
        return K

    def __call__(self, func):
        rng = np.random.default_rng(42)
        dim = self.dim
        budget = self.budget

        # Initial sampling: mix of random points and some structured points
        n_init = min(20, budget // 5)
        X_list = []
        y_list = []

        # Add some structured initial points
        init_points = []
        # Center
        init_points.append(np.zeros(dim))
        # Corners of a smaller box
        for _ in range(5):
            init_points.append(rng.uniform(-3, 3, dim))
        # Some points along axes
        for _ in range(5):
            pt = np.zeros(dim)
            ax = rng.integers(0, dim)
            pt[ax] = rng.uniform(-4, 4)
            init_points.append(pt)
        # Random
        for _ in range(n_init - len(init_points)):
            init_points.append(rng.uniform(-5, 5, dim))

        for pt in init_points[:n_init]:
            pt = np.clip(pt, -5, 5)
            val = func(pt)
            X_list.append(pt)
            y_list.append(val)

        X = np.array(X_list)
        y = np.array(y_list)

        # Optimize GP hyperparameters
        lengths = np.ones(dim) * 1.5
        noise = 1e-4

        # Simple hyperparameter optimization via Nelder-Mead
        def neg_nll(params):
            log_lengths = params[:dim]
            log_noise = params[dim]
            lengths_trial = np.exp(log_lengths)
            noise_trial = np.exp(log_noise)
            return self._log_likelihood(X, y, lengths_trial, noise_trial)

        x0 = np.concatenate([np.log(lengths), np.log(noise)])
        try:
            res = minimize(neg_nll, x0, method='Nelder-Mead',
                           options={'maxiter': 50, 'xatol': 1e-3, 'fatol': 1e-3})
            if res.success:
                lengths = np.exp(res.x[:dim])
                noise = np.exp(res.x[dim])
        except Exception:
            pass

        # Ensure lengths are reasonable
        lengths = np.clip(lengths, 0.1, 5.0)
        noise = max(noise, 1e-5)

        # Main loop: alternate between acquisition and local refinement
        calls_made = n_init
        best_x = X[np.argmin(y)]
        best_y = np.min(y)

        # Track progress
        for iteration in range(10):
            if calls_made >= budget:
                break

            # Re-optimize hyperparameters if we have enough data
            if len(y) >= 15 and iteration % 3 == 0:
                def neg_nll(params):
                    log_lengths = params[:dim]
                    log_noise = params[dim]
                    lengths_trial = np.exp(log_lengths)
                    noise_trial = np.exp(log_noise)
                    return self._log_likelihood(X, y, lengths_trial, noise_trial)

                x0 = np.concatenate([np.log(lengths), np.log(noise)])
                try:
                    res = minimize(neg_nll, x0, method='Nelder-Mead',
                                   options={'maxiter': 30, 'xatol': 1e-3, 'fatol': 1e-3})
                    if res.success:
                        lengths = np.exp(res.x[:dim])
                        noise = np.exp(res.x[dim])
                except Exception:
                    pass
                lengths = np.clip(lengths, 0.1, 5.0)
                noise = max(noise, 1e-5)

            # Compute GP mean and std at candidate points
            K = self._kernel_with_params(X, X, lengths)
            K[np.diag_indices_from(K)] += noise

            try:
                L = np.linalg.cholesky(K + 1e-8 * np.eye(len(y)))
            except np.linalg.LinAlgError:
                L = np.linalg.cholesky(K + 1e-4 * np.eye(len(y)))

            alpha = np.linalg.solve(L.T, np.linalg.solve(L, y))

            def gp_predict(X_test):
                K_s = self._kernel_with_params(X_test, X, lengths)
                mean = K_s @ alpha
                v = np.linalg.solve(L, K_s.T)
                var = np.maximum(1.0 - np.sum(v**2, axis=1), 0)
                std = np.sqrt(var)
                return mean, std

            # Generate candidate points for acquisition
            # Mix of: random points, points near best, points along gradient-like directions
            n_candidates = 200
            candidates = []

            # Random candidates
            for _ in range(80):
                candidates.append(rng.uniform(-5, 5, dim))

            # Near best point
            for _ in range(60):
                candidates.append(np.clip(best_x + rng.uniform(-1.5, 1.5, dim), -5, 5))

            # Along axes from best
            for _ in range(30):
                pt = best_x.copy()
                ax = rng.integers(0, dim)
                pt[ax] += rng.uniform(-2, 2)
                candidates.append(np.clip(pt, -5, 5))

            # Interpolation between best and random
            for _ in range(30):
                other = rng.uniform(-5, 5, dim)
                t = rng.uniform(0.3, 0.7)
                candidates.append(np.clip(t * best_x + (1 - t) * other, -5, 5))

            candidates = np.array(candidates)
            mean_c, std_c = gp_predict(candidates)

            # Expected Improvement acquisition
            current_best = np.min(y)
            with np.errstate(divide='ignore', invalid='ignore'):
                z = (current_best - mean_c) / std_c
                z = np.where(std_c > 1e-10, z, 0)
                EI = std_c * (z * norm.cdf(z) + norm.pdf(z))
                EI = np.where(std_c > 1e-10, EI, 0)

            # Select top few candidates
            top_k = min(5, len(candidates))
            top_indices = np.argsort(EI)[-top_k:][::-1]

            # Evaluate the best candidate
            best_candidate = candidates[top_indices[0]]
            val = func(best_candidate)
            calls_made += 1

            X_list.append(best_candidate)
            y_list.append(val)
            X = np.array(X_list)
            y = np.array(y_list)

            if val < best_y:
                best_y = val
                best_x = best_candidate

            # If we're close to optimum, do local refinement
            # Use the GP to guide a local search
            if calls_made < budget and std_c[top_indices[0]] < 0.5:
                # Do a few local steps using gradient approximation from GP
                local_budget = min(10, budget - calls_made)
                for _ in range(local_budget):
                    if calls_made >= budget:
                        break
                    # Sample from GP posterior near best_x
                    n_local = 20
                    local_pts = np.clip(best_x + rng.normal(0, 0.5, (n_local, dim)), -5, 5)
                    mean_l, std_l = gp_predict(local_pts)
                    with np.errstate(divide='ignore', invalid='ignore'):
                        z_l = (best_y - mean_l) / std_l
                        z_l = np.where(std_l > 1e-10, z_l, 0)
                        EI_l = std_l * (z_l * norm.cdf(z_l) + norm.pdf(z_l))
                        EI_l = np.where(std_l > 1e-10, EI_l, 0)
                    best_local_idx = np.argmax(EI_l)
                    pt = local_pts[best_local_idx]
                    val = func(pt)
                    calls_made += 1
                    X_list.append(pt)
                    y_list.append(val)
                    X = np.array(X_list)
                    y = np.array(y_list)
                    if val < best_y:
                        best_y = val
                        best_x = pt

        return best_x, best_y
# EVOLVE-BLOCK-END
