"""Optimizer using Bayesian Optimization with GP surrogate for BBOB f22."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import norm
from scipy.optimize import minimize
from scipy.spatial.distance import cdist


class Optimizer:
    """Bayesian Optimization with Matern-5/2 GP and Expected Improvement."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        rng = np.random.default_rng(12345)

        # Time management
        start_time = time.time()
        time_limit = 280.0  # safety margin from 300s

        # Initialize with a small Latin hypercube / random set
        n_init = min(20, budget)
        X = rng.uniform(-5, 5, size=(n_init, dim))
        y = np.array([func(x) for x in X])
        calls = n_init

        # Track best
        best_idx = np.argmin(y)
        best_x = X[best_idx].copy()
        best_f = y[best_idx]

        # GP hyperparameters (fixed for speed; could tune but keep simple)
        # Matern 5/2 kernel: k(r) = (1 + sqrt(5)*r + 5*r^2/3) * exp(-sqrt(5)*r)
        # with ARD lengthscales
        ell = np.ones(dim) * 1.5  # initial lengthscales
        sigma_f = 1.0             # signal variance
        sigma_n = 1e-8            # noise variance

        def matern52(X1, X2, ell, sigma_f):
            """Compute Matern 5/2 kernel matrix."""
            D = cdist(X1, X2) / ell
            sqrt5 = np.sqrt(5.0)
            K = (1.0 + sqrt5 * D + 5.0 * D**2 / 3.0) * np.exp(-sqrt5 * D)
            return sigma_f**2 * K

        def predict(X_new, X, y, ell, sigma_f, sigma_n):
            """GP prediction: mean and variance."""
            K = matern52(X, X, ell, sigma_f)
            K += sigma_n * np.eye(len(X))
            Ks = matern52(X_new, X, ell, sigma_f)
            try:
                L = np.linalg.cholesky(K)
                alpha = np.linalg.solve(L.T, np.linalg.solve(L, y))
                mu = Ks @ alpha
                v = np.linalg.solve(L, Ks.T)
                var = sigma_f**2 - np.sum(v**2, axis=1)
                var = np.maximum(var, 1e-12)
            except np.linalg.LinAlgError:
                # Fallback if Cholesky fails
                try:
                    K_reg = K + 1e-6 * np.eye(len(X))
                    L = np.linalg.cholesky(K_reg)
                    alpha = np.linalg.solve(L.T, np.linalg.solve(L, y))
                    mu = Ks @ alpha
                    v = np.linalg.solve(L, Ks.T)
                    var = sigma_f**2 - np.sum(v**2, axis=1)
                    var = np.maximum(var, 1e-12)
                except:
                    # Last resort: use IDW-like prediction
                    dists = cdist(X_new, X)
                    w = 1.0 / (dists + 1e-8)
                    mu = (w @ y) / w.sum(axis=1)
                    var = np.ones(len(X_new)) * sigma_f**2
            return mu, var

        def expected_improvement(mu, var, f_best, xi=0.0):
            """Compute EI acquisition."""
            sigma = np.sqrt(var)
            z = (f_best - mu - xi) / sigma
            ei = (f_best - mu - xi) * norm.cdf(z) + sigma * norm.pdf(z)
            return ei

        def acquisition(x, X, y, ell, sigma_f, sigma_n, f_best, xi=0.0):
            """Negative EI for minimization."""
            mu, var = predict(x.reshape(1, -1), X, y, ell, sigma_f, sigma_n)
            ei = expected_improvement(mu[0], var[0], f_best, xi)
            return -ei

        # Main BO loop
        while calls < budget:
            # Check time
            if time.time() - start_time > time_limit:
                break

            # Determine exploration parameter xi
            # Start with some exploration, decrease over time
            progress = calls / budget
            xi = 0.5 * (1 - progress)

            # Sample candidate points for acquisition optimization
            n_candidates = 200
            candidates = rng.uniform(-5, 5, size=(n_candidates, dim))

            # Also include points near current best
            n_near = 50
            near_best = best_x + rng.normal(0, 0.5, size=(n_near, dim))
            near_best = np.clip(near_best, -5, 5)

            all_candidates = np.vstack([candidates, near_best])

            # Evaluate acquisition function
            mu, var = predict(all_candidates, X, y, ell, sigma_f, sigma_n)
            ei = expected_improvement(mu, var, best_f, xi)

            # Pick top candidates and refine with local optimization
            top_k = 5
            top_indices = np.argsort(ei)[::-1][:top_k]

            best_acq_val = -np.inf
            best_new_x = None

            for idx in top_indices:
                x0 = all_candidates[idx]

                # Local refinement using L-BFGS-B
                try:
                    res = minimize(
                        acquisition,
                        x0,
                        args=(X, y, ell, sigma_f, sigma_n, best_f, xi),
                        method='L-BFGS-B',
                        bounds=[(-5, 5)] * dim,
                        options={'maxiter': 50, 'ftol': 1e-6}
                    )
                    if res.fun < best_acq_val:
                        best_acq_val = res.fun
                        best_new_x = res.x
                except:
                    if -ei[idx] > best_acq_val:
                        best_acq_val = -ei[idx]
                        best_new_x = x0

            if best_new_x is None:
                # Fallback: pick the best from candidates
                best_new_x = all_candidates[top_indices[0]]

            # Evaluate the new point
            new_x = np.clip(best_new_x, -5, 5)
            new_f = func(new_x)
            calls += 1

            # Update data
            X = np.vstack([X, new_x.reshape(1, -1)])
            y = np.append(y, new_f)

            # Update best
            if new_f < best_f:
                best_f = new_f
                best_x = new_x.copy()

            # Adaptively update lengthscales based on progress
            # Scale lengthscales: start wider, narrow as we converge
            scale_factor = max(0.3, 1.0 - 0.5 * progress)
            ell = np.ones(dim) * 1.5 * scale_factor

            # Occasionally re-randomize if stuck
            if calls % 50 == 0 and calls > n_init:
                # Check if we've made progress recently
                recent_y = y[-20:]
                if np.min(recent_y) >= best_f - 1e-10:
                    # Stagnation: add some random restarts
                    n_restart = 3
                    restarts = rng.uniform(-5, 5, size=(n_restart, dim))
                    for rx in restarts:
                        if calls >= budget:
                            break
                        rf = func(rx)
                        calls += 1
                        X = np.vstack([X, rx.reshape(1, -1)])
                        y = np.append(y, rf)
                        if rf < best_f:
                            best_f = rf
                            best_x = rx.copy()

        return best_x, best_f
# EVOLVE-BLOCK-END
