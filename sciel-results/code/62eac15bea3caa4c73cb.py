"""Bayesian Optimization with Gaussian Process surrogate for BBOB f16 (Weierstrass)."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Bayesian Optimization with GP surrogate and Expected Improvement acquisition."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280  # safety margin before 300s limit
        dim = self.dim
        budget = self.budget

        # State
        X = np.zeros((0, dim))
        Y = np.zeros(0)
        best_x = None
        best_f = np.inf

        # Initial exploration: space-filling + random
        n_initial = min(30, budget)
        rng = np.random.default_rng(42)

        # Use a simple Latin hypercube-like sampling for initial points
        # We'll use a combination of grid and random
        initial_points = []
        # Add corners and center
        for i in range(dim):
            for val in [-5, 0, 5]:
                pt = np.zeros(dim)
                pt[i] = val
                initial_points.append(pt)
        initial_points.append(np.zeros(dim))

        # Add random points
        n_random = n_initial - len(initial_points)
        if n_random > 0:
            rand_pts = rng.uniform(-5, 5, (n_random, dim))
            initial_points.extend(rand_pts.tolist())

        # Evaluate initial points
        for pt in initial_points:
            if time.time() > deadline:
                break
            pt = np.clip(np.array(pt, dtype=float), -5, 5)
            f = func(pt)
            X = np.vstack([X, pt.reshape(1, -1)]) if X.size else pt.reshape(1, -1)
            Y = np.append(Y, f)
            if f < best_f:
                best_f = f
                best_x = pt.copy()

        calls_made = len(Y)

        # GP hyperparameters
        # We'll use a simple GP with squared exponential kernel
        # For efficiency, we'll use a simplified approach with fixed hyperparameters
        # that work well for Weierstrass-like functions

        def gp_predict(X_train, Y_train, X_test, length_scale=1.0, sigma_f=1.0, noise=1e-6):
            """Simple GP prediction with squared exponential kernel."""
            X_train = np.atleast_2d(X_train)
            X_test = np.atleast_2d(X_test)

            # Normalize Y for numerical stability
            Y_mean = np.mean(Y_train)
            Y_std = np.std(Y_train) if np.std(Y_train) > 1e-10 else 1.0
            Y_norm = (Y_train - Y_mean) / Y_std

            # Compute kernel matrix
            def kernel(X1, X2, ls, sf):
                # Squared exponential kernel
                d2 = np.sum(X1**2, axis=1)[:, None] - 2 * X1 @ X2.T + np.sum(X2**2, axis=1)[None, :]
                d2 = np.maximum(d2, 0)
                K = (sf**2) * np.exp(-d2 / (2 * ls**2))
                return K

            K = kernel(X_train, X_train, length_scale, sigma_f)
            K[np.diag_indices_from(K)] += noise

            # Compute predictive mean and variance for test points
            K_s = kernel(X_train, X_test, length_scale, sigma_f)

            try:
                alpha = np.linalg.solve(K, Y_norm)
                mu = K_s.T @ alpha
                v = np.linalg.solve(K, K_s)
                var = sigma_f**2 - np.sum(K_s**2 * v, axis=0)
                var = np.maximum(var, 1e-10)
            except np.linalg.LinAlgError:
                # Fallback: use simple interpolation
                mu = np.full(X_test.shape[0], Y_mean)
                var = np.full(X_test.shape[0], Y_std**2)

            # Denormalize
            mu = mu * Y_std + Y_mean
            var = var * Y_std**2
            var = np.maximum(var, 1e-10)

            return mu, var

        # Acquisition function: Expected Improvement
        def expected_improvement(mu, var, best_y, xi=0.0):
            """Compute EI for minimization."""
            s = np.sqrt(var)
            z = (best_y - mu - xi) / (s + 1e-10)
            from scipy.stats import norm
            ei = (best_y - mu - xi) * norm.cdf(z) + s * norm.pdf(z)
            return np.maximum(ei, 0)

        # Main optimization loop
        while calls_made < budget:
            if time.time() > deadline:
                break

            # Number of points to propose per iteration
            n_propose = min(5, budget - calls_made)

            # Generate candidate points
            n_candidates = 200
            candidates = []

            # Mix of:
            # 1. Random points
            # 2. Points near current best
            # 3. Points from GP-guided sampling

            # Random
            rand_cands = rng.uniform(-5, 5, (n_candidates // 2, dim))
            candidates.append(rand_cands)

            # Near best
            if best_x is not None:
                near_cands = best_x + rng.standard_normal((n_candidates // 4, dim)) * 0.5
                near_cands = np.clip(near_cands, -5, 5)
                candidates.append(near_cands)

            # GP-guided: sample points and compute EI
            gp_cands = rng.uniform(-5, 5, (n_candidates // 4, dim))
            candidates.append(gp_cands)

            all_cands = np.vstack(candidates)

            # Compute EI for all candidates
            if len(X) >= 2:
                # Adaptive length scale based on data spread
                data_spread = np.std(X, axis=0)
                avg_spread = np.mean(data_spread[data_spread > 1e-6]) if np.any(data_spread > 1e-6) else 1.0
                length_scale = max(0.5, min(avg_spread, 3.0))
                sigma_f = max(np.std(Y), 0.1)

                mu, var = gp_predict(X, Y, all_cands, length_scale=length_scale, sigma_f=sigma_f)
                ei = expected_improvement(mu, var, best_f)

                # Select top candidates by EI
                top_idx = np.argsort(ei)[::-1][:n_propose]
                selected = all_cands[top_idx]
            else:
                # Not enough data for GP, use random
                selected = all_cands[:n_propose]

            # Evaluate selected points
            for pt in selected:
                if calls_made >= budget:
                    break
                if time.time() > deadline:
                    break
                pt = np.clip(np.array(pt, dtype=float), -5, 5)
                f = func(pt)
                calls_made += 1
                X = np.vstack([X, pt.reshape(1, -1)])
                Y = np.append(Y, f)
                if f < best_f:
                    best_f = f
                    best_x = pt.copy()

            # Stagnation check: if no improvement in last 50 calls, do a restart
            if calls_made >= 50 and len(Y) > 50:
                last_50_best = np.min(Y[-50:])
                prev_best = np.min(Y[:-50])
                if last_50_best >= prev_best - 1e-6:
                    # Stagnation detected, do a few random restarts
                    n_restart = min(10, budget - calls_made)
                    for _ in range(n_restart):
                        if calls_made >= budget or time.time() > deadline:
                            break
                        pt = rng.uniform(-5, 5, dim)
                        f = func(pt)
                        calls_made += 1
                        X = np.vstack([X, pt.reshape(1, -1)])
                        Y = np.append(Y, f)
                        if f < best_f:
                            best_f = f
                            best_x = pt.copy()

        return best_x, best_f
# EVOLVE-BLOCK-END
