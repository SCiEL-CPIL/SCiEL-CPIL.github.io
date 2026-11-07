"""Bayesian-Optimization-style optimizer for BBOB f14 with L-BFGS-B polish."""
import numpy as np
import time
from scipy.optimize import minimize
from scipy.stats import norm


class Optimizer:
    """GP-based Bayesian Optimization with L-BFGS-B local polish."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        t_start = time.time()
        deadline = t_start + 315.0  # safety margin before 330s kill
        rng = np.random.default_rng(42)
        dim = self.dim
        budget = self.budget

        # Storage
        X = np.empty((0, dim))
        Y = np.empty(0)
        n_calls = 0

        def record(x, fx):
            nonlocal X, Y, n_calls
            X = np.vstack([X, x.reshape(1, -1)])
            Y = np.append(Y, fx)
            n_calls += 1

        def time_ok():
            return time.time() < deadline

        # --- Phase 1: Latin Hypercube initialization ---
        n_init = 15
        X_init = np.empty((n_init, dim))
        for d in range(dim):
            perm = rng.permutation(n_init)
            X_init[:, d] = (perm + rng.random(n_init)) / n_init
        X_init = X_init * 10.0 - 5.0  # map to [-5, 5]
        for i in range(n_init):
            if not time_ok() or n_calls >= budget:
                break
            record(X_init[i], func(X_init[i]))

        if n_calls >= budget or not time_ok():
            return X[np.argmin(Y)], Y.min()

        # --- GP surrogate ---
        def gp_predict(X_test, X_train, Y_train, length_scale, noise):
            """Simple GP prediction with RBF kernel."""
            n_train = X_train.shape[0]
            n_test = X_test.shape[0]

            # Compute kernel matrix K(X_train, X_train)
            diff = X_train[:, None, :] - X_train[None, :, :]
            sq_dist = np.sum(diff ** 2, axis=2)
            K = np.exp(-0.5 * sq_dist / (length_scale ** 2)) + noise * np.eye(n_train)

            # Solve K * alpha = Y_train
            try:
                alpha = np.linalg.solve(K, Y_train)
            except np.linalg.LinAlgError:
                alpha = np.linalg.lstsq(K, Y_train, rcond=None)[0]

            # Predict mean and variance at X_test
            K_s = np.exp(-0.5 * np.sum((X_test[:, None, :] - X_train[None, :, :]) ** 2, axis=2) / (length_scale ** 2))
            mu = K_s @ alpha
            K_ss = np.exp(-0.5 * np.sum((X_test[:, None, :] - X_test[None, :, :]) ** 2, axis=2) / (length_scale ** 2))
            var = np.diag(K_ss) - np.sum(K_s * (K_s @ alpha), axis=1)
            var = np.maximum(var, 1e-10)
            return mu, var

        def expected_improvement(mu, var, f_best, xi=0.0):
            """EI acquisition function."""
            eps = 1e-8
            sigma = np.sqrt(var)
            z = (f_best - mu - xi) / (sigma + eps)
            ei = (f_best - mu - xi) * norm.cdf(z) + sigma * norm.pdf(z)
            ei = np.where(sigma > eps, ei, 0.0)
            return ei

        # --- Phase 2: Bayesian Optimization loop ---
        length_scale = 1.5
        noise = 1e-6
        f_best = Y.min()
        x_best = X[np.argmin(Y)]

        # Adaptive length scale: shrink as we find better values
        bo_count = 0
        while n_calls < budget and time_ok():
            # Update GP hyperparameters adaptively
            # Shrink length scale as we progress to focus locally
            progress = bo_count / max(1, budget - n_init)
            length_scale = 1.5 * (1.0 - 0.5 * progress) + 0.3
            noise = 1e-6

            # Generate candidate points for EI maximization
            # Use a combination of random points and local perturbations around best
            n_candidates = 200
            candidates = np.empty((n_candidates, dim))
            n_rand = n_candidates // 2
            n_local = n_candidates - n_rand

            # Random candidates
            candidates[:n_rand] = rng.uniform(-5, 5, (n_rand, dim))

            # Local candidates around best point
            local_sigma = 0.5 * (1.0 - 0.5 * progress) + 0.1
            candidates[n_rand:] = np.clip(x_best + local_sigma * rng.standard_normal((n_local, dim)), -5, 5)

            # Evaluate EI at candidates
            mu, var = gp_predict(candidates, X, Y, length_scale, noise)
            ei = expected_improvement(mu, var, f_best, xi=0.01)

            # Select best candidate
            idx = np.argmax(ei)
            x_new = candidates[idx]
            fx_new = func(x_new)
            record(x_new, fx_new)
            bo_count += 1

            if fx_new < f_best:
                f_best = fx_new
                x_best = x_new.copy()

            # --- Periodic L-BFGS-B polish ---
            # Every 15 BO iterations, run a local optimizer from the best point
            if bo_count % 15 == 0 and n_calls < budget - 5 and time_ok():
                def wrapper(x):
                    if not time_ok() or n_calls >= budget:
                        return f_best
                    x_clipped = np.clip(x, -5, 5)
                    fx = func(x_clipped)
                    record(x_clipped, fx)
                    return fx

                try:
                    res = minimize(wrapper, x_best, method='L-BFGS-B',
                                   bounds=[(-5, 5)] * dim,
                                   options={'maxiter': 20, 'ftol': 1e-10, 'gtol': 1e-8})
                    if res.fun < f_best:
                        f_best = res.fun
                        x_best = np.clip(res.x, -5, 5)
                except Exception:
                    pass

        return x_best, f_best
