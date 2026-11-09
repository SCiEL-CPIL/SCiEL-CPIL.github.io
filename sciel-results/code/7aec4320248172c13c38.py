"""Surrogate-assisted Bayesian Optimization with Expected Improvement for Rastrigin f15."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Surrogate-assisted optimization using a Gaussian Process model and Expected Improvement."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290  # safety margin before 330s kill
        dim = self.dim
        budget = self.budget
        rng = np.random.default_rng(42)

        # Track evaluations
        X = []  # list of evaluated points
        Y = []  # list of function values
        calls_made = 0
        best_x = None
        best_f = np.inf

        def evaluate(x):
            nonlocal calls_made, best_f, best_x
            x_clipped = np.clip(x, -5, 5)
            f_val = func(x_clipped)
            calls_made += 1
            X.append(x_clipped.copy())
            Y.append(f_val)
            if f_val < best_f:
                best_f = f_val
                best_x = x_clipped.copy()
            return f_val

        # Phase 1: Initial exploration with a set of well-spread points
        # Use a Latin hypercube-like spread for initial samples
        n_initial = 30
        for i in range(n_initial):
            if calls_made >= budget or time.time() > deadline:
                break
            x = rng.uniform(-5, 5, dim)
            evaluate(x)

        if calls_made >= budget or time.time() > deadline:
            return best_x, best_f

        # GP parameters
        # We'll use a simple RBF kernel with automatic hyperparameter fitting via a few restarts
        length_scale = 1.0
        noise_var = 1e-6
        signal_var = 10.0

        def kernel(X1, X2, length_scale, noise_var, signal_var):
            """Compute GP kernel matrix between X1 and X2."""
            # Squared distance
            diff = X1[:, np.newaxis, :] - X2[np.newaxis, :, :]
            sq_dist = np.sum(diff ** 2, axis=2)
            K = signal_var * np.exp(-0.5 * sq_dist / (length_scale ** 2))
            # Add noise to diagonal if same points
            if X1.shape[0] == X2.shape[0] and np.allclose(X1, X2):
                K += noise_var * np.eye(X1.shape[0])
            return K

        def gp_predict(X_test, X_train, y_train, length_scale, noise_var, signal_var):
            """Predict mean and variance at X_test given training data."""
            K = kernel(X_train, X_train, length_scale, noise_var, signal_var)
            K += 1e-8 * np.eye(X_train.shape[0])  # for numerical stability
            K_inv = np.linalg.inv(K)
            K_s = kernel(X_train, X_test, length_scale, 0.0, signal_var)
            mu = K_s.T @ K_inv @ y_train
            K_ss = kernel(X_test, X_test, length_scale, 0.0, signal_var)
            var = np.diag(K_ss) - np.sum((K_s @ K_inv) * K_s.T, axis=1)
            var = np.maximum(var, 1e-10)
            return mu, var

        def expected_improvement(mu, var, best_f, xi=0.01):
            """Compute Expected Improvement acquisition function."""
            sigma = np.sqrt(var)
            imp = best_f - mu - xi
            with np.errstate(divide='ignore', invalid='ignore'):
                z = imp / sigma
                ei = imp * np.where(sigma > 1e-10, _norm_cdf(z), 0) + sigma * _norm_pdf(z)
            return np.where(sigma > 1e-10, ei, 0)

        def _norm_cdf(x):
            """Standard normal CDF."""
            return 0.5 * (1 + _erf(x / np.sqrt(2)))

        def _norm_pdf(x):
            """Standard normal PDF."""
            return np.exp(-0.5 * x ** 2) / np.sqrt(2 * np.pi)

        def _erf(x):
            """Approximation of the error function."""
            # Use scipy if available, otherwise approximate
            try:
                from math import erf
                return erf(x)
            except ImportError:
                # Abramowitz and Stegun approximation
                sign = 1 if x >= 0 else -1
                x = abs(x)
                t = 1.0 / (1.0 + 0.3275911 * x)
                y = 1.0 - (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t - 0.284496736) * t + 0.254829592) * t * np.exp(-x * x)
                return sign * y

        # Phase 2: Surrogate-assisted optimization loop
        # Adaptively tune the length_scale based on the spread of evaluated points
        while calls_made < budget and time.time() < deadline:
            # Update GP parameters based on data spread
            if len(X) > 5:
                X_arr = np.array(X)
                # Estimate length scale from typical distances
                from scipy.spatial.distance import pdist
                dists = pdist(X_arr)
                if len(dists) > 0:
                    length_scale = max(np.median(dists) * 0.5, 0.1)
                    length_scale = min(length_scale, 5.0)

            # Generate candidate points for the acquisition function
            n_candidates = 200
            candidates = rng.uniform(-5, 5, (n_candidates, dim))

            # Also add some local perturbations around the best point
            if best_x is not None:
                local_candidates = best_x + rng.normal(0, 0.5, (50, dim))
                local_candidates = np.clip(local_candidates, -5, 5)
                candidates = np.vstack([candidates, local_candidates])

            X_arr = np.array(X)
            y_arr = np.array(Y)

            # Predict mean and variance at candidates
            mu, var = gp_predict(candidates, X_arr, y_arr, length_scale, noise_var, signal_var)

            # Compute EI
            ei = expected_improvement(mu, var, best_f, xi=0.01)

            # Select the best candidate
            best_idx = np.argmax(ei)
            x_next = candidates[best_idx]

            # Evaluate the selected point
            evaluate(x_next)

            # Occasionally, add a random exploration point to avoid getting stuck
            if calls_made % 20 == 0 and calls_made < budget:
                x_explore = rng.uniform(-5, 5, dim)
                evaluate(x_explore)

        return best_x, best_f
# EVOLVE-BLOCK-END
