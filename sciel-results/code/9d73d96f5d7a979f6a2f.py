"""Bayesian Optimization with GP surrogate and EI acquisition for BBOB f16 (Weierstrass)."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import norm

class Optimizer:
    """
    Bayesian Optimization using a Gaussian Process surrogate with Expected Improvement
    acquisition function. Explicitly models multimodality through the GP prior,
    guiding sampling to balance exploration and exploitation.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0  # 20s safety margin before 300s limit
        dim = self.dim
        budget = self.budget

        best_x = None
        best_f = np.inf
        n_calls = 0

        def check_time():
            return time.time() > deadline

        def call_func(x):
            nonlocal n_calls
            if n_calls >= budget:
                return None
            n_calls += 1
            try:
                x_clipped = np.clip(np.asarray(x, dtype=float), -5, 5)
                val = float(func(x_clipped))
                return val, x_clipped
            except Exception:
                return None, None

        rng = np.random.default_rng(42)

        # --- Gaussian Process Surrogate ---
        class GPSurrogate:
            def __init__(self, dim, length_scale=1.0, signal_var=1.0, noise_var=1e-6):
                self.dim = dim
                self.length_scale = length_scale
                self.signal_var = signal_var
                self.noise_var = noise_var
                self.X = []
                self.y = []
                self.L = None
                self.alpha = None

            def _kern(self, X1, X2):
                # Squared exponential kernel
                X1 = np.asarray(X1)
                X2 = np.asarray(X2)
                sq_dists = np.sum(X1**2, axis=1)[:, None] + np.sum(X2**2, axis=1)[None, :] - 2 * X1 @ X2.T
                sq_dists = np.maximum(sq_dists, 0)
                return self.signal_var * np.exp(-0.5 * sq_dists / (self.length_scale ** 2))

            def fit(self, X, y):
                self.X = np.asarray(X)
                self.y = np.asarray(y)
                n = len(self.y)
                K = self._kern(self.X, self.X)
                K += self.noise_var * np.eye(n)
                # Add small jitter for numerical stability
                K += 1e-8 * np.eye(n)
                try:
                    self.L = np.linalg.cholesky(K)
                    self.alpha = np.linalg.solve(self.L.T, np.linalg.solve(self.L, self.y))
                except np.linalg.LinAlgError:
                    # Fallback: add more jitter
                    K += 1e-6 * np.eye(n)
                    try:
                        self.L = np.linalg.cholesky(K)
                        self.alpha = np.linalg.solve(self.L.T, np.linalg.solve(self.L, self.y))
                    except np.linalg.LinAlgError:
                        self.L = None
                        self.alpha = None

            def predict(self, Xs):
                Xs = np.asarray(Xs)
                if self.L is None:
                    # Return constant prediction
                    mu = np.full(len(Xs), np.mean(self.y))
                    var = np.full(len(Xs), self.signal_var)
                    return mu, var
                Ks = self._kern(self.X, Xs)  # n x m
                Kss = self._kern(Xs, Xs)  # m x m
                v = np.linalg.solve(self.L, Ks.T)  # n x m
                mu = v.T @ self.alpha  # m
                var = np.diag(Kss) - np.sum(v ** 2, axis=0)  # m
                var = np.maximum(var, 1e-10)
                return mu, var

        # --- Expected Improvement Acquisition ---
        def expected_improvement(mu, sigma, best_f, xi=0.01):
            if sigma < 1e-10:
                return np.zeros_like(mu)
            imp = best_f - mu - xi
            z = imp / sigma
            ei = imp * norm.cdf(z) + sigma * norm.pdf(z)
            return np.maximum(ei, 0)

        # --- Main Bayesian Optimization Loop ---
        gp = GPSurrogate(dim, length_scale=2.0, signal_var=100.0, noise_var=1e-6)
        
        # Initial sampling: Latin hypercube-like spread
        n_init = min(15, budget)
        X_init = rng.uniform(-5, 5, size=(n_init, dim))
        y_init = []
        for x in X_init:
            if n_calls >= budget or check_time():
                break
            fy, x_eval = call_func(x)
            if fy is None:
                continue
            y_init.append(fy)
            if fy < best_f:
                best_f = fy
                best_x = x_eval.copy()
        
        if len(y_init) >= 2:
            gp.fit(X_init[:len(y_init)], np.array(y_init))
        
        # Bayesian optimization iterations
        max_gp_retrain = 0
        stagnation_count = 0
        
        while n_calls < budget and not check_time():
            # Determine how many candidates to evaluate
            n_cand = min(20, budget - n_calls)
            if n_cand < 1:
                break
            
            # Generate candidate points: mix of exploration and exploitation
            # 1. Random points for exploration
            n_random = max(5, n_cand // 4)
            X_random = rng.uniform(-5, 5, size=(n_random, dim))
            
            # 2. Points near current best (exploitation)
            n_near_best = max(5, n_cand // 4)
            if best_x is not None:
                X_near = best_x[None, :] + rng.normal(0, 0.5, size=(n_near_best, dim))
                X_near = np.clip(X_near, -5, 5)
            else:
                X_near = rng.uniform(-5, 5, size=(n_near_best, dim))
            
            # 3. GP-based candidates if we have enough data
            X_gp = np.array([])
            if len(gp.X) >= 5 and gp.L is not None:
                n_gp = n_cand - n_random - n_near_best
                if n_gp > 0:
                    # Generate a batch of random points and select by EI
                    X_batch = rng.uniform(-5, 5, size=(n_gp * 5, dim))
                    mu, sigma = gp.predict(X_batch)
                    ei = expected_improvement(mu, sigma, best_f, xi=0.01)
                    # Select top n_gp by EI
                    top_idx = np.argsort(-ei)[:n_gp]
                    X_gp = X_batch[top_idx]
            
            # Combine all candidates
            candidates = np.vstack([X_random, X_near])
            if X_gp.size > 0:
                candidates = np.vstack([candidates, X_gp])
            
            # Deduplicate approximately
            if len(candidates) > 1:
                # Keep unique-ish points
                unique_cands = [candidates[0]]
                for i in range(1, len(candidates)):
                    if np.all(np.abs(candidates[i] - np.array(unique_cands)) > 0.1):
                        unique_cands.append(candidates[i])
                    if len(unique_cands) >= n_cand:
                        break
                candidates = np.array(unique_cands[:n_cand])
            
            # Evaluate candidates and update GP
            X_new = []
            y_new = []
            prev_best = best_f
            for x in candidates:
                if n_calls >= budget or check_time():
                    break
                fy, x_eval = call_func(x)
                if fy is None:
                    continue
                X_new.append(x_eval)
                y_new.append(fy)
                if fy < best_f:
                    best_f = fy
                    best_x = x_eval.copy()
            
            if len(X_new) == 0:
                break
            
            # Update GP with new data
            gp.X = np.vstack([gp.X, np.array(X_new)])
            gp.y = np.concatenate([gp.y, np.array(y_new)])
            
            # Refit GP periodically (every few iterations to save time)
            max_gp_retrain += 1
            if max_gp_retrain % 3 == 0 or len(gp.X) <= 50:
                gp.fit(gp.X, gp.y)
            
            # Stagnation detection
            if best_f == prev_best:
                stagnation_count += 1
            else:
                stagnation_count = 0
            
            # If stagnating, increase exploration
            if stagnation_count > 5:
                # Reset GP to encourage fresh exploration
                if len(gp.X) > 20:
                    # Keep only the best 20 points
                    idx = np.argsort(gp.y)[:20]
                    gp.X = gp.X[idx]
                    gp.y = gp.y[idx]
                    gp.fit(gp.X, gp.y)
                stagnation_count = 0
        
        if best_x is None:
            return None

        return best_x, best_f

# EVOLVE-BLOCK-END
