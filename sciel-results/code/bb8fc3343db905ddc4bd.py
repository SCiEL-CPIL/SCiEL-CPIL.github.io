# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.stats import norm


class Optimizer:
    """Derivative-free optimizer using Gaussian Process Bayesian Optimization and Nelder-Mead."""

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
        
        # Store observations for GP
        X_obs = []
        y_obs = []

        def wrapped_func(x):
            nonlocal best_x, best_f, self
            x = np.asarray(x, dtype=float)
            x = np.clip(x, -5.0, 5.0)
            val = float(func(x))
            self.calls += 1
            if val < best_f:
                best_f = val
                best_x = x.copy()
            return val

        def gp_predict(X_new, X_train, y_train):
            """
            Simple GP prediction with RBF kernel.
            Returns mean and std.
            """
            # Hyperparameters
            sigma_f = 1.0  # Signal variance
            l = 1.0        # Length scale
            noise = 1e-6   # Noise variance

            def rbf_kernel(X1, X2):
                # X1: (n1, d), X2: (n2, d)
                X1_sq = np.sum(X1**2, axis=1)[:, np.newaxis]
                X2_sq = np.sum(X2**2, axis=1)[:, np.newaxis].T
                dist_sq = X1_sq + X2_sq - 2 * np.dot(X1, X2.T)
                dist_sq = np.maximum(dist_sq, 0) # numerical stability
                return sigma_f**2 * np.exp(-dist_sq / (2 * l**2))

            K_train = rbf_kernel(X_train, X_train)
            K_train += noise * np.eye(len(X_train))
            
            K_new_train = rbf_kernel(X_new, X_train)

            try:
                L = np.linalg.cholesky(K_train)
                alpha = np.linalg.solve(L.T, np.linalg.solve(L, y_train))
                v = np.linalg.solve(L, K_new_train.T)
            except np.linalg.LinAlgError:
                # Fallback to pseudo-inverse if Cholesky fails
                alpha = np.linalg.pinv(K_train) @ y_train
                v = np.linalg.pinv(K_train) @ K_new_train.T

            # Mean and Variance
            mu = K_new_train @ alpha
            var = sigma_f**2 - np.sum(v**2, axis=0)
            var = np.maximum(var, 0)
            std = np.sqrt(var)
            
            return mu, std

        def expected_improvement(mu, std, f_best):
            """
            Calculate Expected Improvement for minimization.
            """
            with np.errstate(divide='ignore', invalid='ignore'):
                Z = (f_best - mu) / std
                EI = (f_best - mu) * norm.cdf(Z) + std * norm.pdf(Z)
            EI[std == 0] = 0
            return EI

        # Phase 1: Bayesian Optimization
        # Reserve budget for NM
        nm_budget = 50
        bo_budget = max(10, budget - nm_budget)
        
        # Initial random points
        n_init = min(10, bo_budget)
        for _ in range(n_init):
            if self.calls >= bo_budget:
                break
            x = rng.uniform(-5, 5, dim)
            val = wrapped_func(x)
            X_obs.append(x)
            y_obs.append(val)

        if len(X_obs) > 0:
            X_obs = np.array(X_obs)
            y_obs = np.array(y_obs)
        else:
            X_obs = np.empty((0, dim))
            y_obs = np.empty(0)

        # BO Loop
        while self.calls < bo_budget:
            if len(X_obs) < 2:
                # Need at least 2 points for stable GP
                x = rng.uniform(-5, 5, dim)
                val = wrapped_func(x)
                X_obs = np.vstack([X_obs, x])
                y_obs = np.append(y_obs, val)
                continue

            # Fit GP and predict on candidates
            n_candidates = 1000
            X_cand = rng.uniform(-5, 5, (n_candidates, dim))
            
            mu, std = gp_predict(X_cand, X_obs, y_obs)
            
            f_best = np.min(y_obs)
            EI = expected_improvement(mu, std, f_best)
            
            # Select best candidate
            idx = np.argmax(EI)
            x_next = X_cand[idx]
            
            val = wrapped_func(x_next)
            X_obs = np.vstack([X_obs, x_next])
            y_obs = np.append(y_obs, val)

        # Phase 2: Nelder-Mead refinement from the best point
        if self.calls < budget and best_x is not None:
            remaining = budget - self.calls
            nm_budget = int(remaining * 0.8)
            
            try:
                result = minimize(
                    wrapped_func,
                    best_x,
                    method='Nelder-Mead',
                    options={'maxiter': nm_budget, 'maxfev': nm_budget, 'xatol': 1e-8, 'fatol': 1e-10}
                )
            except Exception:
                pass

        # Phase 3: If we still have budget, do a few more Nelder-Mead runs
        if self.calls < budget:
            remaining = budget - self.calls
            if remaining > 50:
                n_extra = min(3, remaining // 50)
                for _ in range(n_extra):
                    if self.calls >= budget - 10:
                        break
                    x_start = best_x + rng.normal(0, 0.5, dim)
                    x_start = np.clip(x_start, -5, 5)
                    try:
                        result = minimize(
                            wrapped_func,
                            x_start,
                            method='Nelder-Mead',
                            options={'maxiter': remaining // (n_extra + 1), 'maxfev': remaining // (n_extra + 1), 'xatol': 1e-8, 'fatol': 1e-10}
                        )
                    except Exception:
                        pass

        return best_x, best_f
# EVOLVE-BLOCK-END
