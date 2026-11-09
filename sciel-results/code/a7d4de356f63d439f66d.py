import numpy as np
import time
import sys
from scipy.optimize import minimize
from scipy.stats import norm

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self._call_count = 0
        self._start_time = None
        self._deadline = None
        self._best_x = None
        self._best_f = np.inf

    def __call__(self, func):
        self._start_time = time.time()
        self._deadline = self._start_time + 320
        self._call_count = 0
        self.budget_remaining = self.budget
        self._best_x = np.zeros(self.dim)
        self._best_f = np.inf
        self._rng = np.random.default_rng(42)

        # Store all observations for GP
        self._X_obs = []
        self._y_obs = []

        def safe_call(x):
            if self._call_count >= self.budget or time.time() >= self._deadline:
                return 1e10
            self._call_count += 1
            self.budget_remaining -= 1
            val = func(x)
            if not np.isfinite(val):
                return 1e10
            self._X_obs.append(x.copy())
            self._y_obs.append(val)
            if val < self._best_f:
                self._best_f = val
                self._best_x = x.copy()
            return val

        # Phase 1: Initial exploration with Sobol points
        try:
            from scipy.stats import qmc
            sampler = qmc.Sobol(d=self.dim, scramble=True, seed=42)
            n_init = 30
            sobol_points = sampler.random(n_init)
            sobol_points = sobol_points * 10.0 - 5.0
        except ImportError:
            n_init = 30
            sobol_points = self._rng.uniform(-5, 5, (n_init, self.dim))

        for i in range(n_init):
            if self._call_count >= self.budget or time.time() >= self._deadline:
                break
            safe_call(sobol_points[i])

        # Phase 2: Bayesian Optimization with GP surrogate
        # Use a simple GP with RBF kernel
        def gp_predict(X_pred, X_train, y_train, length_scale=1.0, sigma_f=1.0, noise=1e-6):
            """Simple GP prediction with RBF kernel"""
            n_train = len(X_train)
            n_pred = len(X_pred)
            
            # Compute kernel matrix K
            def rbf_kernel(A, B, length_scale):
                # A: (n, d), B: (m, d)
                sq_dist = np.sum(A**2, axis=1)[:, None] + np.sum(B**2, axis=1)[None, :] - 2 * A @ B.T
                sq_dist = np.maximum(sq_dist, 0)
                return np.exp(-sq_dist / (2 * length_scale**2))
            
            K = rbf_kernel(X_train, X_train, length_scale) + noise * np.eye(n_train)
            K_pred = rbf_kernel(X_train, X_pred, length_scale)
            
            # Cholesky decomposition
            try:
                L = np.linalg.cholesky(K)
                alpha = np.linalg.solve(L.T, np.linalg.solve(L, y_train))
            except np.linalg.LinAlgError:
                # Add jitter if Cholesky fails
                K += 1e-8 * np.eye(n_train)
                L = np.linalg.cholesky(K)
                alpha = np.linalg.solve(L.T, np.linalg.solve(L, y_train))
            
            # Predict mean and variance
            mu = K_pred.T @ alpha
            v = np.linalg.solve(L, K_pred)
            var = sigma_f**2 - np.sum(v**2, axis=0)
            var = np.maximum(var, 1e-10)
            
            return mu, np.sqrt(var)

        # Expected Improvement acquisition
        def expected_improvement(X_pred, X_train, y_train, best_f, length_scale=1.0, sigma_f=1.0):
            mu, sigma = gp_predict(X_pred, X_train, y_train, length_scale, sigma_f)
            sigma = np.maximum(sigma, 1e-8)
            
            # EI = (mu - best_f) * Phi(Z) + sigma * phi(Z)
            # where Z = (mu - best_f) / sigma
            Z = (mu - best_f) / sigma
            ei = (mu - best_f) * norm.cdf(Z) + sigma * norm.pdf(Z)
            return ei

        # Acquisition function to minimize (negative EI)
        def acquisition(x, X_train, y_train, best_f, length_scale=1.0, sigma_f=1.0):
            x = np.array(x).reshape(1, -1)
            ei = expected_improvement(x, X_train, y_train, best_f, length_scale, sigma_f)
            return -ei[0]

        # GP hyperparameters (tuned for 5D Schwefel)
        length_scale = 1.5
        sigma_f = 50.0  # Schwefel values can be large

        # BO loop
        max_bo_calls = self.budget - 10  # Leave some budget for safety
        bo_calls = 0
        
        while bo_calls < max_bo_calls and self._call_count < self.budget and time.time() < self._deadline:
            if len(self._X_obs) < 3:
                # Need at least 3 points for meaningful GP
                x_new = self._rng.uniform(-5, 5, self.dim)
                safe_call(x_new)
                bo_calls += 1
                continue
            
            X_train = np.array(self._X_obs)
            y_train = np.array(self._y_obs)
            best_f = self._best_f
            
            # Optimize acquisition function using multiple random starts
            best_acq_val = np.inf
            best_x_acq = None
            
            n_starts = 8
            for start_idx in range(n_starts):
                if time.time() >= self._deadline:
                    break
                
                # Random start
                x0 = self._rng.uniform(-5, 5, self.dim)
                
                try:
                    result = minimize(
                        acquisition,
                        x0,
                        args=(X_train, y_train, best_f, length_scale, sigma_f),
                        method='L-BFGS-B',
                        bounds=[(-5.0, 5.0)] * self.dim,
                        options={'maxiter': 20, 'ftol': 1e-10}
                    )
                    if result.fun < best_acq_val:
                        best_acq_val = result.fun
                        best_x_acq = result.x
                except Exception:
                    continue
            
            if best_x_acq is None:
                # Fallback: random point
                best_x_acq = self._rng.uniform(-5, 5, self.dim)
            
            # Evaluate the selected point
            f_new = safe_call(best_x_acq)
            bo_calls += 1
            
            # Adapt GP hyperparameters based on progress
            if bo_calls % 20 == 0:
                # If improving, exploit more (smaller length_scale)
                # If stagnating, explore more (larger length_scale)
                if bo_calls < max_bo_calls * 0.5:
                    length_scale = max(0.5, length_scale * 0.95)
                else:
                    length_scale = min(3.0, length_scale * 1.05)

        # Phase 3: Final polish with local optimization
        if self._call_count < self.budget - 5 and time.time() < self._deadline - 10:
            try:
                def wrapper(x):
                    if self._call_count >= self.budget or time.time() >= self._deadline:
                        return 1e10
                    self._call_count += 1
                    self.budget_remaining -= 1
                    val = func(x)
                    if not np.isfinite(val):
                        return 1e10
                    return val
                
                bounds = [(-5.0, 5.0)] * self.dim
                result = minimize(wrapper, self._best_x, method='L-BFGS-B', 
                                  bounds=bounds, options={'maxiter': 20, 'ftol': 1e-12, 'gtol': 1e-8})
                
                if result.fun < self._best_f and np.isfinite(result.fun):
                    self._best_f = result.fun
                    self._best_x = result.x.copy()
            except Exception:
                pass

        return self._best_x, self._best_f
