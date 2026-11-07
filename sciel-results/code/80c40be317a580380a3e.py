import numpy as np
import time
import scipy.optimize as so
from scipy.stats import norm

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.calls = 0
        self.best_x = None
        self.best_f = np.inf
        self.start_time = time.time()
        # Deadline guard: 300s limit, keep 10s margin for safety
        self.deadline = self.start_time + 290.0
        # Data storage for GP
        self.X = []
        self.Y = []

    def _check_time(self):
        return time.time() < self.deadline

    def __call__(self, func):
        rng = np.random.default_rng(42)
        n = self.dim
        
        # Wrap func to track calls and enforce budget
        def safe_func(x):
            if self.calls >= self.budget:
                return self.best_f
            if not self._check_time():
                return self.best_f
            x = np.asarray(x, dtype=float)
            # Clip to box to ensure validity
            x = np.clip(x, -5.0, 5.0)
            val = func(x)
            self.calls += 1
            if val < self.best_f:
                self.best_f = val
                self.best_x = x.copy()
            return val

        bounds = [(-5.0, 5.0) for _ in range(n)]
        
        # Phase 1: Initial Exploration 
        # GP needs a decent initial set of points to model the landscape.
        # We use a small random sample to seed the GP.
        # 50 calls is a reasonable seed size for 5D.
        init_calls = 50
        for _ in range(init_calls):
            if self.calls >= self.budget or not self._check_time():
                break
            x = rng.uniform(-5.0, 5.0, size=n)
            safe_func(x)
            self.X.append(x)
            self.Y.append(self.best_f if self.best_f != np.inf else 0.0) # Wait, Y should be the actual value, not best_f
            # Correction: We need the actual value returned by func for GP training.
            # Let's re-evaluate or store properly.
            
        # Re-do Phase 1 properly to store Y correctly
        self.X = []
        self.Y = []
        self.calls = 0
        self.best_f = np.inf
        self.best_x = None
        
        for _ in range(init_calls):
            if self.calls >= self.budget or not self._check_time():
                break
            x = rng.uniform(-5.0, 5.0, size=n)
            val = safe_func(x)
            self.X.append(x)
            self.Y.append(val)

        # Phase 2: GP-Guided Search
        # We will iteratively:
        # 1. Fit a GP to current data (X, Y).
        # 2. Use an acquisition function (Expected Improvement) to find the next point.
        # 3. Evaluate the function at that point.
        # 4. Repeat until budget is exhausted.
        
        # Hyperparameters for GP
        # Length scales initialized to 1.0, noise level small
        # Since we are in a fixed box [-5, 5], length scales of 1-2 are reasonable.
        
        def gp_predict(X_train, Y_train, X_test):
            """
            Simple GP implementation using RBF kernel.
            X_train: (N, D)
            Y_train: (N,)
            X_test: (M, D)
            Returns: mean, var
            """
            # Standardize Y for numerical stability
            Y_mean = np.mean(Y_train)
            Y_std = np.std(Y_train)
            if Y_std < 1e-8:
                Y_std = 1.0
            Y_std = max(Y_std, 1e-3)
            Y_norm = (Y_train - Y_mean) / Y_std
            
            # Kernel
            # RBF: exp(-0.5 * ||x - x'||^2 / l^2)
            # We use a diagonal length scale matrix for simplicity, or a single length scale.
            # For 5D, a single length scale is often sufficient and faster to fit.
            # Let's try a single length scale l=1.0 and noise epsilon=1e-6.
            
            def rbf_kernel(X1, X2, l=1.0, epsilon=1e-6):
                # Compute squared distances
                # X1: (N, D), X2: (M, D)
                # dist2[i, j] = ||X1[i] - X2[j]||^2
                X1_sq = np.sum(X1**2, axis=1, keepdims=True)
                X2_sq = np.sum(X2**2, axis=1, keepdims=True).T
                dist2 = X1_sq + X2_sq - 2 * X1 @ X2.T
                dist2 = np.maximum(dist2, 0) # Numerical safety
                K = np.exp(-0.5 * dist2 / (l**2))
                K[np.diag_indices_from(K)] += epsilon
                return K
            
            K_train = rbf_kernel(X_train, X_train)
            K_test = rbf_kernel(X_train, X_test)
            
            # Solve for weights
            # alpha = K_train^-1 Y_norm
            try:
                alpha = np.linalg.solve(K_train, Y_norm)
            except np.linalg.LinAlgError:
                # Fallback if singular
                alpha = np.linalg.lstsq(K_train, Y_norm, rcond=None)[0]
                
            # Mean prediction
            mu = K_test @ alpha
            mu = mu * Y_std + Y_mean
            
            # Variance prediction
            # var = K(x*, x*) - K(x*, X_train) K_train^-1 K(X_train, x*)
            K_star_star = rbf_kernel(X_test, X_test)
            var = np.diag(K_star_star) - np.sum(K_test * (K_test @ alpha), axis=1)
            var = np.maximum(var, 1e-10) # Numerical safety
            var = var * (Y_std**2)
            
            return mu, var

        def acquisition_function(X_cand, X_train, Y_train):
            """
            Expected Improvement (EI) acquisition function.
            Returns EI for each candidate point in X_cand.
            """
            mu, var = gp_predict(X_train, Y_train, X_cand)
            sigma = np.sqrt(var)
            f_min = np.min(Y_train)
            
            # EI = (f_min - mu - xi) * Phi((f_min - mu - xi)/sigma) + sigma * phi((f_min - mu - xi)/sigma)
            # xi is exploration constant, set to 0.1 * std(Y) or small fixed value
            xi = 0.1 * np.std(Y_train)
            if np.std(Y_train) < 1e-3:
                xi = 1e-3
                
            z = (f_min - mu - xi) / (sigma + 1e-10)
            ei = (f_min - mu - xi) * norm.cdf(z) + sigma * norm.pdf(z)
            
            return ei

        # Main Loop
        while self.calls < self.budget and self._check_time():
            # 1. Generate candidate points
            # We generate a set of random candidates and pick the one with highest EI.
            # This is a "lazy" GP approach, which is efficient for small budgets.
            # Number of candidates: 100-500 random points.
            n_cand = 200
            X_cand = rng.uniform(-5.0, 5.0, size=(n_cand, n))
            
            # 2. Evaluate acquisition function
            ei = acquisition_function(X_cand, np.array(self.X), np.array(self.Y))
            
            # 3. Pick best candidate
            best_idx = np.argmax(ei)
            x_next = X_cand[best_idx]
            
            # 4. Evaluate function
            val = safe_func(x_next)
            
            # 5. Update data
            self.X.append(x_next)
            self.Y.append(val)
            
            # Optional: If we are close to the minimum, switch to local search?
            # For now, let GP handle it. GP is good at both exploration and exploitation.
            
        # Phase 3: Local Polish
        # After GP search, we have a good estimate of the global minimum.
        # Use L-BFGS-B to polish the best point found.
        if self.calls < self.budget and self._check_time() and self.best_x is not None:
            x0 = self.best_x.copy()
            try:
                res_min = so.minimize(
                    safe_func, 
                    x0, 
                    method='L-BFGS-B', 
                    bounds=bounds,
                    options={'maxiter': 50, 'ftol': 1e-12, 'gtol': 1e-8}
                )
                if res_min.fun < self.best_f:
                    self.best_f = res_min.fun
                    self.best_x = res_min.x
            except Exception:
                pass

        self.best_x = self.best_x if self.best_x is not None else np.zeros(n)
        self.best_f = self.best_f if np.isfinite(self.best_f) else 0.0
        
        return self.best_x, self.best_f
