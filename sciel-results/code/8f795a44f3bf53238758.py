import numpy as np
import time
import math
from scipy.optimize import minimize
from scipy.stats import norm

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 310.0  # 20s safety margin before 330s kill
        
        dim = self.dim
        budget = self.budget
        
        # State
        X = []  # List of evaluated points
        Y = []  # List of function values
        calls_used = 0
        best_x = None
        best_f = np.inf
        rng = np.random.default_rng(42)
        
        # Helper to evaluate
        def eval_point(x):
            nonlocal best_x, best_f, calls_used
            if calls_used >= budget:
                return None
            x = np.clip(x, -5, 5)
            fx = float(func(x))
            calls_used += 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            return fx

        # Gaussian Process Helper
        # Using a simple GP with RBF kernel and white noise
        class SimpleGP:
            def __init__(self, kernel='rbf', noise=1e-4):
                self.noise = noise
                self.kernel_type = kernel
                self.alpha = None
                self.K_inv = None
                self.X_train = None
                self.y_train = None
                # Hyperparameters (fixed for simplicity and speed, tuned for 5D box)
                self.length_scale = 1.0
                self.variance = 1.0
                
            def _kernel(self, X1, X2):
                # RBF kernel
                # Ensure broadcasting: X1 (n1, d), X2 (n2, d) -> (n1, n2)
                sq_dist = np.sum(X1**2, axis=1)[:, None] + np.sum(X2**2, axis=1)[None, :] - 2 * X1 @ X2.T
                sq_dist = np.maximum(sq_dist, 0)
                return self.variance * np.exp(-0.5 * sq_dist / (self.length_scale ** 2))
            
            def fit(self, X, y):
                self.X_train = np.array(X)
                self.y_train = np.array(y)
                n = len(X)
                K = self._kernel(self.X_train, self.X_train) + self.noise * np.eye(n)
                # Solve K alpha = y
                try:
                    self.alpha = np.linalg.solve(K, self.y_train)
                except np.linalg.LinAlgError:
                    # Fallback to least squares if singular
                    self.alpha = np.linalg.lstsq(K, self.y_train, rcond=None)[0]
                
            def predict(self, X_test, return_std=True):
                X_test = np.array(X_test)
                K_s = self._kernel(self.X_train, X_test)
                K_ss = self._kernel(X_test, X_test)
                
                mu = K_s.T @ self.alpha
                if return_std:
                    var = np.diag(K_ss) - np.sum(K_s**2 * self.alpha[:, None], axis=0)
                    var = np.maximum(var, 1e-12)
                    std = np.sqrt(var)
                    return mu, std
                return mu

        # Phase 1: Initial Sampling (Sobol-like or Halton) to cover space
        # We need enough points to initialize the GP. Let's use 15-20 points.
        n_init = 15
        # Generate Halton points for better coverage than random
        def halton(dim, n):
            points = []
            for i in range(1, n+1):
                p = []
                for d in range(dim):
                    f = 1
                    i_val = i
                    base = d + 2 # 2, 3, 5, 7, 11...
                    res = 0
                    while i_val > 0:
                        f *= base
                        res += (i_val % base) / f
                        i_val //= base
                    p.append(res)
                points.append(p)
            return np.array(points)
        
        # Map [0,1]^d to [-5,5]^d
        halton_pts = halton(dim, n_init)
        halton_pts = halton_pts * 10.0 - 5.0
        
        for x in halton_pts:
            if time.time() > deadline:
                break
            fx = eval_point(x)
            if fx is not None:
                X.append(x)
                Y.append(fx)
        
        # Phase 2: Bayesian Optimization Loop
        gp = SimpleGP()
        
        # Fit initial GP
        if len(X) > 1:
            gp.fit(X, Y)
            
        while calls_used < budget:
            if time.time() > deadline:
                break
            
            # 1. Select next point using Expected Improvement (EI)
            # Candidate points:
            # a) Random points (exploration)
            # b) Points around current best (exploitation)
            # c) Grid points? Too expensive.
            
            # Generate a pool of candidates
            n_candidates = 200
            candidates = []
            
            # 50% random
            n_rand = n_candidates // 2
            rand_pts = rng.uniform(-5, 5, size=(n_rand, dim))
            candidates.extend(rand_pts)
            
            # 50% local search around best
            n_local = n_candidates - n_rand
            local_pts = rng.normal(loc=best_x, scale=0.5, size=(n_local, dim))
            local_pts = np.clip(local_pts, -5, 5)
            candidates.extend(local_pts)
            
            candidates = np.array(candidates)
            
            # Predict mean and std for candidates
            try:
                mu, std = gp.predict(candidates)
            except Exception:
                # If GP fails, fall back to random
                next_x = rng.uniform(-5, 5, size=dim)
                fx = eval_point(next_x)
                if fx is not None:
                    X.append(next_x)
                    Y.append(fx)
                    continue

            # Calculate EI
            best_y = min(Y)
            improvement = best_y - mu
            Z = improvement / (std + 1e-8)
            ei = improvement * norm.cdf(Z) + std * norm.pdf(Z)
            
            # Handle numerical issues
            ei = np.nan_to_num(ei, nan=0.0, posinf=0.0, neginf=0.0)
            
            # Select best candidate
            idx = np.argmax(ei)
            next_x = candidates[idx]
            
            # Evaluate
            fx = eval_point(next_x)
            if fx is None:
                break
            X.append(next_x)
            Y.append(fx)
            
            # 2. Local Refinement
            # If we found a significantly better point or if we have enough budget,
            # run a quick local optimizer (Nelder-Mead) to squeeze out value.
            # We limit the number of local refinement steps to avoid getting stuck.
            # Let's refine if the new point is the new global best and we have > 50 calls left.
            
            if fx == best_f and calls_used < budget - 50:
                # Wrap func for NM
                def nm_func(x):
                    if time.time() > deadline:
                        return best_f
                    return eval_point(x)
                
                # Run NM with tight bounds on iterations
                try:
                    res = minimize(nm_func, next_x, method='Nelder-Mead', 
                                   options={'maxiter': 50, 'xatol': 1e-6, 'fatol': 1e-6})
                    # Note: minimize calls func multiple times.
                    # We must ensure we don't exceed budget.
                    # The nm_func checks budget and time.
                except Exception:
                    pass

            # 3. Update GP
            # Refit GP periodically or every step?
            # Refitting every step is O(n^3). With n=500, 500^3 = 1.25e8 operations.
            # In numpy, this is fast (<0.1s). So we can refit every step.
            if len(X) > 1:
                try:
                    gp.fit(X, Y)
                except Exception:
                    pass

        return best_x, best_f
