import numpy as np
import time
import math
from scipy.optimize import minimize
from scipy.spatial.distance import cdist

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 295
        dim = self.dim
        budget = self.budget
        rng = np.random.default_rng(42)
        
        calls_used = 0
        best_f = float('inf')
        best_x = np.zeros(dim)
        
        # Store evaluated points for surrogate
        X_eval = []
        f_eval = []
        
        # Wrapper to count calls
        def wrapped_func(x):
            nonlocal calls_used, best_f, best_x
            if time.time() > deadline or calls_used >= budget:
                return float('inf')
            calls_used += 1
            fx = func(x)
            X_eval.append(x.copy())
            f_eval.append(fx)
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            return fx

        # RBF surrogate model
        class RBFModel:
            def __init__(self, dim, length_scale=1.0):
                self.dim = dim
                self.length_scale = length_scale
                self.X = None
                self.f = None
                self.alpha = None
                self.mean = 0.0
                self.var = 1.0
                self.fitted = False
                
            def fit(self, X, f):
                self.X = np.array(X)
                self.f = np.array(f)
                n = len(X)
                if n == 0:
                    self.fitted = False
                    return
                if n == 1:
                    self.mean = self.f[0]
                    self.var = 0.0
                    self.fitted = True
                    return
                # Compute RBF kernel matrix
                D = cdist(self.X, self.X)
                K = np.exp(-D**2 / (2 * self.length_scale**2))
                # Add small regularization for numerical stability
                K += 1e-6 * np.eye(n)
                # Solve for alpha
                try:
                    self.alpha = np.linalg.solve(K, self.f - self.mean)
                except np.linalg.LinAlgError:
                    self.alpha = np.linalg.lstsq(K, self.f - self.mean, rcond=None)[0]
                self.mean = np.mean(self.f)
                self.var = np.var(self.f)
                self.fitted = True
                
            def predict(self, X_new):
                if not self.fitted:
                    return np.zeros(len(X_new)), np.zeros(len(X_new))
                X_new = np.array(X_new)
                if len(self.X) == 1:
                    return np.full(len(X_new), self.mean), np.zeros(len(X_new))
                D = cdist(X_new, self.X)
                K = np.exp(-D**2 / (2 * self.length_scale**2))
                mean = self.mean + K @ self.alpha
                # Uncertainty: diagonal of (I - K @ K_inv) * sigma^2
                # Approximate with 1 - sum of squared kernel values
                uncertainty = self.var * np.maximum(0, 1 - np.sum(K**2, axis=1))
                return mean, uncertainty
        
        # Acquisition function: Expected Improvement
        def expected_improvement(mean, uncertainty, best_so_far):
            if uncertainty < 1e-10:
                improvement = best_so_far - mean
                return np.maximum(0, improvement)
            z = (best_so_far - mean) / uncertainty
            from scipy.stats import norm
            ei = uncertainty * (z * norm.cdf(z) + norm.pdf(z))
            ei = np.maximum(0, ei)
            return ei

        # Phase 1: Initial random sampling
        n_initial = 15
        for i in range(n_initial):
            if calls_used >= budget or time.time() > deadline:
                break
            x = rng.uniform(-5, 5, dim)
            wrapped_func(x)
        
        # Phase 2: Surrogate-assisted optimization
        # Fit initial model
        model = RBFModel(dim, length_scale=1.5)
        if len(X_eval) > 0:
            model.fit(X_eval, f_eval)
        
        # Iteratively select next point using EI acquisition
        max_surrogate_iters = 200
        for it in range(max_surrogate_iters):
            if calls_used >= budget - 50 or time.time() > deadline - 30:
                break
            
            # Generate candidate points
            n_candidates = 200
            candidates = []
            # Mix of random and local perturbations around best
            for j in range(n_candidates):
                if j < n_candidates // 3:
                    # Random in box
                    x_cand = rng.uniform(-5, 5, dim)
                elif j < 2 * n_candidates // 3:
                    # Perturbation around best
                    if len(best_x) > 0:
                        x_cand = best_x + rng.normal(0, 0.5, dim)
                    else:
                        x_cand = rng.uniform(-5, 5, dim)
                else:
                    # Perturbation around second-best if available
                    if len(f_eval) >= 2:
                        sorted_idx = np.argsort(f_eval)
                        second_best_x = X_eval[sorted_idx[1]]
                        x_cand = second_best_x + rng.normal(0, 0.5, dim)
                    else:
                        x_cand = rng.uniform(-5, 5, dim)
                # Clip to box
                x_cand = np.clip(x_cand, -5, 5)
                candidates.append(x_cand)
            
            candidates = np.array(candidates)
            
            # Predict with surrogate
            if model.fitted:
                mean_pred, unc_pred = model.predict(candidates)
                # Compute EI
                ei = expected_improvement(mean_pred, unc_pred, best_f)
                # Add small random noise to break ties
                ei += rng.normal(0, 1e-4, len(ei))
                # Select best EI point
                best_idx = np.argmax(ei)
                x_next = candidates[best_idx]
            else:
                x_next = rng.uniform(-5, 5, dim)
            
            # Evaluate
            fx_next = wrapped_func(x_next)
            
            # Update surrogate periodically (every 5 evaluations)
            if it % 5 == 4 and len(X_eval) >= 5:
                # Adapt length scale based on function landscape
                f_range = max(f_eval) - min(f_eval)
                if f_range > 1e-10:
                    # Estimate characteristic length
                    if len(X_eval) >= 2:
                        X_arr = np.array(X_eval)
                        f_arr = np.array(f_eval)
                        # Simple heuristic: adjust length scale
                        model.length_scale = max(0.5, min(3.0, model.length_scale * 0.95))
                model.fit(X_eval, f_eval)
        
        # Phase 3: Final local polish with Nelder-Mead if budget remains
        if calls_used < budget - 20 and time.time() < deadline - 10:
            try:
                res = minimize(
                    wrapped_func,
                    best_x,
                    method='Nelder-Mead',
                    options={'maxiter': 30, 'maxfev': 40, 'xatol': 1e-4, 'fatol': 1e-6}
                )
                if res.fun < best_f:
                    best_f = res.fun
                    best_x = res.x.copy()
            except Exception:
                pass
        
        return (best_x, best_f)
