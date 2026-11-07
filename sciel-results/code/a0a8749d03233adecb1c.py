import numpy as np
import time
from scipy.optimize import minimize
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern
from scipy.stats import norm

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        t0 = time.time()
        deadline = t0 + 290.0  # Safety margin before 330s limit

        rng = np.random.default_rng(42)

        # Phase 1: Initial diverse sampling
        # Include structured points and random samples to find a good basin
        n_structured = 5
        structured_points = [
            np.zeros(self.dim),
            np.ones(self.dim) * 0.5,
            np.ones(self.dim) * -0.5,
            np.ones(self.dim) * 1.0,
            np.ones(self.dim) * -1.0,
        ]
        
        n_random = 35
        random_points = rng.uniform(-5, 5, (n_random, self.dim))
        
        all_points = np.vstack([structured_points, random_points])
        n_initial = len(all_points)
        
        # Evaluate all initial points
        fvals = np.array([func(x) for x in all_points])
        calls_used = n_initial
        
        best_idx = np.argmin(fvals)
        best_x = all_points[best_idx].copy()
        best_f = fvals[best_idx]
        
        if time.time() > deadline - 10:
            return best_x, best_f

        remaining = self.budget - calls_used
        
        # Phase 2: L-BFGS-B local refinement from the best point
        call_count = [0]
        
        def lbfgs_wrapper(x):
            if call_count[0] >= remaining:
                raise ValueError("Budget exceeded")
            if time.time() > deadline - 5:
                raise ValueError("Time exceeded")
            call_count[0] += 1
            return func(x)
        
        try:
            res = minimize(
                lbfgs_wrapper, 
                best_x, 
                method='L-BFGS-B',
                bounds=[(-5, 5)] * self.dim,
                options={'maxiter': 500, 'maxfun': remaining, 'maxls': 20, 'gtol': 1e-6}
            )
            
            if res.fun < best_f:
                best_f = res.fun
                best_x = res.x.copy()
                
        except Exception:
            pass
        
        remaining = self.budget - (calls_used + call_count[0])
        
        if time.time() > deadline - 15:
            return best_x, best_f

        # Phase 3: Bayesian Optimization for global exploration
        # Use GP surrogate to intelligently explore the landscape
        if remaining > 50:
            # Collect data for GP
            X_data = all_points.copy()
            y_data = fvals.copy()
            
            # Add the L-BFGS-B result
            X_data = np.vstack([X_data, best_x.reshape(1, -1)])
            y_data = np.append(y_data, best_f)
            
            bo_call_count = [0]
            bo_remaining = min(remaining, 150)  # Cap BO at 150 calls
            
            # Define kernel for GP
            length_scale = 1.0
            kernel = Matern(length_scale=length_scale, nu=2.5)
            
            # Bayesian Optimization loop
            for bo_iter in range(bo_remaining):
                if time.time() > deadline - 10:
                    break
                if bo_call_count[0] >= bo_remaining:
                    break
                
                # Fit GP to current data
                try:
                    gp = GaussianProcessRegressor(
                        kernel=kernel, 
                        alpha=1e-10, 
                        n_restarts_optimizer=5,
                        normalize_y=True
                    )
                    gp.fit(X_data, y_data)
                except Exception:
                    # If GP fitting fails, fall back to random sampling
                    x_candidate = rng.uniform(-5, 5, self.dim)
                    f_candidate = func(x_candidate)
                    bo_call_count[0] += 1
                    
                    if f_candidate < best_f:
                        best_f = f_candidate
                        best_x = x_candidate.copy()
                    
                    X_data = np.vstack([X_data, x_candidate.reshape(1, -1)])
                    y_data = np.append(y_data, f_candidate)
                    continue
                
                # Generate candidate points
                n_candidates = 50
                X_candidates = rng.uniform(-5, 5, (n_candidates, self.dim))
                
                # Predict mean and variance
                mu, sigma = gp.predict(X_candidates, return_std=True)
                
                # Expected Improvement acquisition function
                sigma = np.maximum(sigma, 1e-8)
                improvement = best_f - mu
                Z = improvement / sigma
                EI = improvement * norm.cdf(Z) + sigma * norm.pdf(Z)
                
                # Select best candidate
                best_candidate_idx = np.argmax(EI)
                x_candidate = X_candidates[best_candidate_idx]
                
                # Evaluate candidate
                f_candidate = func(x_candidate)
                bo_call_count[0] += 1
                
                if f_candidate < best_f:
                    best_f = f_candidate
                    best_x = x_candidate.copy()
                
                # Add to training data
                X_data = np.vstack([X_data, x_candidate.reshape(1, -1)])
                y_data = np.append(y_data, f_candidate)
                
                # Update length scale based on data spread
                if len(X_data) > 5:
                    data_spread = np.std(X_data, axis=0)
                    length_scale = np.mean(data_spread) + 1e-6
                    kernel = Matern(length_scale=length_scale, nu=2.5)
        
        remaining = self.budget - calls_used - call_count[0] - bo_call_count[0] if 'bo_call_count' in locals() else self.budget - calls_used - call_count[0]
        
        if time.time() > deadline - 15:
            return best_x, best_f

        # Phase 4: Additional local optimizations from perturbed starts
        if remaining > 30:
            n_restarts = min(5, remaining // 15)
            for i in range(n_restarts):
                if time.time() > deadline - 10:
                    break
                if remaining < 15:
                    break
                
                # Perturb the best point with decreasing magnitude
                scale = 0.5 * (1.0 - i / n_restarts) + 0.1
                perturbation = rng.normal(0, scale, self.dim)
                start_x = np.clip(best_x + perturbation, -5, 5)
                
                restart_count = [0]
                
                def restart_wrapper(x, rc=restart_count):
                    if rc[0] >= remaining:
                        raise ValueError("Budget exceeded")
                    if time.time() > deadline - 5:
                        raise ValueError("Time exceeded")
                    rc[0] += 1
                    return func(x)
                
                try:
                    res_r = minimize(
                        restart_wrapper, 
                        start_x, 
                        method='L-BFGS-B',
                        bounds=[(-5, 5)] * self.dim,
                        options={'maxiter': 200, 'maxfun': remaining, 'maxls': 15, 'gtol': 1e-6}
                    )
                    
                    if res_r.fun < best_f:
                        best_f = res_r.fun
                        best_x = res_r.x.copy()
                        
                except Exception:
                    pass
                
                remaining = self.budget - calls_used - call_count[0] - (bo_call_count[0] if 'bo_call_count' in locals() else 0) - restart_count[0]

        # Phase 5: Final Nelder-Mead polish if budget remains
        total_used = calls_used + call_count[0] + (bo_call_count[0] if 'bo_call_count' in locals() else 0)
        remaining = self.budget - total_used
        
        if remaining > 50 and time.time() < deadline - 20:
            nm_count = [0]
            nm_remaining = min(remaining - 10, 100)  # Cap NM at 100 calls
            
            def nm_wrapper(x):
                if nm_count[0] >= nm_remaining:
                    raise ValueError("NM budget exceeded")
                if time.time() > deadline - 5:
                    raise ValueError("Time exceeded")
                nm_count[0] += 1
                return func(x)
            
            try:
                res_nm = minimize(
                    nm_wrapper, 
                    best_x, 
                    method='Nelder-Mead',
                    options={'maxiter': 100, 'xatol': 1e-8, 'fatol': 1e-10, 'adaptive': True}
                )
                
                if res_nm.fun < best_f:
                    best_f = res_nm.fun
                    best_x = res_nm.x.copy()
                    
            except Exception:
                pass

        return best_x, best_f
