import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290  # Safety margin for 300s limit
        
        dim = self.dim
        budget = self.budget
        n = dim
        
        # CMA-ES Parameters
        lambda_pop = 10
        mu = lambda_pop // 2
        # Weights
        w = np.log(mu + 1) - np.log(np.arange(1, mu + 1))
        w /= np.sum(w)
        mu_eff = 1.0 / np.sum(w**2)
        
        # Learning rates
        c_sigma = (mu_eff + 2) / (n + mu_eff + 5)
        c_c = (4 + n / 4) / (n + 4 + mu_eff)
        c_1 = 2 / ((n + 1.3)**2 + mu_eff)
        c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((n + 2)**2 + mu_eff))
        chi_N = np.sqrt(n) * (1 - 1 / (4 * n) + 1 / (21 * n**2))
        
        # State
        x_mean = np.zeros(n)
        sigma = 1.0
        B = np.eye(n)
        D = 1.0
        C = np.eye(n)
        p_sigma = np.zeros(n)
        p_c = np.zeros(n)
        
        best_x = x_mean.copy()
        best_fx = func(best_x)
        call_count = 1
        
        # Main CMA-ES Loop
        while call_count + lambda_pop <= budget:
            if time.time() > deadline:
                break
                
            # Sample lambda points
            ys = np.random.multivariate_normal(np.zeros(n), C, size=lambda_pop)
            xs = x_mean + sigma * (B @ (D * ys).T).T
            
            # Evaluate
            fvals = np.array([func(x) for x in xs])
            call_count += lambda_pop
            
            # Update best
            idx_best = np.argmin(fvals)
            if fvals[idx_best] < best_fx:
                best_fx = fvals[idx_best]
                best_x = xs[idx_best].copy()
                
            # Sort indices by fitness
            sorted_indices = np.argsort(fvals)
            ys_sorted = ys[sorted_indices]
            xs_sorted = xs[sorted_indices]
            
            # Update mean
            y_w = w @ ys_sorted[:mu]
            x_mean_new = x_mean + sigma * (B @ (D * y_w))
            
            # Update sigma (CSA)
            p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) / np.linalg.norm(B @ (D * y_w)) * (B @ (D * y_w))
            h_sigma = 1 if np.linalg.norm(p_sigma) / np.sqrt(1 - (1 - c_sigma)**2) < (1.4 + 2 / (n + 1)) * chi_N else 0
            sigma *= np.exp(c_sigma / c_sigma * (np.linalg.norm(p_sigma) / chi_N - 1))
            
            # Update C (CMA)
            p_c = (1 - c_c) * p_c + h_sigma * np.sqrt(c_c * (2 - c_c) * mu_eff) * y_w
            C = (1 + c_1 * h_sigma * (mu_eff - 2) - c_1) * C + \
                c_1 * np.outer(y_w, y_w) + \
                c_mu * np.sum(w[:mu] * np.outer(ys_sorted[:mu], ys_sorted[:mu]), axis=0)
            
            # Ensure C is symmetric
            C = (C + C.T) / 2
            
            # Eigen decomposition for B and D
            try:
                D_sq, B = np.linalg.eigh(C)
                D_sq = np.clip(D_sq, 1e-12, None)
                D = np.sqrt(D_sq)
            except np.linalg.LinAlgError:
                # Fallback if eigen fails
                D = 1.0
                B = np.eye(n)
                
            x_mean = x_mean_new
            
            # Convergence check
            if sigma < 1e-12:
                break

        # Polish Phase: Newton's Method with Hessian Estimation
        # We have budget - call_count calls left.
        # Estimating Hessian in 5D costs 15 calls (central diff).
        # We can do several Newton steps.
        
        x_curr = best_x.copy()
        fx_curr = best_fx
        
        # Check if we have enough budget for Hessian estimation
        # Central diff Hessian: 2 * n * (n+1)/2 ? No, just n(n+1)/2 second derivatives.
        # Actually, for symmetric Hessian, we need n(n+1)/2 unique elements.
        # Central difference for Hessian element H_ij:
        # f(x+h e_i + h e_j) - f(x+h e_i - h e_j) - f(x-h e_i + h e_j) + f(x-h e_i - h e_j) / (4 h^2)
        # This is expensive. 
        # Alternative: Use L-BFGS-B from scipy if available? 
        # The prompt says SciPy is available.
        
        from scipy.optimize import minimize
        
        # Use L-BFGS-B for polish. It is robust and fast for smooth functions.
        # We need to be careful with the budget.
        # L-BFGS-B uses a callback to count function calls.
        
        if call_count < budget - 5:
            # Define a counter
            counter = {'count': call_count}
            
            def counted_func(x):
                if counter['count'] >= budget:
                    return fx_curr # Return current best to stop? Or raise?
                    # Better to just return a large value or current best
                    # But L-BFGS might get confused.
                    # Let's just let it run until it exceeds, and break if time/budget.
                    # Actually, we can wrap the function to check budget.
                
                val = func(x)
                counter['count'] += 1
                return val
            
            # We want to minimize from x_curr
            # Bounds are [-5, 5]
            bounds = [(-5.0, 5.0)] * n
            
            try:
                res = minimize(counted_func, x_curr, method='L-BFGS-B', bounds=bounds, 
                               options={'maxiter': 100, 'ftol': 1e-15, 'gtol': 1e-15})
                if res.fun < best_fx:
                    best_fx = res.fun
                    best_x = res.x
            except Exception:
                pass # If L-BFGS fails, keep previous best

        return best_x, best_fx
