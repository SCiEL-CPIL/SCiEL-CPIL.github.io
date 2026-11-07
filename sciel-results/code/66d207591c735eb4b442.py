import numpy as np
from scipy.optimize import minimize

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        
        # CMA-ES Parameters
        # Population size lambda
        lambda_ = 4 + int(3 * np.log(dim))
        if lambda_ < 10: lambda_ = 10
        if lambda_ > 20: lambda_ = 20
        
        mu = lambda_ // 2
        # Weights
        weights = np.log(mu + 1) - np.log(np.arange(1, mu + 1))
        weights = weights / np.sum(weights)
        w_eff = np.sum(weights)**2 / np.sum(weights**2)
        
        # Strategy parameters
        c_sigma = (w_eff + 2) / (dim + w_eff + 5)
        d_sigma = 1 + 2 * max(0, np.sqrt((w_eff - 1) / (dim + 1)) - 1) + c_sigma
        c_1 = 2 / ((dim + 1.3)**2 + w_eff)
        c_mu = 2 * (w_eff - 2 + 1 / w_eff) / ((dim + 2)**2 + w_eff)
        
        # Initial state
        x_mean = np.zeros(dim)
        sigma = 2.0 # Large initial step to cover [-5, 5]
        C = np.eye(dim)
        C_inv_sqrt = np.eye(dim)
        
        # Path accumulators
        p_sigma = np.zeros(dim)
        p_c = np.zeros(dim)
        
        best_x = None
        best_f = np.inf
        calls = 0
        
        # Pre-compute Cholesky factor
        L = np.linalg.cholesky(C)
        
        # Max iterations
        max_iter = int(budget / lambda_)
        
        for it in range(max_iter):
            # Generate population
            # z ~ N(0, I)
            Z = np.random.randn(lambda_, dim)
            # x = x_mean + sigma * (L @ z.T).T
            X = x_mean + sigma * (Z @ L.T)
            
            # Evaluate
            F = np.array([func(x) for x in X])
            calls += lambda_
            
            # Update best
            for i in range(lambda_):
                if F[i] < best_f:
                    best_f = F[i]
                    best_x = X[i].copy()
            
            # Sort indices
            idx = np.argsort(F)
            X_sorted = X[idx]
            F_sorted = F[idx]
            
            # Update mean
            x_mean_new = np.dot(weights, X_sorted[:mu])
            
            # Update step size (sigma)
            # y_mean = (x_mean_new - x_mean) / sigma
            y_mean = (x_mean_new - x_mean) / sigma
            p_sigma = (1 - c_sigma) * p_sigma + c_sigma * y_mean
            
            # h_sigma condition (simplified)
            # Check if p_sigma is within expected range
            # E[||p_sigma||] approx
            # We use a simpler check or just update
            # Standard CMA-ES uses h_sigma to prevent divergence
            # Here we use a simplified version
            h_sigma = 1.0
            # More accurate h_sigma:
            # c_sigma * (1 - (1 - c_sigma)^(2 * (it+1)))
            # This is getting complex. Let's use a robust fixed h_sigma or simplified update.
            # For robustness, let's just update sigma based on p_sigma norm.
            # If p_sigma norm is too large, increase sigma? No, if it's too large, it means we are moving fast, maybe keep sigma.
            # Standard update:
            sigma = sigma * np.exp((1.0 / (d_sigma * c_sigma)) * (np.linalg.norm(p_sigma) / np.sqrt(2 * dim) - 1.0))
            # Clamp sigma
            if sigma < 1e-8:
                break
            if sigma > 10.0:
                sigma = 10.0

            # Update Covariance
            # y_i = (x_i - x_mean) / sigma
            Y = (X_sorted[:mu] - x_mean) / sigma
            C_delta = np.dot(weights[:, np.newaxis] * Y, Y.T)
            
            # p_c update
            h_sigma_val = 1.0 # Simplified
            # Actually, we need to be careful.
            # Let's use the standard formula with h_sigma approx 1 if we don't track history perfectly.
            # Or just use the rank-mu update without the rank-1 correction for simplicity?
            # No, rank-1 is important.
            
            # Let's implement a safer CMA-ES variant that is less prone to implementation errors:
            # We will use the 'CMA-ES' from scipy if possible? No.
            # Let's stick to the code above but ensure numerical stability.
            
            # Update C
            C = (1 - c_1 - c_mu) * C + c_1 * np.outer(p_sigma, p_sigma) + c_mu * C_delta
            # Symmetrize
            C = (C + C.T) / 2
            
            # Ensure positive definite
            try:
                L = np.linalg.cholesky(C + 1e-8 * np.eye(dim))
            except:
                # If not PD, reset C or regularize
                C = C + 1e-4 * np.eye(dim)
                L = np.linalg.cholesky(C)
            
            x_mean = x_mean_new
            
            # Check convergence
            if np.linalg.norm(x_mean_new - x_mean) < 1e-8:
                break

        # Use remaining budget for local refinement
        if calls < budget and best_x is not None:
            # Nelder-Mead on best_x
            remaining = budget - calls
            # Simple Nelder-Mead implementation
            # Or just random hill climbing
            # Let's do a simple local search with decreasing step size
            current_x = best_x.copy()
            current_f = best_f
            step = 0.5
            for _ in range(remaining):
                # Try a few random directions
                # Actually, let's just do a few Nelder-Mead steps manually or use scipy
                # Since we have budget, let's use scipy.optimize.minimize with Nelder-Mead
                # But we need to count calls.
                # We'll do a manual hill climb to save code complexity
                for _ in range(min(5, remaining)):
                    if remaining <= 0: break
                    dx = np.random.randn(dim) * step
                    new_x = np.clip(current_x + dx, -5, 5)
                    new_f = func(new_x)
                    remaining -= 1
                    if new_f < current_f:
                        current_x = new_x
                        current_f = new_f
                        if current_f < best_f:
                            best_f = current_f
                            best_x = current_x.copy()
                step *= 0.7
                if step < 1e-5:
                    break

        return best_x, best_f
