import numpy as np

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        rng = np.random.default_rng(42)
        dim = self.dim
        x = np.zeros(dim)
        fx = func(x)
        calls = 1
        
        # Adaptive step size for finite differences
        h = 1e-3
        
        while calls < self.budget:
            # Estimate gradient and Hessian using random perturbations
            # Use a small set of random directions to estimate gradient
            # and second-order differences for Hessian approximation
            
            # Gradient estimation via central differences along random directions
            n_grad_samples = min(5, self.budget - calls)
            if n_grad_samples <= 0:
                break
            
            g_est = np.zeros(dim)
            H_est = np.zeros((dim, dim))
            
            for _ in range(n_grad_samples):
                if calls + 2 > self.budget:
                    break
                
                # Random direction
                d = rng.standard_normal(dim)
                d /= np.linalg.norm(d)
                
                x_plus = x + h * d
                x_minus = x - h * d
                
                f_plus = func(x_plus)
                f_minus = func(x_minus)
                calls += 2
                
                # Gradient contribution: (f_plus - f_minus) / (2h) * d
                g_est += (f_plus - f_minus) / (2 * h) * d
                
                # Hessian approximation: (f_plus - 2*fx + f_minus) / h^2 * d*d^T
                H_est += (f_plus - 2 * fx + f_minus) / (h * h) * np.outer(d, d)
            
            # Average the estimates
            g_est /= n_grad_samples
            H_est /= n_grad_samples
            
            # Add diagonal regularization to Hessian for stability
            H_reg = H_est + 1e-4 * np.eye(dim)
            
            # Check if gradient is small enough
            if np.linalg.norm(g_est) < 1e-8:
                break
            
            # Compute Newton step
            try:
                step = np.linalg.solve(H_reg, -g_est)
            except np.linalg.LinAlgError:
                step = -g_est / np.linalg.norm(g_est)
            
            # Limit step size
            step_norm = np.linalg.norm(step)
            if step_norm > 1.0:
                step /= step_norm
            
            # Line search with backtracking
            alpha = 1.0
            c1 = 1e-4
            rho = 0.5
            found_alpha = False
            best_f = fx
            best_x = x
            
            for _ in range(8):
                if calls + 1 > self.budget:
                    break
                x_trial = x + alpha * step
                f_trial = func(x_trial)
                calls += 1
                
                if f_trial < best_f:
                    best_f = f_trial
                    best_x = x_trial
                
                # Armijo condition
                if f_trial < fx + c1 * alpha * np.dot(g_est, step):
                    found_alpha = True
                    break
                alpha *= rho
            
            if not found_alpha:
                # Use the best point found during line search
                if best_f < fx:
                    x = best_x
                    fx = best_f
                else:
                    # No improvement, break
                    break
            else:
                x = x_trial
                fx = f_trial
            
            # Adaptive h: reduce if we're making progress, increase if not
            if fx < 0:  # Rough heuristic
                h *= 0.9
            else:
                h *= 1.1
            h = np.clip(h, 1e-5, 1e-1)

        return x, fx
