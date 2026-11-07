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
        
        # Stochastic Gradient Descent with adaptive step size
        lr = 1.0
        best_x = x.copy()
        best_fx = fx
        no_improve_count = 0
        max_no_improve = 20
        
        while calls < self.budget:
            # Estimate gradient using random perturbations
            # Use a small number of random directions to estimate gradient
            n_perturb = min(dim, max(2, (self.budget - calls) // 10))
            h = 1e-3
            
            grad_est = np.zeros(dim)
            
            for _ in range(n_perturb):
                if calls + 1 > self.budget:
                    break
                    
                # Random direction
                direction = rng.standard_normal(dim)
                direction /= np.linalg.norm(direction)
                
                # Evaluate at x + h * direction
                x_plus = x + h * direction
                f_plus = func(x_plus)
                calls += 1
                
                # Gradient estimate contribution
                grad_est += (f_plus - fx) / h * direction
            
            # Average gradient estimate
            if n_perturb > 0:
                grad_est /= n_pperturb if n_perturb > 0 else 1
            else:
                grad_est /= n_perturb
            
            # Check convergence
            if np.linalg.norm(grad_est) < 1e-8:
                break
            
            # Take step in negative gradient direction
            step = -grad_est / (np.linalg.norm(grad_est) + 1e-10) * lr
            
            # Try the step
            x_trial = x + step
            f_trial = func(x_trial)
            calls += 1
            
            # Adaptive learning rate
            if f_trial < fx:
                # Improvement
                x = x_trial
                fx = f_trial
                no_improve_count = 0
                # Increase learning rate slightly
                lr = min(lr * 1.1, 5.0)
                
                if fx < best_fx:
                    best_fx = fx
                    best_x = x.copy()
            else:
                # No improvement
                no_improve_count += 1
                # Decrease learning rate
                lr = max(lr * 0.7, 0.01)
                
                if no_improve_count >= max_no_improve:
                    # Reset to best point and try larger steps
                    x = best_x.copy()
                    fx = best_fx
                    lr = 2.0
                    no_improve_count = 0
                    max_no_improve = 15
        
        # Return the best point found
        return best_x, best_fx
