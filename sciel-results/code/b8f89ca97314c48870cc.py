import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0
        
        dim = self.dim
        budget = self.budget
        
        # Portfolio strategy: multiple (1+1)-ES runs
        # Number of restarts. With budget 500, we can afford many.
        # Let's try 10 restarts, each with 50 evaluations.
        n_restarts = 10
        budget_per_run = budget // n_restarts
        
        best_x_global = None
        best_f_global = np.inf
        
        # Initial population of starting points for the restarts
        # We can use deterministic points or random.
        # Let's use a mix of center and random points to cover the space.
        # Or just random points in [-4, 4]^5 as the minimizer is there.
        
        for r in range(n_restarts):
            if time.time() > deadline:
                break
            
            # Check if we have enough budget left
            if budget - budget_per_run < 0:
                budget_per_run = max(10, budget) # Ensure at least some steps
            else:
                budget_per_run = budget // (n_restarts - r)
            
            # Initialize (1+1)-ES
            # Start point: random in [-4, 4]
            x = np.random.uniform(-4, 4, dim)
            f_x = func(x)
            budget -= 1
            
            if f_x < best_f_global:
                best_f_global = f_x
                best_x_global = x.copy()
            
            # Initial step size
            sigma = 1.0
            
            # 1/5 success rule parameters
            # If success rate > 1/5, increase sigma
            # If success rate < 1/5, decrease sigma
            
            # We will track successes in a window or just adapt every step.
            # Standard 1/5 rule: adjust sigma based on recent history.
            # Simpler: adjust every step based on whether last step was success?
            # No, that's noisy. Let's use a window of 10.
            
            successes = []
            window_size = 10
            
            for i in range(budget_per_run):
                if time.time() > deadline:
                    break
                
                # Generate neighbor
                z = np.random.randn(dim)
                x_new = x + sigma * z
                # Clip to box
                x_new = np.clip(x_new, -5, 5)
                
                f_new = func(x_new)
                budget -= 1
                
                # Check success
                if f_new < f_x:
                    success = 1
                    x = x_new
                    f_x = f_new
                    if f_x < best_f_global:
                        best_f_global = f_x
                        best_x_global = x.copy()
                else:
                    success = 0
                
                successes.append(success)
                if len(successes) > window_size:
                    successes.pop(0)
                
                # Adapt sigma if we have enough history
                if len(successes) == window_size:
                    success_rate = np.mean(successes)
                    if success_rate > 0.2:
                        sigma *= 1.1
                    else:
                        sigma *= 1/1.1
                    
                    # Clip sigma to reasonable range
                    sigma = np.clip(sigma, 1e-5, 10.0)

        return best_x_global, best_f_global
