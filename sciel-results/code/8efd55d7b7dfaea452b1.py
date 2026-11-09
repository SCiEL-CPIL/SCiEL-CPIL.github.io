import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        n = self.dim
        budget = self.budget
        
        # Time management
        start_time = time.time()
        # 330s limit, use 325s as hard stop to be safe against overhead.
        time_limit = 325.0 

        # 1. Initialization: Scout for a good starting point
        # Evaluate 10 random points
        init_candidates = np.random.uniform(-5, 5, (10, n))
        best_init_val = np.inf
        best_init_point = init_candidates[0]
        
        # We must ensure we don't exceed budget even in init, though 10 is small
        # But we must respect the time limit too
        for i in range(10):
            if time.time() - start_time > time_limit:
                break
            x = init_candidates[i]
            fx = func(x)
            if fx < best_init_val:
                best_init_val = fx
                best_init_point = x.copy()
        
        x = best_init_point.copy()
        fx = best_init_val
        x_best = x.copy()
        f_best = fx
        eval_count = 10
        
        # 2. (1+1)-ES with adaptive step size and restarts
        sigma = 2.0 # Initial step size
        sigma_min = 1e-8
        sigma_max = 10.0
        
        # Parameters for 1/5 rule approximation
        # We can use a window or just simple multiplicative updates.
        # Simple multiplicative:
        # Success: sigma *= 1.2
        # Failure: sigma *= 0.85
        # This is roughly equivalent to 1/5 rule for n=5?
        # For n=5, 1/5 rule suggests sigma update based on success rate.
        # Let's stick to simple multiplicative as it's fast and effective for small n.
        
        steps_since_best_improvement = 0
        restart_threshold = 30 # Restart local search if no global best improvement for 30 steps
        
        while eval_count < budget:
            if time.time() - start_time > time_limit:
                break
            
            # Generate candidate
            z = np.random.standard_normal(n)
            y = x + sigma * z
            y = np.clip(y, -5, 5)
            
            try:
                fy = func(y)
            except Exception:
                break
                
            eval_count += 1
            
            if fy < fx:
                # Success
                x = y
                fx = fy
                sigma *= 1.2
                if fy < f_best:
                    f_best = fy
                    x_best = y.copy()
                    steps_since_best_improvement = 0
                else:
                    steps_since_best_improvement += 1
            else:
                # Failure
                fx = fx # Keep x
                sigma *= 0.85
                steps_since_best_improvement += 1
                
            # Clamp sigma
            if sigma > sigma_max:
                sigma = sigma_max
            if sigma < sigma_min:
                sigma = sigma_min
                
            # Restart mechanism
            # If we haven't improved the global best in 'restart_threshold' steps,
            # reset the local search to the best point with a larger step size
            # This helps escape local minima in multimodal functions
            if steps_since_best_improvement >= restart_threshold:
                x = x_best.copy()
                fx = f_best
                sigma = 2.0 # Reset to a reasonable exploration size
                steps_since_best_improvement = 0
                
        # 3. Final Polish: Use remaining budget for a high-precision local search
        # This addresses the "final high-precision polish pass" directive.
        # We use a smaller sigma and more iterations to fine-tune the best point found.
        if eval_count < budget and time.time() - start_time < time_limit:
            # Use a fraction of remaining budget for polish
            remaining = budget - eval_count
            polish_budget = min(remaining, 50) # Cap polish at 50 steps
            polish_sigma = max(sigma, 0.1) * 0.1 # Start with smaller sigma
            x_polish = x_best.copy()
            f_polish = f_best
            
            for i in range(polish_budget):
                if time.time() - start_time > time_limit:
                    break
                if eval_count >= budget:
                    break
                    
                z = np.random.standard_normal(n)
                y = x_polish + polish_sigma * z
                y = np.clip(y, -5, 5)
                
                try:
                    fy = func(y)
                except Exception:
                    break
                    
                eval_count += 1
                
                if fy < f_polish:
                    x_polish = y
                    f_polish = fy
                    polish_sigma *= 1.05 # Slightly increase if successful
                else:
                    polish_sigma *= 0.9 # Decrease if failed
                    
                if polish_sigma < sigma_min:
                    polish_sigma = sigma_min
                    
            if f_polish < f_best:
                f_best = f_polish
                x_best = x_polish.copy()
                
        return x_best, f_best
