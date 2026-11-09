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
        time_limit = 325.0 

        # 1. Initialization: Use a mix of random and structured points
        # Start with center, then random points
        init_points = np.zeros((1, n))  # center
        init_points = np.vstack([init_points, np.random.uniform(-5, 5, (9, n))])
        
        best_init_val = np.inf
        best_init_point = init_points[0]
        
        for i in range(10):
            if time.time() - start_time > time_limit:
                break
            x = init_points[i]
            fx = func(x)
            if fx < best_init_val:
                best_init_val = fx
                best_init_point = x.copy()
        
        x = best_init_point.copy()
        fx = best_init_val
        x_best = x.copy()
        f_best = fx
        eval_count = 10
        
        # 2. (1+1)-ES with improved sigma adaptation and restarts
        # Use the more aggressive adaptation from Inspiration 1 (1.2 / 0.85)
        # This allows for faster convergence in smooth regions (Sphere, Rosenbrock)
        # while still being able to explore.
        sigma = 2.0
        sigma_min = 1e-8
        sigma_max = 10.0
        
        steps_since_best_improvement = 0
        restart_threshold = 30
        
        # Track multiple local optima for diversity (from Inspiration 2)
        local_optima = [(x_best.copy(), f_best)]
        max_local_optima = 5
        
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
                    # Update local optima: add the new best
                    local_optima.append((x_best.copy(), f_best))
                    # Sort by value and keep only the best 5
                    local_optima.sort(key=lambda t: t[1])
                    local_optima = local_optima[:max_local_optima]
                else:
                    steps_since_best_improvement += 1
            else:
                # Failure
                sigma *= 0.85
                steps_since_best_improvement += 1
                
            # Clamp sigma
            sigma = np.clip(sigma, sigma_min, sigma_max)
            
            # Restart mechanism: if stuck, try from a different local optimum
            if steps_since_best_improvement >= restart_threshold:
                # Pick a random local optimum or the best one
                # Bias towards better ones but allow exploration
                if len(local_optima) > 1:
                    # Simple random selection from stored optima
                    idx = np.random.randint(len(local_optima))
                    x = local_optima[idx][0].copy()
                    fx = local_optima[idx][1]
                else:
                    x = x_best.copy()
                    fx = f_best
                
                # Reset sigma to a larger value for exploration
                sigma = 2.0
                steps_since_best_improvement = 0
                
                # Occasionally add a new random point to local_optima for diversity
                if np.random.random() < 0.2 and len(local_optima) < max_local_optima:
                    x_new = np.random.uniform(-5, 5, n)
                    try:
                        f_new = func(x_new)
                        eval_count += 1
                        local_optima.append((x_new.copy(), f_new))
                        local_optima.sort(key=lambda t: t[1])
                        local_optima = local_optima[:max_local_optima]
                    except Exception:
                        pass

        # 3. Final Polish: Use remaining budget for high-precision local search
        # This is a refinement from the current program which scored well.
        # We use a smaller sigma and gentler adaptation for fine-tuning.
        if eval_count < budget and time.time() - start_time < time_limit:
            # Start from the best point with a small sigma
            x = x_best.copy()
            fx = f_best
            sigma = 0.5  # Start polish with moderate sigma, not too small
            
            polish_steps = 0
            max_polish_steps = min(budget - eval_count, 50)
            
            for _ in range(max_polish_steps):
                if eval_count >= budget or time.time() - start_time > time_limit:
                    break
                
                z = np.random.standard_normal(n)
                y = x + sigma * z
                y = np.clip(y, -5, 5)
                
                try:
                    fy = func(y)
                except Exception:
                    break
                    
                eval_count += 1
                polish_steps += 1
                
                if fy < fx:
                    x = y
                    fx = fy
                    if fy < f_best:
                        f_best = fy
                        x_best = y.copy()
                    sigma *= 0.95  # Slightly decrease sigma on success
                else:
                    sigma *= 0.9  # Decrease sigma on failure for faster convergence
                
                sigma = np.clip(sigma, 1e-12, 1.0)
        
        return x_best, f_best
