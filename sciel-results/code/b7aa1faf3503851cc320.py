import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        rng = np.random.default_rng(42)
        n = self.dim
        budget = self.budget
        
        # Deadline guard
        start_time = time.time()
        max_time = 290.0 # 300s limit, keep 10s margin
        
        best_x = None
        best_f = np.inf
        
        # Phase 1: Global search with Simulated Annealing
        n_restarts_global = 10
        budget_per_restart_global = (budget // n_restarts_global) - 2
        
        for r in range(n_restarts_global):
            if time.time() - start_time > max_time:
                break
                
            # Initialize random point
            x = rng.uniform(-5, 5, n)
            fx = func(x)
            
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            
            # SA parameters for this restart
            T = 5.0 
            alpha = 0.95
            sigma = 1.5
            
            # Run SA for budget_per_restart steps
            for i in range(budget_per_restart_global):
                if time.time() - start_time > max_time:
                    break
                
                y = x + sigma * rng.standard_normal(n)
                y = np.clip(y, -5, 5)
                fy = func(y)
                
                delta = fy - fx
                
                # Acceptance probability
                if delta < 0 or rng.random() < np.exp(-delta / T):
                    x = y
                    fx = fy
                    if fx < best_f:
                        best_f = fx
                        best_x = x.copy()
                    # Adapt step size: increase on success
                    sigma *= 1.02
                else:
                    # Adapt step size: decrease on failure
                    sigma *= 0.98
                
                T *= alpha
        
        # Phase 2: Local Polish around the best point found
        # Use remaining budget for a more precise local search
        calls_used = 0
        # Estimate calls used in phase 1: roughly n_restarts_global * (1 + budget_per_restart_global)
        # But we don't track exactly, so we estimate based on budget allocation.
        # We reserved 2 calls per restart, so total calls in phase 1 <= n_restarts_global * (1 + budget_per_restart_global)
        # budget_per_restart_global = (budget // n_restarts_global) - 2
        # Total calls <= n_restarts_global * (1 + (budget // n_restarts_global) - 2)
        #           = n_restarts_global * (budget // n_restarts_global - 1)
        #           <= budget - n_restarts_global
        # So we have at least n_restarts_global calls left. Let's use up to 50 calls for polish.
        
        if best_x is not None and time.time() - start_time < max_time:
            polish_budget = min(50, budget - int(n_restarts_global * (budget_per_restart_global + 1)))
            if polish_budget > 0:
                x = best_x.copy()
                fx = best_f
                sigma = 0.5
                T = 0.1
                alpha = 0.9
                
                for i in range(polish_budget):
                    if time.time() - start_time > max_time:
                        break
                    
                    y = x + sigma * rng.standard_normal(n)
                    y = np.clip(y, -5, 5)
                    fy = func(y)
                    
                    delta = fy - fx
                    
                    # Stricter acceptance for polish
                    if delta < 0 or rng.random() < np.exp(-delta / T):
                        x = y
                        fx = fy
                        if fx < best_f:
                            best_f = fx
                            best_x = x.copy()
                        sigma *= 1.01
                    else:
                        sigma *= 0.95
                    
                    T *= alpha
        
        return best_x, best_f
