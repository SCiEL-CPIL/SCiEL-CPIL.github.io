import numpy as np
import time

class Optimizer:
    """Multi-start (1+1)-ES with adaptive sigma and independent restarts."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        rng = np.random.default_rng(42)
        
        # Safety margin: 330s limit, stop at 325s
        deadline = time.time() + 325
        
        best_x = None
        best_f = np.inf
        calls = 0
        
        # Number of independent restarts
        n_restarts = 8
        budget_per_restart = max(10, (budget - n_restarts) // n_restarts)
        
        for restart in range(n_restarts):
            if calls >= budget or time.time() > deadline:
                break
            
            # Initialize this restart with a random point
            x = rng.uniform(-5, 5, dim)
            fx = func(x)
            calls += 1
            
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            
            current_x = x.copy()
            current_f = fx
            
            # Adaptive sigma for this restart
            sigma = 2.0
            window_size = 10
            successes = []
            stagnation = 0
            max_stagnation = 15
            
            # Run (1+1)-ES for this restart
            while calls < budget and time.time() < deadline:
                # Generate candidate
                y = current_x + sigma * rng.standard_normal(dim)
                y = np.clip(y, -5, 5)
                
                fy = func(y)
                calls += 1
                
                # Update best
                improved = fy < best_f
                if improved:
                    best_f = fy
                    best_x = y.copy()
                    stagnation = 0
                else:
                    stagnation += 1
                    
                # Update success history for 1/5 rule
                success = fy < current_f
                successes.append(1 if success else 0)
                if len(successes) > window_size:
                    successes.pop(0)
                
                # Adapt sigma based on success rate (1/5 rule)
                if len(successes) == window_size:
                    rate = sum(successes) / window_size
                    if rate > 0.2:
                        sigma *= 1.3
                    elif rate < 0.2:
                        sigma /= 1.3
                
                # Update current position
                if success:
                    current_x = y
                    current_f = fy
                
                # Prevent sigma explosion or vanishing
                sigma = np.clip(sigma, 1e-6, 8.0)
                
                # Stagnation check: break out of this restart if stuck
                if stagnation >= max_stagnation:
                    break
                
                # Check if we've used up this restart's budget
                restart_calls = calls - (budget - budget_per_restart)
                if restart_calls >= budget_per_restart:
                    break
        
        return best_x, best_f
