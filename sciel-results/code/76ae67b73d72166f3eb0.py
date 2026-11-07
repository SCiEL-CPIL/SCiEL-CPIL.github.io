import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        # Safety margin: The limit is 300s (or 330s kill). 280s is safe.
        deadline = start_time + 280.0
        
        rng = np.random.default_rng(42)
        dim = self.dim
        budget = self.budget
        
        # Global best tracking
        best_x = None
        best_f = np.inf
        calls_made = 0
        
        # Strategy: Multi-start (1+1)-ES with 1/5 success rule
        # We split the budget into a few "phases" or "restarts".
        # Let's use 3 restarts.
        n_restarts = 3
        # Reserve some budget for a final polishing phase if possible
        # Let's allocate 80% of budget to the main search and 20% to polishing
        main_budget = int(budget * 0.8)
        polish_budget = budget - main_budget
        
        budget_per_restart = main_budget // n_restarts
        # Adjust last restart to take remaining main budget
        restart_budgets = [budget_per_restart] * (n_restarts - 1)
        restart_budgets.append(main_budget - sum(restart_budgets))
        
        for r_idx in range(n_restarts):
            if calls_made >= budget:
                break
            if time.time() >= deadline:
                break
                
            # Initialize start point
            # For the first start, we could try center, but random is safer for unknown shift
            x = rng.uniform(-5, 5, dim)
            fx = func(x)
            calls_made += 1
            
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
                
            if calls_made >= budget:
                break

            # (1+1)-ES with 1/5 success rule
            sigma = 2.0
            sigma_min = 1e-5
            sigma_max = 10.0
            
            # 1/5 rule state
            success_count = 0
            trial_count = 0
            max_trials_in_phase = restart_budgets[r_idx]
            
            for step in range(max_trials_in_phase):
                if calls_made >= budget:
                    break
                if time.time() >= deadline:
                    break
                
                # Generate candidate
                y = np.clip(x + sigma * rng.standard_normal(dim), -5, 5)
                fy = func(y)
                calls_made += 1
                
                # Update global best
                if fy < best_f:
                    best_f = fy
                    best_x = y.copy()
                
                # 1/5 success rule logic
                if fy < fx:
                    x = y
                    fx = fy
                    success_count += 1
                else:
                    pass # x and fx unchanged
                
                trial_count += 1
                
                # Adapt sigma every 5 trials
                if trial_count >= 5:
                    if success_count >= 1:
                        # Success: increase sigma
                        sigma *= 1.5
                    else:
                        # Failure: decrease sigma
                        sigma *= 1.0 / 1.5
                    
                    sigma = np.clip(sigma, sigma_min, sigma_max)
                    success_count = 0
                    trial_count = 0
            
            # If we haven't used all budget, the next restart will start fresh

        # Polishing Phase
        # If we have budget left and time, refine the global best
        if calls_made < budget and best_x is not None and time.time() < deadline:
            x = best_x.copy()
            fx = best_f
            sigma = 0.5 # Start with a smaller sigma for refinement
            sigma_min = 1e-6
            sigma_max = 2.0
            
            success_count = 0
            trial_count = 0
            remaining = budget - calls_made
            
            for _ in range(remaining):
                if time.time() >= deadline:
                    break
                
                y = np.clip(x + sigma * rng.standard_normal(dim), -5, 5)
                fy = func(y)
                calls_made += 1
                
                if fy < fx:
                    x = y
                    fx = fy
                    if fx < best_f:
                        best_f = fx
                        best_x = x.copy()
                    success_count += 1
                
                trial_count += 1
                
                if trial_count >= 5:
                    if success_count >= 1:
                        sigma *= 1.5
                    else:
                        sigma *= 1.0 / 1.5
                    sigma = np.clip(sigma, sigma_min, sigma_max)
                    success_count = 0
                    trial_count = 0

        # Fallback if no calls were made (should not happen)
        if best_x is None:
            best_x = np.zeros(dim)
            best_f = func(best_x)
            calls_made += 1

        return best_x, best_f
