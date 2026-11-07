import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        total_budget = self.budget
        calls_made = 0
        best_f = float('inf')
        best_x = np.zeros(dim)
        
        start_time = time.time()
        # Safety margin: 330s limit, use 325s to be safe
        deadline = start_time + 325.0

        def wrapper(x):
            nonlocal calls_made, best_f, best_x
            if time.time() > deadline:
                raise TimeoutError("Deadline exceeded")
            calls_made += 1
            x = np.clip(x, -5, 5)
            try:
                fx = func(x)
            except Exception:
                fx = float('inf')
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            return fx

        # (1+1)-ES with 1/5 success rule
        def es_1plus1(start_x, budget_remaining, rng, alpha=1.0):
            nonlocal calls_made, best_f, best_x
            
            n = dim
            x = start_x.copy()
            fx = wrapper(x)
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
                
            sigma = 1.0 # Initial step size, domain is [-5, 5], so 1.0 is reasonable
            # Success history
            successes = []
            window_size = 10
            
            # Run until budget or time is up
            while calls_made < total_budget and time.time() < deadline:
                # Generate candidate
                z = rng.standard_normal(n)
                x_trial = x + sigma * z
                x_trial = np.clip(x_trial, -5, 5)
                
                fx_trial = wrapper(x_trial)
                
                # Update best
                if fx_trial < best_f:
                    best_f = fx_trial
                    best_x = x_trial.copy()
                
                # Accept or Reject
                if fx_trial < fx:
                    x = x_trial
                    fx = fx_trial
                    successes.append(1)
                else:
                    successes.append(0)
                    
                # Adapt sigma based on 1/5 success rule
                if len(successes) >= window_size:
                    success_rate = np.mean(successes[-window_size:])
                    if success_rate > 0.2:
                        sigma *= 1.05 # Increase
                    elif success_rate < 0.2:
                        sigma /= 1.05 # Decrease
                    # Cap sigma to prevent divergence
                    sigma = np.clip(sigma, 1e-6, 10.0)
                    
                # Stagnation check: if no improvement in best_f for last 20 calls, break to allow restart
                # We track this in the outer loop by checking best_f changes, 
                # but here we can just run for a fixed number of steps or until time.
                # For simplicity, we let the outer loop manage restarts by calling this function
                # with a limited budget per phase.

            return x, fx

        rng = np.random.default_rng(42)
        
        # Strategy:
        # 1. Perform a few random initializations.
        # 2. For each, run (1+1)-ES for a chunk of budget.
        # 3. If best_f hasn't improved significantly, try a new start.
        
        # We will divide the budget into phases.
        # Phase 1: Exploration with multiple starts.
        # Phase 2: Exploitation on the best found.
        
        # Let's try 3 restarts initially.
        num_restarts = 3
        budget_per_restart = int(total_budget * 0.3) # 30% each
        
        # Initial random starts
        starts = []
        # One start at center
        starts.append(np.zeros(dim))
        # Two random starts
        for _ in range(num_restarts - 1):
            starts.append(rng.uniform(-4, 4, dim))
            
        for i, start_x in enumerate(starts):
            if calls_made >= total_budget or time.time() > deadline:
                break
                
            # Run ES from this start
            # We give it a specific budget slice, but the wrapper stops it if global budget/time is hit
            # We just call it, and it will run until global limits are hit or it finishes its internal loop (which is infinite here, so it relies on global limits).
            # To prevent one restart from eating all budget, we can't easily limit it inside es_1plus1 without passing a counter.
            # Instead, we rely on the fact that if it stagnates, the sigma will shrink and it will become inefficient.
            # But better: we can modify es_1plus1 to take a max_steps parameter.
            
            # Let's refine es_1plus1 to take a max_steps
            # Actually, the current es_1plus1 runs until global budget. 
            # This is bad for restarts. 
            # Let's rewrite es_1plus1 to accept a step_limit.
            
            pass # Placeholder, will implement below

        # Re-implementing es_1plus1 with step_limit
        def es_1plus1_limited(start_x, step_limit, rng):
            nonlocal calls_made, best_f, best_x
            
            n = dim
            x = start_x.copy()
            fx = wrapper(x)
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
                
            sigma = 1.0
            successes = []
            window_size = 10
            steps = 0
            
            while steps < step_limit and calls_made < total_budget and time.time() < deadline:
                z = rng.standard_normal(n)
                x_trial = x + sigma * z
                x_trial = np.clip(x_trial, -5, 5)
                
                fx_trial = wrapper(x_trial)
                
                if fx_trial < best_f:
                    best_f = fx_trial
                    best_x = x_trial.copy()
                
                if fx_trial < fx:
                    x = x_trial
                    fx = fx_trial
                    successes.append(1)
                else:
                    successes.append(0)
                    
                if len(successes) >= window_size:
                    success_rate = np.mean(successes[-window_size:])
                    if success_rate > 0.2:
                        sigma *= 1.05
                    elif success_rate < 0.2:
                        sigma /= 1.05
                    sigma = np.clip(sigma, 1e-6, 10.0)
                    
                steps += 1

            return x, fx

        # Execute Restart Strategy
        # We will try 3 starts, each with 150 steps (75% of budget total).
        # If we have budget left, we do a final polish on the best_x with a very small sigma.
        
        starts = [np.zeros(dim), rng.uniform(-4, 4, dim), rng.uniform(-4, 4, dim)]
        
        for start_x in starts:
            if calls_made >= total_budget * 0.75: # Use 75% for exploration/restarts
                break
            # Run for up to 150 steps or until budget/time runs out
            # We use a dynamic step limit: min(150, remaining_budget)
            step_limit = min(150, total_budget - calls_made)
            es_1plus1_limited(start_x, step_limit, rng)
            
        # Final Polish
        if calls_made < total_budget and time.time() < deadline:
            # Use remaining budget for a fine-grained search around best_x
            # Reset sigma to small value
            # We can call es_1plus1_limited again with best_x and a small initial sigma?
            # The function resets sigma to 1.0. 
            # Let's create a specialized polish function or modify es_1plus1_limited to accept initial_sigma.
            
            def es_polish(start_x, step_limit, rng, init_sigma=0.1):
                nonlocal calls_made, best_f, best_x
                n = dim
                x = start_x.copy()
                fx = wrapper(x)
                if fx < best_f:
                    best_f = fx
                    best_x = x.copy()
                    
                sigma = init_sigma
                successes = []
                window_size = 10
                steps = 0
                
                while steps < step_limit and calls_made < total_budget and time.time() < deadline:
                    z = rng.standard_normal(n)
                    x_trial = x + sigma * z
                    x_trial = np.clip(x_trial, -5, 5)
                    
                    fx_trial = wrapper(x_trial)
                    
                    if fx_trial < best_f:
                        best_f = fx_trial
                        best_x = x_trial.copy()
                    
                    if fx_trial < fx:
                        x = x_trial
                        fx = fx_trial
                        successes.append(1)
                    else:
                        successes.append(0)
                        
                    if len(successes) >= window_size:
                        success_rate = np.mean(successes[-window_size:])
                        if success_rate > 0.2:
                            sigma *= 1.02 # Slower adaptation for polish
                        elif success_rate < 0.2:
                            sigma /= 1.02
                        sigma = np.clip(sigma, 1e-8, 1.0)
                        
                    steps += 1

                return x, fx

            remaining_budget = total_budget - calls_made
            if remaining_budget > 0:
                es_polish(best_x, remaining_budget, rng, init_sigma=0.1)

        return best_x, best_f
