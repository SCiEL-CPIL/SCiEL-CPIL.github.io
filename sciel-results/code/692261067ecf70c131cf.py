"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-start (1+1)-ES with aggressive sigma adaptation and final polish."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280  # Safety margin before 300s limit
        rng = np.random.default_rng(0)
        d = self.dim
        budget = self.budget

        best_x = None
        best_f = float('inf')
        total_calls = 0

        def check_deadline():
            return time.time() > deadline or total_calls >= budget

        # Phase 1: Multi-start (1+1)-ES
        # We allocate the budget to several independent runs to explore the landscape.
        # Run 0: Start at origin.
        # Runs 1-4: Start at random points.
        
        # Reserve budget for polishing
        polish_budget = 80
        search_budget = budget - polish_budget
        
        # Define the start points
        # 1 origin + 4 random
        n_starts = 5
        # Approximate budget per start
        budget_per_start = search_budget // n_starts
        
        # To be safe, we iterate and check deadlines
        for start_idx in range(n_starts):
            if check_deadline():
                break
            
            # Initialize start point
            if start_idx == 0:
                x = np.zeros(d)
            else:
                x = rng.uniform(-4.5, 4.5, d)
            
            fx = func(x)
            total_calls += 1
            if fx < best_f:
                best_f, best_x = fx, x.copy()
            
            sigma = 2.0
            no_improve = 0
            max_no_improve = 30 # Restart/switch start if stalled
            
            # Run local (1+1)-ES
            while total_calls < search_budget and not check_deadline():
                y = np.clip(x + sigma * rng.standard_normal(d), -5, 5)
                fy = func(y)
                total_calls += 1
                
                if fy < fx:
                    x, fx = y, fy
                    no_improve = 0
                    sigma *= 1.5
                    if fx < best_f:
                        best_f, best_x = fx, x.copy()
                else:
                    sigma *= 0.87
                    no_improve += 1
                
                # If stalled significantly, break to next start
                if no_improve >= max_no_improve:
                    break
                    
                # If we've used our approximate share of the budget, break to next start
                # This is a soft limit; the global search_budget check is the hard limit.
                # We track calls in this specific start to ensure we don't hog the budget.
                # For simplicity, we rely on the global search_budget and the stall condition.
                # However, to ensure all starts get a chance, we should track local calls.
                # Let's add a local call counter.
                # Actually, the loop above doesn't have a local counter. Let's fix that.
                # We can't easily break out of the for loop from inside the while without a flag.
                # Let's refactor to use a local counter.
                
                # Re-evaluating: The current while loop runs until search_budget or stall.
                # If start 0 is very successful, it might consume all search_budget.
                # To prevent this, we should limit each start to roughly budget_per_start.
                
                # Let's restart the logic for the inner loop to include local budget.
                # I will restructure the code below to be cleaner.
                
                pass # Placeholder, actual logic below

        # RESTRUCTURED PHASE 1 for clarity and correctness
        # Reset state for the actual run
        best_x = None
        best_f = float('inf')
        total_calls = 0
        
        # We will run 5 starts.
        # Start 0: Origin. Budget ~100.
        # Starts 1-4: Random. Budget ~80 each.
        
        # To strictly enforce budget per start, we track calls in the inner loop.
        
        for start_idx in range(5):
            if check_deadline():
                break
            
            # Determine start point
            if start_idx == 0:
                x = np.zeros(d)
            else:
                x = rng.uniform(-4.5, 4.5, d)
            
            fx = func(x)
            total_calls += 1
            if fx < best_f:
                best_f, best_x = fx, x.copy()
            
            sigma = 2.0
            no_improve = 0
            local_calls = 1
            max_local_calls = 100 if start_idx == 0 else 80
            max_no_improve = 30
            
            while local_calls < max_local_calls and not check_deadline():
                y = np.clip(x + sigma * rng.standard_normal(d), -5, 5)
                fy = func(y)
                total_calls += 1
                local_calls += 1
                
                if fy < fx:
                    x, fx = y, fy
                    no_improve = 0
                    sigma *= 1.5
                    if fx < best_f:
                        best_f, best_x = fx, x.copy()
                else:
                    sigma *= 0.87
                    no_improve += 1
                
                if no_improve >= max_no_improve:
                    break

        # Phase 2: Final polish around the best point found
        if best_x is not None and not check_deadline():
            x = best_x.copy()
            fx = best_f
            sigma = 0.3
            
            remaining_calls = budget - total_calls
            polish_steps = min(polish_budget, remaining_calls)
            
            for _ in range(polish_steps):
                if check_deadline():
                    break
                
                y = np.clip(x + sigma * rng.standard_normal(d), -5, 5)
                fy = func(y)
                total_calls += 1
                
                if fy < fx:
                    x, fx = y, fy
                    if fx < best_f:
                        best_f, best_x = fx, x.copy()
                    sigma *= 1.1
                else:
                    sigma *= 0.9
                
                # Stop if sigma is very small
                if sigma < 1e-8:
                    break

        return best_x, best_f
# EVOLVE-BLOCK-END
