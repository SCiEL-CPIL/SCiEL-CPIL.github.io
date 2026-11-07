import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        start_time = time.time()
        deadline = start_time + 290  # Safety margin before 300s kill

        calls = 0
        best_x = None
        best_f = float('inf')
        
        def eval_func(x):
            nonlocal best_x, best_f, calls
            x = np.clip(x, -5, 5)
            f = func(x)
            calls += 1
            if f < best_f:
                best_f = f
                best_x = x.copy()
            return f

        rng = np.random.default_rng(42)

        # Phase 1: Initial Exploration
        # Sample a diverse set of points to identify promising basins.
        # Rastrigin has many local minima. We want to find the global one or a very good local one.
        n_init = 30
        candidates = []
        
        # Always try the origin, as shifts are often small
        eval_func(np.zeros(dim))
        
        for _ in range(n_init - 1):
            if calls >= budget or time.time() > deadline:
                break
            x = rng.uniform(-5, 5, dim)
            f = eval_func(x)
            candidates.append((f, x))
            
        # Sort candidates by function value
        candidates.sort(key=lambda t: t[0])
        
        # Phase 2: Multi-start Local Optimization with Adaptive Budget
        # For the top K candidates, run a (1+1)-ES to refine them.
        # We use a dynamic budget allocation: if a candidate is improving, we give it more time.
        # If it stagnates, we move to the next one.
        
        top_k = 5
        max_local_budget_per_candidate = 80  # Max calls per candidate
        min_local_budget_per_candidate = 20  # Min calls per candidate to avoid wasting time
        
        for i in range(top_k):
            if calls >= budget or time.time() > deadline:
                break
            if i >= len(candidates):
                break
            
            f_start, x_start = candidates[i]
            
            # Skip if this candidate is not significantly better than the current global best
            # This prevents wasting budget on points that are worse than what we already have
            if i > 0 and f_start > best_f + 1.0: 
                # If the starting point is much worse than the global best, 
                # it's unlikely to lead to a better solution quickly unless it's in a different basin.
                # But since we sorted, if the 2nd best is > 1.0 worse than global best, 
                # it might be a different basin. We still try it, but with less budget.
                pass

            x = x_start.copy()
            f_x = f_start
            
            # (1+1)-ES Parameters
            sigma = 0.5  # Initial step size. Rastrigin minima are spaced by ~1.0.
            window = 10
            successes = 0
            stagnation = 0
            max_stagnation = 20  # If no improvement for 20 steps, try to jump
            local_calls = 0
            
            # Dynamic budget: start with a small portion, increase if improving
            current_budget = min_local_budget_per_candidate
            
            while calls < budget and time.time() < deadline and local_calls < max_local_budget_per_candidate:
                # Check if we should stop this candidate early
                if local_calls >= current_budget and f_x > best_f:
                    break
                
                y = x + sigma * rng.standard_normal(dim)
                y = np.clip(y, -5, 5)
                f_y = eval_func(y)
                local_calls += 1
                
                if f_y < f_x:
                    x = y
                    f_x = f_y
                    successes += 1
                    stagnation = 0
                    # If improving, we can afford to spend more time here
                    if local_calls % 10 == 0:
                        current_budget = min(max_local_budget_per_candidate, current_budget + 10)
                else:
                    stagnation += 1
                    
                # Adapt sigma based on success rate
                if successes % window == 0:
                    sr = successes / window
                    if sr > 0.5:
                        sigma *= 1.2
                    elif sr < 0.3:
                        sigma *= 0.8
                    else:
                        sigma *= 1.0
                    sigma = np.clip(sigma, 0.01, 5.0)
                    successes = 0

                # Stagnation handling: Try to jump to escape local minimum
                if stagnation > max_stagnation:
                    # Reset sigma to a larger value to encourage larger jumps
                    sigma = max(sigma, 0.5)
                    stagnation = 0
                    # Optional: Perturb x slightly to break symmetry
                    # x = x + 0.1 * rng.standard_normal(dim)
                    # f_x = eval_func(x) # This costs a call, so we avoid it if possible
                    
        # Phase 3: Final Polish
        # If we have budget left, do a very fine search around best_x
        if calls < budget and time.time() < deadline:
            x = best_x.copy()
            f_x = best_f
            sigma = 0.05
            # Use remaining budget for this phase
            remaining = budget - calls
            max_polish_steps = min(remaining, 50)
            for _ in range(max_polish_steps):
                if calls >= budget or time.time() > deadline:
                    break
                y = x + sigma * rng.standard_normal(dim)
                y = np.clip(y, -5, 5)
                f_y = eval_func(y)
                if f_y < f_x:
                    x = y
                    f_x = f_y
                    sigma *= 0.9
                else:
                    sigma *= 0.95
                sigma = max(sigma, 0.001)

        return best_x, best_f
