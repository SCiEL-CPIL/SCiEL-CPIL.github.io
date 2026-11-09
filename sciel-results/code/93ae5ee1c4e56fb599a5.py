import numpy as np
import time

class Optimizer:
    """
    Optimizer for BBOB-style Lunacek bi-Rastrigin (f24).
    Uses (1+1)-ES with 1/5 success rule and multiple restarts.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290  # 10s safety margin before 300s limit

        n = self.dim
        total_budget = self.budget
        
        # Use a fixed seed for reproducibility within the instance, 
        # though the evaluator creates a new Optimizer instance for each run.
        rng = np.random.default_rng(42)

        best_x = None
        best_f = np.inf
        calls_made = 0

        # Restart strategy: Front-loaded budgets to maximize early progress (AOCC)
        # 5 restarts with geometrically decreasing budgets
        num_restarts = 5
        # Budgets: 30%, 25%, 20%, 15%, 10% (sums to 100%)
        # We allocate slightly more to the first one to ensure a solid initial descent
        budgets = [
            int(total_budget * 0.30),
            int(total_budget * 0.25),
            int(total_budget * 0.20),
            int(total_budget * 0.15),
            total_budget - (int(total_budget * 0.30) + int(total_budget * 0.25) + 
                           int(total_budget * 0.20) + int(total_budget * 0.15))
        ]
        
        # Ensure the last budget is positive
        if budgets[-1] < 1:
             budgets[-1] = 1
             budgets[0] -= 1

        for restart_idx in range(num_restarts):
            if calls_made >= total_budget:
                break
            if time.time() > deadline:
                break

            budget_this_restart = min(budgets[restart_idx], total_budget - calls_made)
            if budget_this_restart < 5:
                break

            # Initialization strategy based on restart index
            if restart_idx == 0:
                # Start near origin, often a good starting point for shifted functions
                x = np.zeros(n)
            elif restart_idx == 1:
                # Broad random search
                x = rng.uniform(-4, 4, n)
            elif restart_idx == 2:
                # Even broader random search
                x = rng.uniform(-5, 5, n)
            elif restart_idx == 3:
                # Perturb the best found so far to escape local minima
                if best_x is not None:
                    x = np.clip(best_x + rng.standard_normal(n) * 2.0, -5, 5)
                else:
                    x = rng.uniform(-5, 5, n)
            else:
                # Final restart: larger perturbation of best found
                if best_x is not None:
                    x = np.clip(best_x + rng.standard_normal(n) * 3.0, -5, 5)
                else:
                    x = rng.uniform(-5, 5, n)

            # Evaluate initial point
            fx = func(x)
            calls_made += 1
            
            if fx < best_f:
                best_f = fx
                best_x = x.copy()

            # (1+1)-ES Parameters
            sigma = 1.5  # Initial step size
            window = 10  # Window size for success rule
            successes = 0
            window_count = 0
            stale_count = 0
            max_stale = max(20, budget_this_restart // 3) # Cut restart if stuck for a while

            # Main (1+1)-ES loop
            while calls_made < total_budget:
                if time.time() > deadline:
                    break
                
                # Check if we've used up this restart's budget
                used_in_restart = calls_made - (total_budget - budget_this_restart)
                if used_in_restart >= budget_this_restart:
                    break

                # Generate candidate
                y = np.clip(x + sigma * rng.standard_normal(n), -5, 5)
                fy = func(y)
                calls_made += 1

                if fy < fx:
                    x, fx = y, fy
                    successes += 1
                    stale_count = 0
                    if fx < best_f:
                        best_f = fx
                        best_x = x.copy()
                else:
                    stale_count += 1

                window_count += 1

                # Adapt sigma using 1/5 success rule
                if window_count >= window:
                    rate = successes / window_count
                    if rate > 0.2:
                        sigma *= 1.3
                    elif rate < 0.15:
                        sigma *= 0.8
                    successes = 0
                    window_count = 0

                # Clamp sigma to reasonable bounds
                sigma = np.clip(sigma, 1e-8, 10.0)

                # Stagnation check: if no improvement for a long time, break restart
                if stale_count > max_stale:
                    break

        # Ensure we return valid values
        if best_x is None:
            best_x = np.zeros(n)
            best_f = func(best_x)
            calls_made += 1

        return best_x, best_f
