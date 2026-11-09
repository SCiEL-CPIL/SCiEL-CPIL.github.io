"""Multi-restart Adaptive (1+1)-ES for BBOB f15 (Rastrigin rotated)."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-restart (1+1)-ES with adaptive sigma and diverse initializations."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 325.0  # safety margin before 330s kill

        rng = np.random.default_rng(42)
        dim = self.dim
        budget = self.budget

        def safe_eval(x):
            nonlocal calls_made
            if calls_made >= budget:
                return np.inf
            calls_made += 1
            try:
                return float(func(x))
            except Exception:
                return np.inf

        calls_made = 0

        # Multi-restart strategy:
        # Divide budget into multiple restarts with diverse starting points.
        # For Rastrigin (many local minima), diversity in initialization is key.
        
        # Strategy: 
        # - First restart: larger budget, start at origin (common good starting point)
        # - Subsequent restarts: smaller budgets, random diverse starts
        # - Use adaptive sigma (1/5 success rule) within each restart
        # - Track global best across all restarts
        
        # Determine number of restarts based on budget
        # With budget=500, dim=5:
        # - Use ~8-12 restarts for good coverage
        # - First restart gets ~40% of budget, rest split evenly
        
        n_restarts = 10
        # Budget allocation: first restart gets more, others get less
        first_budget = int(budget * 0.4)
        remaining_budget = budget - first_budget
        per_restart = remaining_budget // (n_restarts - 1) if n_restarts > 1 else 0

        # Define diverse initial points
        # Mix of: origin, random uniform, and structured grid-like points
        init_points = []
        init_points.append(np.zeros(dim))  # origin
        
        # Add random points
        for _ in range(n_restarts - 1):
            init_points.append(rng.uniform(-5, 5, dim))
        
        # Shuffle to mix origin with randoms (but keep origin as first for guaranteed early progress)
        # Actually, keep origin first for early AOCC benefit, then randomize the rest
        random_indices = list(range(1, len(init_points)))
        rng.shuffle(random_indices)
        init_points = [init_points[0]] + [init_points[i] for i in random_indices]

        global_best_x = init_points[0].copy()
        global_best_val = np.inf

        # Run each restart
        for r in range(n_restarts):
            if calls_made >= budget:
                break
            if time.time() > deadline - 2.0:
                break

            # Determine budget for this restart
            if r == 0:
                restart_budget = min(first_budget, budget - calls_made)
            else:
                restart_budget = min(per_restart, budget - calls_made)

            if restart_budget < 5:
                break

            x = init_points[r].copy()
            fx = safe_eval(x)
            calls_made += 0  # safe_eval already incremented

            if fx < global_best_val:
                global_best_val = fx
                global_best_x = x.copy()

            # Adaptive (1+1)-ES with 1/5 success rule
            sigma = 1.5
            success_count = 0
            steps_in_restart = 0
            max_steps = restart_budget - 1  # minus the initial eval

            for step in range(max_steps):
                if calls_made >= budget:
                    break
                if time.time() > deadline - 1.0:
                    break

                # Generate trial point
                y = np.clip(x + sigma * rng.standard_normal(dim), -5, 5)
                fy = safe_eval(y)

                steps_in_restart += 1

                if fy < fx:
                    # Success: move to y, increase sigma
                    x, fx = y, fy
                    sigma *= 1.3
                    success_count += 1
                    if fx < global_best_val:
                        global_best_val = fx
                        global_best_x = x.copy()
                else:
                    # Failure: decrease sigma
                    sigma *= 0.85

                # 1/5 success rule: adjust sigma based on success rate
                # Check every 5 steps
                if (step + 1) % 5 == 0:
                    # If success rate is above 1/5, increase sigma; below, decrease
                    # Simple heuristic: if we've had more than 1 success in last 5, sigma might be too small
                    # But standard 1/5 rule adjusts based on cumulative success rate
                    # For simplicity, use a moving window or just rely on the multiplicative updates
                    # The multiplicative updates above already provide adaptation.
                    # Let's add a soft reset if sigma gets too small or too large
                    if sigma < 0.01:
                        sigma = 0.01
                    elif sigma > 10.0:
                        sigma = 10.0

            # End of restart

        return global_best_x, global_best_val
# EVOLVE-BLOCK-END
