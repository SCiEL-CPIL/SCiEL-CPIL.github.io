import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0  # Safety margin for 300s limit
        dim = self.dim
        budget = self.budget

        best_x = None
        best_f = float('inf')
        calls_made = 0

        def clip(x):
            return np.clip(x, -5, 5)

        def remaining_time():
            return deadline - time.time()

        def evaluate(x):
            nonlocal calls_made, best_x, best_f
            x = clip(np.asarray(x, dtype=float))
            fx = float(func(x))
            calls_made += 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            return fx

        rng = np.random.default_rng(42)

        # Initial exploration
        x = rng.uniform(-5, 5, dim)
        fx = evaluate(x)

        # Adaptive (1+1)-ES with restarts
        # We will run multiple (1+1)-ES attempts.
        # Each attempt starts from a new point (random or perturbed best).
        # If no improvement for 'stagnation_limit' steps, we restart.
        
        stagnation_limit = 30
        max_attempts = 20  # Safety cap on number of restarts
        attempt_count = 0

        while calls_made < budget and attempt_count < max_attempts and remaining_time() > 5.0:
            attempt_count += 1
            
            # Choose starting point
            if attempt_count == 1:
                x = best_x.copy()
            else:
                # 50% chance to restart from best, 50% from random
                if rng.random() < 0.5:
                    x = clip(best_x + 2.0 * rng.standard_normal(dim))
                else:
                    x = clip(rng.uniform(-5, 5, dim))
            
            fx = evaluate(x)
            sigma = 1.0
            last_improvement_call = calls_made
            no_improve_count = 0

            # (1+1)-ES loop
            while calls_made < budget and remaining_time() > 2.0:
                y = clip(x + sigma * rng.standard_normal(dim))
                fy = evaluate(y)

                if fy < fx:
                    x, fx = y, fy
                    sigma *= 1.05
                    no_improve_count = 0
                    last_improvement_call = calls_made
                else:
                    sigma *= 0.95
                    no_improve_count += 1

                # Stagnation check
                if no_improve_count >= stagnation_limit:
                    break

            # Update global best
            if fx < best_f:
                best_f = fx
                best_x = x.copy()

        # Final polish: very small steps around the best point
        if calls_made < budget and remaining_time() > 2.0:
            x = best_x.copy()
            fx = best_f
            sigma = 0.05
            # Use remaining budget for fine-tuning
            while calls_made < budget and remaining_time() > 1.0:
                y = clip(x + sigma * rng.standard_normal(dim))
                fy = evaluate(y)
                if fy < fx:
                    x, fx = y, fy
                    sigma *= 1.02
                else:
                    sigma *= 0.9
                    if sigma < 0.001:
                        sigma = 0.001

            if fx < best_f:
                best_f = fx
                best_x = x.copy()

        return best_x, best_f
