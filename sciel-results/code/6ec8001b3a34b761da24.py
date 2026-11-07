"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-restart (1+1)-ES with windowed 1/5 success rule and batch offspring for Rastrigin."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0  # safety margin before 300s kill
        dim = self.dim
        budget = self.budget
        rng = np.random.default_rng(42)

        global_best_x = None
        global_best_f = np.inf
        calls_made = 0

        def check_deadline():
            return time.time() > deadline

        def run_es(x_start, fx_start, max_calls, sigma_init=2.0, window=15, pop_size=3):
            """Run (1+1)-ES with windowed 1/5 success rule and small batch for up to max_calls evaluations."""
            nonlocal calls_made, global_best_x, global_best_f

            x = x_start.copy()
            fx = fx_start
            sigma = sigma_init
            min_sigma = 1e-8
            max_sigma = 10.0
            successes = 0
            local_best_x = x.copy()
            local_best_f = fx
            no_improve = 0
            max_no_improve = 20

            while calls_made < budget and not check_deadline():
                # Determine how many offspring we can evaluate in this batch
                remaining = budget - calls_made
                if remaining <= 0:
                    break
                
                # Use a small batch to reduce variance and improve exploration
                batch_size = min(pop_size, remaining)
                
                # Generate offspring
                offspring = x + sigma * rng.standard_normal((batch_size, dim))
                offspring = np.clip(offspring, -5, 5)
                
                # Evaluate offspring and track best
                min_f_offspring = np.inf
                best_offspring = None
                batch_successes = 0
                
                for y in offspring:
                    if calls_made >= budget or check_deadline():
                        break
                    fy = func(y)
                    calls_made += 1
                    
                    if fy < local_best_f:
                        local_best_f = fy
                        local_best_x = y.copy()
                    
                    if fy < best_f:
                        best_f = fy
                        best_x = y.copy()
                    
                    if fy < fx:
                        batch_successes += 1
                        if fy < min_f_offspring:
                            min_f_offspring = fy
                            best_offspring = y.copy()
                
                if min_f_offspring < fx:
                    x = best_offspring.copy()
                    fx = min_f_offspring
                    sigma *= 1.2
                    no_improve = 0
                else:
                    sigma *= 0.85
                    no_improve += 1
                    if no_improve >= max_no_improve:
                        break
                
                # Windowed 1/5 success rule: adapt sigma every 'window' steps
                successes += batch_successes
                if calls_made % window == 0:
                    success_rate = successes / window
                    if success_rate > 0.2:
                        sigma *= 1.3
                    elif success_rate < 0.2:
                        sigma *= 0.7
                    sigma = np.clip(sigma, min_sigma, max_sigma)
                    successes = 0

            # Update global best
            if local_best_f < global_best_f:
                global_best_f = local_best_f
                global_best_x = local_best_x.copy()

            return local_best_x, local_best_f

        # Phase 1: Multi-start exploration with diverse initial points
        # Start at origin (likely near optimum due to BBOB shift constraints)
        x0 = np.zeros(dim)
        fx0 = func(x0)
        calls_made += 1

        if fx0 < global_best_f:
            global_best_f = fx0
            global_best_x = x0.copy()

        # Allocate budget: ~40% to initial exploration with multiple restarts,
        # ~60% to refinement of the best point found

        # Phase 1: Exploration with multiple restarts
        exploration_budget = int(budget * 0.4)
        restarts = 5
        per_restart_budget = max(1, (exploration_budget - 1) // restarts)

        for r in range(restarts):
            if calls_made >= budget or check_deadline():
                break
            
            # Generate diverse starting points
            if r == 0:
                x_start = np.zeros(dim)
                fx_start = global_best_f
            else:
                x_start = rng.uniform(-4, 4, dim)
                fx_start = func(x_start)
                calls_made += 1
                if fx_start < global_best_f:
                    global_best_f = fx_start
                    global_best_x = x_start.copy()
            
            run_es(x_start, fx_start, per_restart_budget, sigma_init=2.0)

        # Phase 2: Refinement of the best point found
        if global_best_x is not None and not check_deadline() and calls_made < budget:
            remaining = budget - calls_made
            run_es(global_best_x, global_best_f, remaining, sigma_init=0.5, window=10, pop_size=2)

        return global_best_x, global_best_f
# EVOLVE-BLOCK-END
