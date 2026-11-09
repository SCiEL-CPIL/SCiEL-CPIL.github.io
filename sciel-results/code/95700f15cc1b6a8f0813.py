"""
Multi-Start (1+1)-ES with 1/5 success-rule sigma adaptation and adaptive restarts.
This is a fundamentally different strategy from the current DE approach:
- Uses a single individual (1+1-ES) rather than a population
- Adapts step size via the 1/5 success rule
- Performs multiple restarts from the running global best
- Final polish phase with tight local search
"""
import numpy as np
import time


class Optimizer:
    """
    Multi-start (1+1)-ES for BBOB f18 (Schaffers F7 ill-conditioned).
    Strategy:
      - 1+1-ES with 1/5 success-rule sigma adaptation
      - Multiple restarts from global best when stagnation detected
      - Final polish with tight local search
      - Time guard with safety margin
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 310.0  # safety margin before 330s kill

        rng = np.random.default_rng(42)
        dim = self.dim
        budget = self.budget
        n_evals = 0

        # Global best tracking
        global_best_x = None
        global_best_f = np.inf

        def update_global(x, f):
            nonlocal global_best_x, global_best_f
            if f < global_best_f:
                global_best_f = f
                global_best_x = x.copy()

        # Phase 1: Initial exploration with several random starts
        # Do a small random sampling to get a good starting point
        n_initial = 20
        for i in range(n_initial):
            if n_evals >= budget:
                break
            x = rng.uniform(-5, 5, dim)
            fx = func(x)
            n_evals += 1
            update_global(x, fx)

        # Phase 2: Multi-start 1+1-ES
        # Parameters
        lambda_ = 1  # 1+1-ES
        c_sigma = 1.0 / (dim + 4)
        c_c = (4 + dim / dim) / (dim + 4 + 4)
        # For 1+1-ES, we use simpler adaptation
        # 1/5 success rule: if 1/5 of steps improve, keep sigma; else increase/decrease
        
        # We'll do multiple restarts
        max_restarts = 8
        restarts = 0
        stall_threshold = 30  # no improvement in this many evals triggers restart
        last_improve_eval = n_evals
        
        while n_evals < budget and restarts < max_restarts:
            if time.time() > deadline:
                break
            
            # Start a new 1+1-ES from current global best (or random if first)
            if global_best_x is not None:
                x = global_best_x.copy()
                f = global_best_f
            else:
                x = rng.uniform(-5, 5, dim)
                f = func(x)
                n_evals += 1
                update_global(x, f)
            
            # Initial sigma: start with a moderate step size
            sigma = 1.0
            # Success counter for 1/5 rule
            success_count = 0
            total_steps = 0
            no_improve = 0
            
            # Run 1+1-ES until stagnation or budget
            while n_evals < budget:
                if time.time() > deadline:
                    break
                
                # Generate candidate
                z = rng.standard_normal(dim)
                x_new = x + sigma * z
                x_new = np.clip(x_new, -5, 5)
                
                f_new = func(x_new)
                n_evals += 1
                
                total_steps += 1
                
                if f_new < f:
                    # Step succeeded
                    x = x_new
                    f = f_new
                    success_count += 1
                    no_improve = 0
                    update_global(x, f)
                    last_improve_eval = n_evals
                else:
                    no_improve += 1
                
                # Apply 1/5 success rule every dim steps
                if total_steps % dim == 0:
                    success_rate = success_count / total_steps
                    if success_rate > 0.2:
                        sigma *= 1.05
                    elif success_rate < 0.2:
                        sigma *= 0.95
                    success_count = 0
                    total_steps = 0
                
                # Clamp sigma
                sigma = np.clip(sigma, 1e-6, 10.0)
                
                # Check for stagnation: if no improvement for stall_threshold evals
                if n_evals - last_improve_eval >= stall_threshold:
                    break
            
            # If we broke due to stagnation, do a restart
            if n_evals < budget and n_evals - last_improve_eval >= stall_threshold:
                restarts += 1
                # Slightly perturb the global best for next restart
                if global_best_x is not None:
                    perturbation = 0.5 * rng.standard_normal(dim)
                    x_new = global_best_x + perturbation
                    x_new = np.clip(x_new, -5, 5)
                    f_new = func(x_new)
                    n_evals += 1
                    update_global(x_new, f_new)
                # Reset stall counter
                last_improve_eval = n_evals
        
        # Phase 3: Final polish - spend remaining calls on tight local search
        if n_evals < budget and global_best_x is not None:
            x = global_best_x.copy()
            f = global_best_f
            sigma = 0.1  # small step for polish
            
            while n_evals < budget:
                if time.time() > deadline:
                    break
                
                z = rng.standard_normal(dim)
                x_new = x + sigma * z
                x_new = np.clip(x_new, -5, 5)
                
                f_new = func(x_new)
                n_evals += 1
                
                if f_new < f:
                    x = x_new
                    f = f_new
                    update_global(x, f)
                    sigma *= 1.1  # slightly increase if improving
                else:
                    sigma *= 0.9  # decrease if not improving
                
                sigma = np.clip(sigma, 1e-8, 1.0)
        
        return global_best_x, global_best_f
