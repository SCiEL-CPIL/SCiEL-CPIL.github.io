"""Optimizer for BBOB f23 (Katsuura) using Multi-Start (1+1)-ES with LHS initialization."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """Multi-Start (1+1)-ES with LHS initialization for Katsuura f23."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0  # Safety margin
        rng = np.random.default_rng(42)
        dim = self.dim
        budget = self.budget
        bounds = np.array([-5.0] * dim)
        upper = np.array([5.0] * dim)
        
        calls_made = 0
        x_global_best = None
        fx_global_best = np.inf

        def update_best(x, fx):
            nonlocal x_global_best, fx_global_best
            if fx < fx_global_best:
                fx_global_best = fx
                x_global_best = x.copy()

        # Phase 1: Latin Hypercube Sampling for global coverage
        # Reduced to 30 points to allocate more budget to local search
        n_lhs = 30
        lhs_points = np.zeros((n_lhs, dim))
        for i in range(dim):
            perm = rng.permutation(n_lhs)
            lhs_points[:, i] = (perm + 0.5) / n_lhs * 10.0 - 5.0

        for i in range(n_lhs):
            if calls_made >= budget:
                break
            x = lhs_points[i]
            fx = func(x)
            calls_made += 1
            update_best(x, fx)
            
            if time.time() > deadline:
                return x_global_best, fx_global_best

        # Phase 2: Multi-Start Local Search
        # We divide the remaining budget into trials.
        # Each trial consists of a "Jump" from the current best (or random) 
        # followed by a local (1+1)-ES refinement.
        
        remaining_budget = budget - calls_made
        if remaining_budget <= 0:
            return x_global_best, fx_global_best
            
        # Reserve budget for final polishing phase
        reserve_polish = min(30, remaining_budget // 5)
        search_budget = remaining_budget - reserve_polish
        
        # Define trial structure
        # Longer trials allow deeper convergence on Katsuura's narrow valleys
        trial_length = 70
        n_trials = search_budget // trial_length
        
        # If we can't do a full trial, do one big remaining chunk
        if n_trials == 0:
            trial_length = search_budget
            n_trials = 1

        sigma_init = 1.5
        min_sigma = 0.001  # Finer convergence for Katsuura's sharp minima
        max_sigma = 4.0
        jump_scale = 2.5 # Moderate jump to balance exploration/exploitation

        for t in range(n_trials):
            if calls_made >= budget - reserve_polish:
                break
            
            # Determine starting point for this trial
            # Strategy: Jump from global best with some randomness
            # If this is the first trial after LHS, we can also try a pure random point 
            # to ensure we don't just refine the LHS best too much if it's bad.
            if t == 0:
                # First trial: Start from the best LHS point, but maybe jump a bit?
                # Or just refine it. Let's refine it first.
                x = x_global_best.copy()
            else:
                # Subsequent trials: Jump from global best to explore new basin
                # The jump should be large enough to escape the current local min
                # but not so large it goes out of bounds (handled by clip)
                x = x_global_best + jump_scale * rng.standard_normal(dim)
                x = np.clip(x, bounds, upper)
            
            # Local (1+1)-ES Phase
            fx = func(x)
            calls_made += 1
            update_best(x, fx)
            
            sigma = sigma_init
            
            # Run local search for up to trial_length steps
            for step in range(trial_length - 1): # -1 because we already evaluated x
                if calls_made >= budget - reserve_polish:
                    break
                if time.time() > deadline:
                    return x_global_best, fx_global_best
                
                # Generate candidate
                y = x + sigma * rng.standard_normal(dim)
                y = np.clip(y, bounds, upper)
                fy = func(y)
                calls_made += 1
                
                if fy < fx:
                    x, fx = y, fy
                    # Increase sigma on success to move faster
                    sigma = min(sigma * 1.25, max_sigma)
                    update_best(x, fx)
                else:
                    # Decrease sigma on failure to refine
                    sigma = max(sigma * 0.75, min_sigma)
                    
        # Phase 3: High-precision polish
        # Use remaining budget for fine-grained convergence
        remaining = budget - calls_made
        if remaining > 10 and x_global_best is not None:
            # Start with a very small step size for precision
            x = x_global_best.copy()
            fx = fx_global_best
            sigma_polish = 0.05
            min_sigma_polish = 0.0005
            
            # Coarse-to-fine polishing with adaptive step size
            while calls_made < budget - 2 and time.time() < deadline:
                # Generate candidate with current step size
                y = x + sigma_polish * rng.standard_normal(dim)
                y = np.clip(y, bounds, upper)
                fy = func(y)
                calls_made += 1
                
                if fy < fx:
                    x, fx = y, fy
                    update_best(x, fx)
                    # Keep or slightly increase step on success
                    sigma_polish = min(sigma_polish * 1.1, 0.2)
                else:
                    # Decrease step on failure for finer search
                    sigma_polish = max(sigma_polish * 0.7, min_sigma_polish)
                    
                    # If step is very small and we're not improving, try a few random restarts
                    if sigma_polish <= min_sigma_polish * 1.01:
                        # Try a small random perturbation to escape potential flat region
                        y2 = x + 0.01 * rng.standard_normal(dim)
                        y2 = np.clip(y2, bounds, upper)
                        fy2 = func(y2)
                        calls_made += 1
                        if fy2 < fx:
                            x, fx = y2, fy2
                            update_best(x, fx)
                            sigma_polish = 0.02  # Reset to moderate step
                        else:
                            # Break out of local stall
                            break

        return x_global_best, fx_global_best
# EVOLVE-BLOCK-END
