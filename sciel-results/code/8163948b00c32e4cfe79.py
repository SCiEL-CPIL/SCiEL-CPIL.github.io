"""BBOB f14 (Different powers) optimizer: (1+1)-ES with restarts."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """(1+1)-ES with 1/5 success rule and random restarts."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        deadline = time.time() + 325.0  # Safety margin before 330s kill
        
        rng = np.random.default_rng(42)
        
        best_x_global = None
        best_f_global = np.inf
        
        # We use the entire budget.
        # Strategy: Multiple short runs of (1+1)-ES with random restarts.
        # If a run stagnates, restart from a random point or the current best.
        
        # Parameters for (1+1)-ES
        # Step size adaptation: 1/5 success rule
        # sigma *= exp(0.2 * (successes - 0.2 * total)) every 'adapt_interval' steps
        
        adapt_interval = 10
        # Initial step size: reasonable fraction of box size
        initial_sigma = 1.0
        
        # We will run until budget is exhausted or time runs out.
        # Since (1+1)-ES is very fast, we can do many steps.
        # Let's just run a single long (1+1)-ES with restarts on stagnation.
        
        x = rng.uniform(-5, 5, dim)
        f_x = func(x)
        
        if np.isfinite(f_x):
            best_f_global = f_x
            best_x_global = x.copy()
        else:
            # If initial eval fails (shouldn't happen), try center
            x = np.zeros(dim)
            f_x = func(x)
            best_f_global = f_x
            best_x_global = x.copy()
            
        sigma = initial_sigma
        successes = 0
        total = 0
        call_count = 1
        
        # Stagnation detection
        last_improvement_iter = 0
        stagnation_limit = 100 # If no improvement for 100 evals, restart
        
        while call_count < budget and time.time() < deadline:
            # Adapt step size
            if total > 0 and total % adapt_interval == 0:
                # 1/5 success rule
                # Target success rate 0.2
                # sigma = sigma * exp(0.2 * (successes/total - 0.2)) ?
                # Standard: sigma *= exp( (successes/total - 0.2) / adapt_interval * c )
                # Let's use a simple multiplicative update
                success_rate = successes / total
                # If success rate > 0.2, increase sigma (move bigger)
                # If success rate < 0.2, decrease sigma
                # Adjustment factor
                factor = 1.0 + 0.3 * (success_rate - 0.2)
                sigma *= factor
                sigma = np.clip(sigma, 1e-6, 10.0)
                successes = 0
                total = 0

            # Generate candidate
            # Use random direction
            direction = rng.standard_normal(dim)
            direction /= np.linalg.norm(direction)
            
            y = x + sigma * direction
            # Clip to box
            y = np.clip(y, -5, 5)
            
            f_y = func(y)
            call_count += 1
            
            if not np.isfinite(f_y):
                # Treat as very bad
                f_y = np.inf
                total += 1
                continue

            total += 1
            
            if f_y < f_x:
                # Improvement
                x = y
                f_x = f_y
                successes += 1
                
                if f_x < best_f_global:
                    best_f_global = f_x
                    best_x_global = x.copy()
                    last_improvement_iter = call_count
            else:
                # No improvement
                pass
            
            # Stagnation check
            if call_count - last_improvement_iter > stagnation_limit:
                # Restart
                # Option 1: Random restart
                # Option 2: Perturb best
                # Let's try a random restart in a region around the best known point
                # Or just a fresh random start to escape local minima
                # f14 is multimodal. Global random restarts are good.
                
                # Mix: 50% random, 50% perturb best
                if rng.random() < 0.5:
                    x = rng.uniform(-5, 5, dim)
                    f_x = func(x)
                    call_count += 1
                    if f_x < best_f_global:
                        best_f_global = f_x
                        best_x_global = x.copy()
                else:
                    # Perturb best
                    perturbation = rng.standard_normal(dim) * 0.5
                    x = best_x_global + perturbation
                    x = np.clip(x, -5, 5)
                    f_x = func(x)
                    call_count += 1
                    if f_x < best_f_global:
                        best_f_global = f_x
                        best_x_global = x.copy()
                
                # Reset stagnation counter and sigma
                last_improvement_iter = call_count
                sigma = initial_sigma
                successes = 0
                total = 0

        return best_x_global, best_f_global
# EVOLVE-BLOCK-END
