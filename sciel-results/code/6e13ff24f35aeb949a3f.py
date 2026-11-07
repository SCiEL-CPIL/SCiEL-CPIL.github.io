"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """(1+1)-ES with adaptive sigma, multi-start, and local refinement."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        deadline = time.time() + 270.0  # safety margin

        best_x = np.zeros(dim)
        best_f = np.inf
        calls = 0

        def eval_func(x):
            nonlocal calls, best_x, best_f
            if calls >= budget or time.time() > deadline:
                return np.inf
            calls += 1
            try:
                val = float(func(x))
            except:
                return np.inf
            if np.isnan(val) or np.isinf(val):
                return np.inf
            if val < best_f:
                best_f = val
                best_x = x.copy()
            return val

        # Strategy: (1+1)-ES with restarts
        # 1. Random initialization to find a decent starting point.
        # 2. Main loop: Run (1+1)-ES for a certain number of steps.
        #    - If improvement is made, continue.
        #    - If no improvement for a while, restart from a new random point 
        #      or from the best point with a larger sigma.
        
        # Phase 1: Random Initialization (use ~10% of budget or fixed small number)
        init_budget = min(20, budget // 5)
        for _ in range(init_budget):
            if calls >= budget or time.time() > deadline:
                break
            x = np.random.uniform(-5, 5, dim)
            eval_func(x)

        # Phase 2: (1+1)-ES with restarts
        # Parameters
        sigma = 0.5
        no_improve_count = 0
        max_no_improve = 20 # Restart if no improvement for this many steps
        
        while calls < budget and time.time() < deadline:
            # Generate candidate
            x = best_x + sigma * np.random.randn(dim)
            # Clip to box to stay within valid domain (though func clips too)
            x = np.clip(x, -5, 5)
            
            f_x = eval_func(x)
            
            if f_x < best_f:
                # Improvement
                best_f = f_x
                best_x = x.copy()
                no_improve_count = 0
                # Slightly increase sigma to keep exploring
                sigma *= 1.1
                if sigma > 2.0: sigma = 2.0
            else:
                # No improvement
                no_improve_count += 1
                # Decrease sigma to refine
                sigma *= 0.9
                if sigma < 1e-4: sigma = 1e-4
                
                if no_improve_count >= max_no_improve:
                    # Restart
                    # Option A: New random point
                    # Option B: Perturb best point with large sigma
                    # Let's try a mix: 50% random, 50% large perturbation of best
                    if np.random.rand() < 0.5:
                        best_x = np.random.uniform(-5, 5, dim)
                        best_f = eval_func(best_x)
                    else:
                        # Large perturbation
                        sigma = 2.0
                        x_new = best_x + sigma * np.random.randn(dim)
                        x_new = np.clip(x_new, -5, 5)
                        f_new = eval_func(x_new)
                        if f_new < best_f:
                            best_f = f_new
                            best_x = x_new.copy()
                    
                    no_improve_count = 0
                    sigma = 0.5 # Reset sigma

        return best_x, best_f
# EVOLVE-BLOCK-END
