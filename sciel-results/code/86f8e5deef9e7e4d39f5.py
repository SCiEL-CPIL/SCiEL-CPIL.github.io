"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """Deterministic multi-scale coordinate-wise search for Lunacek bi-Rastrigin."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280  # Safety margin before 300s limit
        
        dim = self.dim
        budget = self.budget
        
        # Track best solution found so far
        best_x = None
        best_f = np.inf
        
        calls_made = 0
        
        # Helper to evaluate a point safely
        def try_eval(x):
            nonlocal best_x, best_f, calls_made
            if calls_made >= budget:
                return
            if time.time() > deadline:
                return
            x = np.clip(x, -5, 5)
            f_val = func(x)
            calls_made += 1
            if f_val < best_f:
                best_f = f_val
                best_x = x.copy()

        # 1. Initial coarse grid exploration
        # We want to cover the space [-5, 5]^5.
        # A full grid of 5^5 = 3125 points is too expensive (budget 500).
        # We use a stratified sampling or a sparse grid.
        # Let's try a few key points first: center, corners, mid-points.
        
        # Center
        try_eval(np.zeros(dim))
        
        # Simple random sampling to get a diverse start? 
        # The inspirations suggest deterministic multi-scale. 
        # Let's do a small random set first to find a good starting point for hill climbing.
        rng = np.random.default_rng(42)
        
        # Evaluate 20 random points to find a good basin
        for _ in range(20):
            if calls_made >= budget: break
            if time.time() > deadline: break
            x_rand = rng.uniform(-4.5, 4.5, dim)
            try_eval(x_rand)
            
        if best_x is None:
             best_x = np.zeros(dim)
             try_eval(best_x)
             
        # 2. Multi-scale coordinate-wise hill climbing
        # We will iterate through a sequence of step sizes, shrinking them.
        # For each step size, we try moving in each coordinate +/- step.
        # If we find an improvement, we keep it and restart the coordinate loop (or continue).
        # This is essentially a Nelder-Mead variant but axis-aligned.
        
        # Step sizes: Start large to jump between basins, shrink to refine.
        # Domain is [-5, 5]. 
        # Steps: 2.0, 1.0, 0.5, 0.25, 0.1, 0.05, 0.02, 0.01, 0.005, 0.002, 0.001
        steps = [2.0, 1.0, 0.5, 0.25, 0.1, 0.05, 0.02, 0.01, 0.005, 0.002, 0.001, 0.0005, 0.0002]
        
        current_x = best_x.copy()
        
        # We will perform multiple passes over the step sizes.
        # Since budget is 500, we need to be efficient.
        # Each pass over all coordinates with a given step takes 2*dim calls = 10 calls.
        # 13 steps * 10 calls = 130 calls per full pass.
        # We can afford ~3-4 full passes.
        
        for pass_idx in range(4):
            for step in steps:
                if calls_made >= budget: break
                if time.time() > deadline: break
                
                improved = True
                # Iterate through coordinates until no more improvement is found for this step size
                while improved:
                    improved = False
                    for i in range(dim):
                        if calls_made >= budget: break
                        if time.time() > deadline: break
                        
                        # Try positive step
                        x_try = current_x.copy()
                        x_try[i] += step
                        if np.all((x_try >= -5) & (x_try <= 5)):
                            f_try = func(x_try)
                            calls_made += 1
                            if f_try < best_f:
                                best_f = f_try
                                best_x = x_try.copy()
                                current_x = x_try.copy()
                                improved = True
                                # Continue to next coordinate or restart? 
                                # Standard hill climbing: continue.
                                # But since we changed current_x, the other coordinates' optimal shifts might change.
                                # For simplicity, we just continue to next coordinate.
                                
                        # Try negative step
                        x_try = current_x.copy()
                        x_try[i] -= step
                        if np.all((x_try >= -5) & (x_try <= 5)):
                            f_try = func(x_try)
                            calls_made += 1
                            if f_try < best_f:
                                best_f = f_try
                                best_x = x_try.copy()
                                current_x = x_try.copy()
                                improved = True
        
        # 3. Final fine-tuning if budget remains
        # Use a very small random jitter search around the best point found
        if calls_made < budget:
             base = best_x
             # 50 random perturbations with small magnitude
             for _ in range(50):
                 if calls_made >= budget: break
                 if time.time() > deadline: break
                 # Perturb each dimension slightly
                 noise = rng.normal(0, 0.01, dim)
                 x_try = base + noise
                 try_eval(x_try)

        # If no calls were made (shouldn't happen), return center
        if best_x is None:
            best_x = np.zeros(dim)
            # Ensure at least one call is made to be valid? 
            # "exception before the first call... invalid". 
            # If we didn't call func, is it invalid? 
            # "Every call is recorded... More than 500 calls... invalid". 
            # If 0 calls, AOCC is based on n_i. 
            # "repeating the last n_i if fewer calls are made". 
            # If 0 calls, what is n_1? Probably undefined or penalty. 
            # Better to make at least 1 call.
            if calls_made == 0:
                try_eval(best_x)

        return best_x, best_f
# EVOLVE-BLOCK-END
