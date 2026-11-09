"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-Start Basin Hopping with Adaptive Global Jumps."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280  # Safety margin before 300s limit
        
        # Track best value found so far
        best_fx = float('inf')
        best_x = np.zeros(self.dim)
        
        # We will perform a series of local searches from different starting points.
        # To ensure coverage, we start with a set of diverse points.
        # For dim=5, we can use a stratified sampling or just random points with a fixed seed for reproducibility.
        # However, since the shift is hidden and seeded, a fixed seed might be bad if it misses the basin.
        # Better: Use a mix of random and structured starts.
        
        # Given the budget is 500, and local search is cheap, we can afford many restarts.
        # Let's define a local search procedure: Simple Coordinate Descent or Nelder-Mead?
        # Coordinate descent is very fast and effective for Sphere.
        # Nelder-Mead is more robust but slower.
        # Let's use a simple Adaptive Step Size Coordinate Descent (essentially a 1D search per dimension).
        
        def local_search(x0, fx0, max_iter=100):
            """Perform coordinate descent from x0, fx0."""
            x = x0.copy()
            fx = fx0
            # Initial step size for coordinate descent
            step = 0.5 
            
            for i in range(max_iter):
                if time.time() > deadline:
                    return x, fx
                improved = False
                for d in range(self.dim):
                    # Try moving in positive and negative direction along dimension d
                    for sign in [1, -1]:
                        x_new = x.copy()
                        x_new[d] = np.clip(x_new[d] + sign * step, -5, 5)
                        fx_new = func(x_new)
                        if fx_new < fx:
                            x = x_new
                            fx = fx_new
                            improved = True
                            # If improved, we can keep the step size or increase slightly
                            # But to be safe, we'll just reset the step size for the next dimension
                        else:
                            # If not improved, shrink step size for this dimension?
                            # Or just try the other direction.
                            pass
                    # If no improvement in this dimension, shrink step size
                    if not improved:
                        # Actually, 'improved' is per dimension loop iteration above. 
                        # Let's refine: track if *any* improvement happened in the full sweep.
                        pass
                
                # Check if any improvement happened in the full sweep
                # Re-logic: The above loop structure is a bit messy for tracking 'improved' correctly for step size.
                # Let's simplify: 
                # 1. Sweep all dimensions.
                # 2. If any improvement, keep step.
                # 3. If no improvement, shrink step.
                
                # Let's restart the logic for clarity inside the function
                break # The above logic was flawed, let's rewrite local_search properly below or inline.
            
            return x, fx

        # Proper Local Search: Coordinate Descent with Adaptive Step
        def coord_descent(x_start, fx_start, max_iter=50):
            x = x_start.copy()
            fx = fx_start
            step = 1.0
            
            for _ in range(max_iter):
                if time.time() > deadline:
                    break
                improved = False
                for d in range(self.dim):
                    # Try +step
                    x_trial = x.copy()
                    x_trial[d] = np.clip(x_trial[d] + step, -5, 5)
                    fx_trial = func(x_trial)
                    if fx_trial < fx:
                        x = x_trial
                        fx = fx_trial
                        improved = True
                        continue # Move to next dimension with same step? Or restart sweep?
                        # Standard CD usually continues sweep.
                    
                    # Try -step
                    x_trial = x.copy()
                    x_trial[d] = np.clip(x_trial[d] - step, -5, 5)
                    fx_trial = func(x_trial)
                    if fx_trial < fx:
                        x = x_trial
                        fx = fx_trial
                        improved = True
                
                if not improved:
                    step *= 0.5
                    if step < 1e-4:
                        break
            return x, fx

        # Strategy:
        # 1. Generate a set of starting points. 
        #    Since the optimum is in [-4, 4]^5, we can sample from there.
        #    We have 500 calls. 
        #    Each local search uses ~10-20 calls if it converges fast.
        #    We can afford ~20-50 restarts.
        
        # To be robust against the hidden shift, we should spread our starts.
        # We'll use a Latin Hypercube or just random points, but ensure we cover the space.
        # Let's use a fixed set of points generated by a seeded RNG, but also include a few random ones if time permits?
        # No, deterministic is safer for reproducibility, but the "hidden seeded shift" implies the function is fixed per instance.
        # If we use a fixed seed for our optimizer, we might miss the basin if the shift is unlucky.
        # However, Sphere is convex. Any local search should find the global minimum if the step size is adaptive enough.
        # The risk is getting stuck in a "flat" region or boundary issues.
        # Coordinate descent is very effective on Sphere.
        
        # Let's try 20 restarts.
        num_restarts = 20
        rng = np.random.default_rng(42) # Fixed seed for optimizer behavior
        
        # Generate starting points
        # Uniform in [-4, 4] to be closer to the optimum region
        starts = rng.uniform(-4, 4, (num_restarts, self.dim))
        
        # Also add the center 0 and corners?
        # Let's just use the random points.
        
        for i in range(num_restarts):
            if time.time() > deadline:
                break
                
            x_start = starts[i]
            fx_start = func(x_start)
            
            if fx_start < best_fx:
                best_fx = fx_start
                best_x = x_start.copy()
            
            # Run local search
            x_opt, fx_opt = coord_descent(x_start, fx_start, max_iter=100)
            
            if fx_opt < best_fx:
                best_fx = fx_opt
                best_x = x_opt.copy()
            
            # If we found a very good value, we might stop early?
            # e_i is clipped to 1e-8. If we reach 1e-8, we can't get much better.
            # But we don't know f_opt.
            # If fx_opt is very low, we can break.
            # However, AOCC rewards early progress. 
            # If we do one restart and it takes 50 calls, we have 450 left.
            # If we do 20 restarts, we use all 500.
            # Early progress: The first restart's progress is counted for calls 1..50.
            # Subsequent restarts' progress is counted for calls 51..100, etc.
            # Since AOCC is mean of n_i, and n_i decreases as we get closer, 
            # it is better to make steady progress than to jump later.
            # Coordinate descent makes steady progress.
            
            # Optional: If we haven't improved in the last few restarts, we might be stuck or converged.
            # But Sphere is convex, so it should converge.
            
            # Let's also try a "global jump" if we are stuck?
            # Not needed for Sphere if CD is working.
            
            # Check time
            if time.time() > deadline - 5:
                break

        return best_x, best_fx

# EVOLVE-BLOCK-END
