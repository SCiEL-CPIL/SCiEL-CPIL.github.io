"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """
    Hybrid Global-Local Optimizer for BBOB f14.
    Strategy:
    1. Coarse Grid Search: Systematically samples a grid over [-5, 5]^5 to locate the basin of attraction of the global minimum. This handles the hidden shift/rotation robustly.
    2. Local Refinement: Uses a fast (1+1)-ES with adaptive step size (1/5 rule) starting from the best grid point to converge to the minimum.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290  # Safety margin before 330s limit
        
        rng = np.random.default_rng(42)
        calls_used = 0
        dim = self.dim
        budget = self.budget
        
        def check_time():
            nonlocal calls_used
            return (time.time() >= deadline) or (calls_used >= budget)

        # Phase 1: Global Coarse Search
        # We allocate ~25% of the budget to global search.
        # For dim=5, we need enough points to ensure the global minimum (shifted by <= 4) is found.
        # A grid of 3^5 = 243 points covers [-5, 5] with step 5. 
        # However, the minimum is in [-4, 4]. 
        # Let's use a smarter allocation: 
        # If we use 3 points per dimension, we have 243 calls. 
        # This leaves 257 calls for local search.
        
        # To be safer against the rotation/shift, we can use a slightly finer grid in the center 
        # or just stick to a uniform grid that covers the likely region.
        # The problem states minimizer is in [-4, 4]^5.
        # Let's evaluate points in a grid that covers [-4.5, 4.5] with 3 steps per dim? 
        # No, 3^5 = 243 is a good number. 
        # Let's define grid points for each dimension: -4.5, 0, 4.5
        # This covers the range where the minimum is guaranteed to be ([-4, 4]).
        
        grid_points = []
        # Generate 3^5 points
        # Values: -4.5, 0.0, 4.5
        coords = [-4.5, 0.0, 4.5]
        
        # To improve resolution without blowing up the count, we can add the origin and maybe midpoints?
        # 3^5 = 243.
        # Let's add 0.0 explicitly if not there (it is).
        # Let's also add points at +/- 2.25? That would be 5^5 = 3125, too many.
        
        # Let's stick to 3^5 = 243 points for the global phase.
        # This guarantees we sample the center of each 5x5... wait, range is 9.
        # Step size is 4.5.
        # Points: -4.5, 0, 4.5.
        # Distance between neighbors is 4.5.
        # Since the minimum is in [-4, 4], the closest grid point is at most 2.25 away in each dimension?
        # Actually, if min is at 3.9, closest is 4.5 (dist 0.6) or 0 (dist 3.9). 
        # If min is at 2.0, closest is 0 (dist 2.0) or 4.5 (dist 2.5).
        # This might be too coarse for f14 which can be sharp.
        
        # Alternative: Use a random global search with a fixed number of points, but seeded for reproducibility?
        # The prompt implies deterministic behavior is good for scoring stability, but random is allowed.
        # However, "Different powers" f14 is often ill-conditioned. 
        # A better global strategy: 
        # 1. Evaluate origin.
        # 2. Evaluate vertices of a box around origin?
        
        # Let's try a different global strategy: 
        # Use a set of 50 random points + Origin + Vertices of [-3,3]^5.
        # But the shift can be up to 4. So [-3,3] might miss it.
        
        # Let's go back to the grid but make it adaptive or slightly finer in the center.
        # Actually, 243 calls is 48% of budget. That's a lot.
        # Let's reduce global search to ~100 calls.
        
        # New Global Strategy:
        # 1. Origin (1 call)
        # 2. 8 Vertices of [-4, 4]^5 (8 calls)
        # 3. 8 Midpoints of edges of [-4, 4]^5? No.
        # 4. Random points.
        
        # Let's use a "Space Filling" approach with a small number of points.
        # Halton sequence or just random.
        # Given the 300s limit and simple function, we can afford more calls.
        # Let's use 100 calls for global search.
        
        x_best = np.zeros(dim)
        f_best = func(x_best)
        calls_used += 1
        
        # Generate 99 additional global candidates
        # We will use a mix of structured and random points to ensure coverage.
        # Structured: Vertices of [-4,4]^5 (32 vertices, not 8. 2^5=32).
        # 32 vertices + 1 origin = 33 calls.
        
        # Let's evaluate all 32 vertices of the box [-4, 4]^5.
        # This ensures we check the corners of the likely region.
        
        for v in range(32):
            if check_time():
                break
            x_candidate = np.array([(v >> i) & 1 for i in range(dim)])
            x_candidate = x_candidate * 8.0 - 4.0 # Map 0..1 to -4..4
            f_candidate = func(x_candidate)
            calls_used += 1
            if f_candidate < f_best:
                f_best = f_candidate
                x_best = x_candidate.copy()
        
        # Add some random points to fill gaps
        # Budget remaining: 500 - 33 = 467.
        # Let's use 50 more random points for global exploration.
        for _ in range(50):
            if check_time():
                break
            x_candidate = rng.uniform(-4.5, 4.5, size=dim)
            f_candidate = func(x_candidate)
            calls_used += 1
            if f_candidate < f_best:
                f_best = f_candidate
                x_best = x_candidate.copy()
        
        # Phase 2: Local Refinement
        # Use (1+1)-ES starting from x_best.
        # Initial step size: 0.5 (since we are likely close now)
        # Use 1/5 rule for adaptation.
        
        x = x_best.copy()
        fx = f_best
        sigma = 0.5
        success_count = 0
        steps_since_adapt = 0
        adaptation_interval = 5 # Adapt every 5 steps
        
        while not check_time():
            # Generate candidate
            z = rng.standard_normal(dim)
            x_trial = x + sigma * z
            x_trial = np.clip(x_trial, -5, 5)
            
            f_trial = func(x_trial)
            calls_used += 1
            
            if f_trial < fx:
                # Success
                x = x_trial
                fx = f_trial
                success_count += 1
                if fx < f_best:
                    f_best = fx
                    x_best = x.copy()
            else:
                # Failure
                pass
                
            steps_since_adapt += 1
            
            # Adapt sigma every 'adaptation_interval' steps
            if steps_since_adapt >= adaptation_interval:
                if success_count > 0.5 * adaptation_interval:
                    sigma *= 1.2
                elif success_count < 0.5 * adaptation_interval:
                    sigma /= 1.2
                success_count = 0
                steps_since_adapt = 0
                
            # Stop if sigma is very small
            if sigma < 1e-6:
                break

        return x_best, f_best
# EVOLVE-BLOCK-END
