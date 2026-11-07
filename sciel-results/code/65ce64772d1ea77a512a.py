import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        timeout = 290.0  # Safety margin against 300s limit
        
        def time_check():
            return (time.time() - start_time) > timeout

        dim = self.dim
        budget = self.budget
        
        best_x = None
        best_f = np.inf
        eval_count = 0
        
        # Strategy: Deterministic Hierarchical Grid Search
        # The BBOB f7 (Step Ellipsoidal) has a flat plateau. 
        # Standard stochastic methods (like ES) can get stuck on the plateau 
        # because gradients are zero. A grid search systematically crosses 
        # the step boundary, allowing us to "jump" into the valley of the 
        # ellipsoidal part quickly.
        
        # Domain is [-5, 5]^5, but minimizer is in [-4, 4]^5.
        # We will focus search on [-4, 4]^5 to maximize resolution.
        
        # Phase 1: Coarse Grid Scan
        # Goal: Locate the general region of the step boundary and the minimum.
        # We use a stratified sampling approach (Halton-like or simple grid).
        # For dim=5, a full grid of 5^5 = 3125 points is too many for "coarse".
        # Let's use a smaller grid or a space-filling curve.
        
        # Actually, a simple grid with 3 points per dimension gives 3^5 = 243 points.
        # This covers the domain [-4, 4] with step size 4.
        # Points: -4, 0, 4.
        
        # Let's refine the grid slightly. 
        # If we use 4 points per dim: 4^5 = 1024 points. 
        # Budget is 500. So we can't do a full 4^5 grid.
        
        # Let's do a two-stage grid:
        # Stage 1: Very coarse grid to find the "quadrant" containing the min.
        # Divide [-4, 4] into 2x2x2x2x2 = 32 boxes.
        # Evaluate the center of each box? That's 32 evals.
        # Then pick the best box and refine?
        
        # Better: Use a space-filling sequence (Sobol or Halton) to ensure coverage.
        # But deterministic grid is requested.
        
        # Let's use a grid of 3 points per dimension: 3^5 = 243 evaluations.
        # This leaves ~257 evaluations for refinement.
        
        # Grid points: -4, 0, 4 is too sparse.
        # Let's use 4 points per dimension but only evaluate a subset?
        # Or use a "Latin Hypercube" style deterministic sample?
        
        # Let's stick to a structured grid but adaptively.
        
        # 1. Coarse Grid: 3 points per dim in [-4, 4] -> 243 points.
        #    Points: -4, 0, 4.
        #    This might miss the minimum if it's between -4 and 0.
        
        #    Let's use 4 points per dim: -4, -1.33, 1.33, 4.
        #    4^5 = 1024 points. Too many for full scan.
        
        #    Alternative: Use a grid of 2 points per dim initially?
        #    2^5 = 32 points. Centers of 32 sub-boxes.
        #    Find the best sub-box.
        #    Then refine that sub-box with a 3^5 grid?
        
        #    Let's try:
        #    Phase 1: 2^5 grid (32 evals). Find best region.
        #    Phase 2: Refine the best region with a finer grid.
        
        #    However, the "step" might be anywhere. 
        #    A global coarse grid is safer.
        
        #    Let's use a Halton sequence (deterministic, low discrepancy) for the first 200 points.
        #    Halton is deterministic and covers the space well.
        
        def halton_sequence(index, base):
            f = 1.0
            r = 0.0
            i = index
            while i > 0:
                f /= base
                r += f * (i % base)
                i = i // base
            return r
        
        # Generate Halton points in [0, 1]^dim
        n_halton = 150
        halton_points = []
        bases = [2, 3, 5, 7, 11]
        
        for i in range(1, n_halton + 1):
            point = []
            for j in range(dim):
                point.append(halton_sequence(i, bases[j]))
            halton_points.append(point)
        
        halton_points = np.array(halton_points)
        
        # Map [0, 1]^dim to [-4, 4]^dim
        # x = -4 + 8 * h
        for h in halton_points:
            if eval_count >= budget or time_check():
                break
            x = -4.0 + 8.0 * h
            f = func(x)
            eval_count += 1
            if f < best_f:
                best_f = f
                best_x = x.copy()
        
        # Phase 2: Local Refinement using Coordinate Descent / Grid around best point
        # The best point from Halton is likely near the step or in the valley.
        # We perform a local grid search around best_x.
        
        # Determine a local box size.
        # If we are on the plateau, we need to find the edge.
        # If we are in the valley, we need to converge.
        
        # Let's use a multi-resolution local grid.
        # Start with a box of size 2.0 around best_x (clipped to [-4, 4]).
        # Do a 3^5 grid in this local box? 243 points.
        # Total so far: 150 + 243 = 393. Remaining: ~100.
        
        # Let's be more aggressive.
        # If we have ~350 points left, we can do a 3^5 grid (243) and still have ~100 for final polish.
        
        if best_x is None:
             # Fallback
             best_x = np.zeros(dim)
             if eval_count < budget:
                 best_f = func(best_x)
                 eval_count += 1
        
        local_box_size = 2.0
        # Create local grid points
        # 3 points per dim: center - box_size, center, center + box_size
        # This covers a range of 2 * box_size.
        
        grid_points_local = []
        # Use 3 levels: -1, 0, 1 scaled by box_size
        levels = np.array([-1.0, 0.0, 1.0]) * (local_box_size / 2.0)
        
        # Generate all combinations
        # For dim=5, 3^5 = 243 combinations
        # We can generate them recursively or with itertools.product
        import itertools
        
        for combo in itertools.product(levels, repeat=dim):
            if eval_count >= budget or time_check():
                break
            x = best_x + np.array(combo)
            # Clip to valid domain [-5, 5] or [-4, 4]?
            # Task says func clips to [-5, 5]. Minimizer in [-4, 4].
            # We should stay within [-4, 4] to be safe, or let func clip.
            # Let's clip to [-4.5, 4.5] to stay inside.
            x = np.clip(x, -4.5, 4.5)
            
            f = func(x)
            eval_count += 1
            if f < best_f:
                best_f = f
                best_x = x.copy()
                
        # Phase 3: Final Polish with small step (1+1)-ES or tiny grid
        # Use remaining budget for fine-tuning.
        # Let's use a simple (1+1)-ES with very small sigma.
        
        sigma = 0.1
        x = best_x.copy()
        f = best_f
        rng = np.random.default_rng(42)
        
        while eval_count < budget and not time_check():
            z = rng.standard_normal(dim)
            x_new = x + sigma * z
            x_new = np.clip(x_new, -4.5, 4.5)
            f_new = func(x_new)
            eval_count += 1
            
            if f_new < f:
                x = x_new
                f = f_new
                if f < best_f:
                    best_f = f
                    best_x = x.copy()
                # Increase sigma slightly if we are moving?
                # For step functions, large moves are risky. Keep sigma constant or decrease.
                # Let's keep it constant to explore the local neighborhood.
            else:
                # If we fail, decrease sigma to converge
                sigma *= 0.9
                if sigma < 1e-5:
                    break
                    
        return best_x, best_f
