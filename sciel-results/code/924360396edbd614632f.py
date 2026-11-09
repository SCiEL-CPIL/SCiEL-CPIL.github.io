"""Deterministic Multi-Resolution Local Search (Grid Descent) for BBOB f7 (Step ellipsoidal)."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """
    Deterministic Multi-Resolution Local Search for BBOB f7.
    
    Strategy:
    1. Multi-start: Initialize several random points.
    2. For each point, perform a local search using a geometrically shrinking step size.
    3. The local search explores a grid around the current best (coordinate-wise or full grid).
    4. This is effective for 'step' functions because it systematically sweeps the plateau 
       to find the edge where the function value drops.
    5. If a local search stalls (no improvement over several step sizes), restart from a new random point.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 295.0  # 5s safety margin
        
        d = self.dim
        budget = self.budget
        
        # Global best tracking
        global_best_f = np.inf
        global_best_x = np.zeros(d)
        
        # Number of initial random starts
        num_initial_starts = 10
        calls_used = 0
        
        # List of starting points
        start_points = []
        
        # Generate initial random points
        # Spread them out a bit more than uniform to cover the space
        for i in range(num_initial_starts):
            if calls_used >= budget:
                break
            x0 = np.random.uniform(-4.5, 4.5, d)
            f0 = func(x0)
            calls_used += 1
            start_points.append((x0, f0))
            if f0 < global_best_f:
                global_best_f = f0
                global_best_x = x0.copy()
        
        # Sort start points by fitness (best first)
        start_points.sort(key=lambda x: x[1])
        
        # Define step sizes for local search
        # Start with a moderately large step and shrink geometrically
        # For d=5, a step of 1.0 is reasonable to start with.
        initial_sigma = 1.5
        num_step_scales = 20  # How many times to halve the step size
        min_sigma = 1e-6
        
        # Local search routine
        def local_search(x_start, f_start, max_calls):
            nonlocal global_best_f, global_best_x, calls_used
            
            x = x_start.copy()
            f = f_start
            sigma = initial_sigma
            
            # Track if we made progress in the last few scales
            progress_made = 0
            
            while sigma > min_sigma and calls_used < budget and calls_used < max_calls:
                if time.time() > deadline:
                    break
                
                improved = False
                
                # Explore a grid around x with step size sigma
                # For d=5, a full grid of 3^5 = 243 points is affordable.
                # However, to save calls, we can do coordinate-wise first, or just full grid.
                # Let's do a full grid but limit the number of candidates if d is large.
                # For d=5, 243 is fine.
                
                # Generate grid points: x + {-sigma, 0, sigma} in each dimension
                # We skip the center (0,0,...,0) as we already evaluated x.
                
                # To be efficient, we can iterate through combinations
                # But generating 243 points and evaluating them is 243 calls.
                # With 500 budget, we can only do ~2 full grid searches.
                # This is too expensive for a "multi-start" with 10 starts.
                
                # Revised Local Search: Coordinate-wise descent with step shrinking.
                # This is much cheaper: 2*d evaluations per coordinate pass.
                # Total per scale: 2*d evaluations.
                # For d=5, 10 evaluations per scale.
                # 20 scales -> 200 evaluations per start.
                # This fits better.
                
                # Coordinate-wise search
                for i in range(d):
                    if calls_used >= budget or calls_used >= max_calls:
                        break
                    if time.time() > deadline:
                        break
                    
                    # Try x + sigma * e_i and x - sigma * e_i
                    x_plus = x.copy()
                    x_plus[i] += sigma
                    x_minus = x.copy()
                    x_minus[i] -= sigma
                    
                    # Clip to box
                    x_plus = np.clip(x_plus, -5.0, 5.0)
                    x_minus = np.clip(x_minus, -5.0, 5.0)
                    
                    # Evaluate
                    f_plus = func(x_plus)
                    calls_used += 1
                    f_minus = func(x_minus)
                    calls_used += 1
                    
                    best_f_step = min(f, f_plus, f_minus)
                    if f_plus <= f_minus:
                        best_x_step = x_plus
                    else:
                        best_x_step = x_minus
                    
                    if best_f_step < f:
                        x = best_x_step
                        f = best_f_step
                        improved = True
                        if f < global_best_f:
                            global_best_f = f
                            global_best_x = x.copy()
                
                if improved:
                    progress_made += 1
                else:
                    # No improvement in this coordinate pass, shrink step size
                    sigma *= 0.7  # Shrink by 30%
                    
                    # If we have not improved for 5 consecutive scales, stop local search
                    if not improved and progress_made == 0:
                         # Actually, we should stop if we haven't improved in a while.
                         # Let's keep a counter of 'no_improve_scales'
                         pass
                
                # More robust stagnation check:
                # If we didn't improve in the last 'k' scales, break.
                # Let's just rely on the sigma shrinking and the max_calls limit.
                # But to be efficient, if we don't improve, we should stop sooner.
                
                # Let's track 'stall_count'
                if not improved:
                    # If we haven't improved in 3 consecutive scales, break
                    # We need a variable to track this.
                    pass

            return x, f

        # The above local_search is a bit messy with the stagnation logic.
        # Let's rewrite it cleanly.
        
        def local_search_clean(x_start, f_start):
            nonlocal global_best_f, global_best_x, calls_used
            
            x = x_start.copy()
            f = f_start
            sigma = initial_sigma
            stall_count = 0
            max_stalls = 5  # Stop if no improvement for 5 scales
            
            while sigma > min_sigma and calls_used < budget:
                if time.time() > deadline:
                    break
                
                improved = False
                
                # Coordinate-wise search
                for i in range(d):
                    if calls_used >= budget:
                        break
                    if time.time() > deadline:
                        break
                    
                    x_plus = x.copy()
                    x_plus[i] += sigma
                    x_minus = x.copy()
                    x_minus[i] -= sigma
                    
                    x_plus = np.clip(x_plus, -5.0, 5.0)
                    x_minus = np.clip(x_minus, -5.0, 5.0)
                    
                    f_plus = func(x_plus)
                    calls_used += 1
                    f_minus = func(x_minus)
                    calls_used += 1
                    
                    # Determine best move
                    if f_plus < f and f_plus <= f_minus:
                        x = x_plus
                        f = f_plus
                        improved = True
                        if f < global_best_f:
                            global_best_f = f
                            global_best_x = x.copy()
                    elif f_minus < f and f_minus < f_plus:
                        x = x_minus
                        f = f_minus
                        improved = True
                        if f < global_best_f:
                            global_best_f = f
                            global_best_x = x.copy()
                
                if improved:
                    stall_count = 0
                    # Optional: expand sigma slightly if we are making fast progress?
                    # No, for step functions, we want to shrink to find the edge.
                else:
                    stall_count += 1
                    sigma *= 0.7  # Shrink step size
                    if stall_count > max_stalls:
                        break
                        
            return x, f

        # Process start points
        for x0, f0 in start_points:
            if calls_used >= budget:
                break
            if time.time() > deadline:
                break
            
            # Run local search from this point
            # We don't know how many calls it will take, but it will stop on stall or budget.
            # To ensure we try multiple starts, we could limit calls per local search.
            # But with 500 budget and 10 starts, 50 calls per start is reasonable.
            # However, if the first start finds the global min, we waste budget.
            # The local search will stall quickly if it's in a bad spot.
            
            local_search_clean(x0, f0)
            
            # If we still have budget and time, and we haven't found a very good solution,
            # we might want to try more random starts.
            # But the initial 10 should be enough.
            
        # If we still have budget, do a final polish around the global best
        if calls_used < budget and time.time() < deadline:
            # Refine around global_best_x with very small steps
            x = global_best_x.copy()
            f = global_best_f
            sigma = 0.1
            stall_count = 0
            max_stalls = 10
            
            while sigma > 1e-7 and calls_used < budget:
                if time.time() > deadline:
                    break
                
                improved = False
                for i in range(d):
                    if calls_used >= budget:
                        break
                    if time.time() > deadline:
                        break
                    
                    x_plus = x.copy()
                    x_plus[i] += sigma
                    x_minus = x.copy()
                    x_minus[i] -= sigma
                    
                    x_plus = np.clip(x_plus, -5.0, 5.0)
                    x_minus = np.clip(x_minus, -5.0, 5.0)
                    
                    f_plus = func(x_plus)
                    calls_used += 1
                    f_minus = func(x_minus)
                    calls_used += 1
                    
                    if f_plus < f and f_plus <= f_minus:
                        x = x_plus
                        f = f_plus
                        improved = True
                        if f < global_best_f:
                            global_best_f = f
                            global_best_x = x.copy()
                    elif f_minus < f and f_minus < f_plus:
                        x = x_minus
                        f = f_minus
                        improved = True
                        if f < global_best_f:
                            global_best_f = f
                            global_best_x = x.copy()
                
                if improved:
                    stall_count = 0
                else:
                    stall_count += 1
                    sigma *= 0.7
                    if stall_count > max_stalls:
                        break
                        
        return global_best_x, global_best_f
# EVOLVE-BLOCK-END
