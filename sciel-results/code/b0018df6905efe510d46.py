"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """
    Structural change: Replace the single-trajectory (1+1)-ES with a multi-start
    strategy that combines broad LHS-based exploration with targeted local refinement.
    
    Mechanism:
    1. Broad Exploration: Uses Latin Hypercube Sampling (LHS) to spread initial points
       evenly across the domain. This is superior to purely random sampling for ensuring
       coverage of the 101 peaks in f21 within a limited budget.
    2. Targeted Refinement: For the most promising starting points, it runs a short,
       adaptive (1+1)-ES. The step size (sigma) is adapted using a 1/5 success rule
       variant to quickly converge to local minima.
    3. Deadline Guard: Explicitly tracks wall-clock time to ensure the program returns
       valid results before the 330s evaluator kill-switch, prioritizing safety over
       exhaustive search.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 320.0  # Safety margin before 330s kill
        rng = np.random.default_rng(42)
        dim = self.dim
        budget = self.budget

        def time_left():
            return deadline - time.time()

        # Helper: Generate LHS points
        def lhs_points(n, d, rng):
            pts = np.zeros((n, d))
            for dd in range(d):
                perm = rng.permutation(n)
                for i in range(n):
                    pts[perm[i], dd] = -5.0 + 10.0 * (i + 0.5) / n
            return pts

        best_x = None
        best_f = float('inf')
        call_count = 0

        # --- Phase 1: Broad Exploration with LHS ---
        # We allocate a portion of the budget to evaluating diverse points.
        # 20 points is a good balance for dim=5 to cover the space without
        # spending too much budget before refinement.
        n_lhs = 20
        lhs_pts = lhs_points(n_lhs, dim, rng)

        for pt in lhs_pts:
            if call_count >= budget or time_left() < 2.0:
                break
            f_val = func(pt)
            call_count += 1
            if f_val < best_f:
                best_f = f_val
                best_x = pt.copy()

        if best_x is None:
            # Fallback if budget was too small for even one call (shouldn't happen with 500)
            best_x = np.zeros(dim)
            best_f = func(best_x)
            call_count += 1

        # --- Phase 2: Targeted Refinement ---
        # We will perform local searches from the best point found so far,
        # and potentially from a few other good points if budget allows.
        # Strategy: 
        # 1. Refine the global best.
        # 2. If budget remains, refine the 2nd and 3rd best points from LHS.
        
        # Sort LHS points by function value to identify top candidates
        # Note: We already evaluated them. We can re-evaluate or store values.
        # For simplicity and memory efficiency, we'll just re-evaluate the top few
        # if we didn't store them, but storing is better. Let's store values from Phase 1.
        
        # Re-do Phase 1 logic slightly to store values for sorting?
        # Actually, we can just run refinement from `best_x` first.
        # If we have a lot of budget left, we can try other points.
        # Let's keep it simple: Refine the best point found in Phase 1.
        # Then, if budget allows, do a second round of LHS with smaller step size?
        # No, better to do multiple short restarts from diverse points.
        
        # Let's define a function to run a short (1+1)-ES
        def run_local_search(start_x, max_calls, current_best_f, current_best_x):
            nonlocal call_count, best_f, best_x
            
            if max_calls <= 0:
                return current_best_f, current_best_x

            x = start_x.copy()
            # Evaluate start point if not already the global best? 
            # We assume start_x is already evaluated in the caller context 
            # or we evaluate it here. 
            # To be safe and consistent with "calls are recorded", we must count every func call.
            # If start_x is the global best from Phase 1, we already have its f.
            # If it's a new point, we need to evaluate.
            # Let's assume the caller passes in a point that might not be evaluated.
            # However, in our flow, Phase 1 points ARE evaluated.
            # So for the best_x from Phase 1, we have best_f.
            
            # We need to be careful: if we start refining a point that is NOT the current global best,
            # we need its function value. 
            # To simplify, we will only refine points that we have already evaluated in Phase 1.
            
            # For the primary refinement of best_x:
            fx = best_f
            # We don't call func(best_x) again to save budget.
            
            sigma = 1.0
            successes = 0
            window = 10
            
            for i in range(max_calls):
                if call_count >= budget or time_left() < 1.0:
                    break
                
                y = np.clip(x + sigma * rng.standard_normal(dim), -5, 5)
                fy = func(y)
                call_count += 1
                
                if fy < fx:
                    x = y
                    fx = fy
                    successes += 1
                    # Update global best
                    if fx < current_best_f:
                        current_best_f = fx
                        current_best_x = x.copy()
                else:
                    successes = max(0, successes - 1)
                
                # Adapt sigma
                if i > 0 and i % window == 0:
                    if successes > window / 2:
                        sigma *= 1.5
                    elif successes < window / 4:
                        sigma *= 0.7
                    successes = 0
                    
                # Clamp sigma
                sigma = np.clip(sigma, 0.01, 5.0)

            return current_best_f, current_best_x

        # Determine how many calls to spend on refinement
        # We have `budget - call_count` calls left.
        # Let's spend most of it on refining the best point.
        # But to escape local optima, we might want to restart.
        # Given the "STAGNATION" directive, a structural change is needed.
        # The current approach is a single long refinement. 
        # A better structural change for f21 (many peaks) is **Multi-Start Local Search**.
        
        # Let's restart the logic for Phase 2 to be a multi-start approach.
        # We will pick the top K points from Phase 1 and refine each for a short time.
        
        # To do this, we need the function values of the LHS points.
        # We didn't store them in the loop above. Let's refactor Phase 1 to store them.
        
        # --- Refactored Implementation ---
        
        # Reset state
        best_x = None
        best_f = float('inf')
        call_count = 0
        
        # Phase 1: Evaluate LHS points
        n_lhs = 30  # Increase slightly for better candidate pool
        if n_lhs > budget:
            n_lhs = budget
        
        lhs_pts = lhs_points(n_lhs, dim, rng)
        lhs_vals = []
        
        for pt in lhs_pts:
            if call_count >= budget or time_left() < 2.0:
                break
            f_val = func(pt)
            call_count += 1
            lhs_vals.append((f_val, pt))
            
        if not lhs_vals:
            # Fallback
            best_x = np.zeros(dim)
            best_f = func(best_x)
            call_count += 1
            return best_x, best_f

        # Sort by function value
        lhs_vals.sort(key=lambda x: x[0])
        
        # The best point found so far
        best_f, best_x = lhs_vals[0]
        
        # Phase 2: Multi-Start Refinement
        # We have `remaining = budget - call_count` calls.
        # We will distribute these among the top `n_restarts` points.
        # Let's pick the top 5 points if we have enough budget.
        
        remaining = budget - call_count
        if remaining <= 0:
            return best_x, best_f
        
        # How many restarts?
        # Each restart needs at least 10 calls to be meaningful.
        max_restarts = min(5, remaining // 10)
        if max_restarts < 1:
            max_restarts = 1
            
        calls_per_restart = remaining // max_restarts
        
        # We refine the top `max_restarts` points
        for i in range(max_restarts):
            if call_count >= budget or time_left() < 1.0:
                break
            
            start_f, start_x = lhs_vals[i]
            
            # If this start point is the current global best, we don't need to re-evaluate.
            # Otherwise, we need to evaluate it to start the ES.
            # However, we already evaluated it in Phase 1. So we have start_f.
            
            x = start_x.copy()
            fx = start_f
            
            # Run local search
            sigma = 1.0
            successes = 0
            window = 10
            
            for step in range(calls_per_restart):
                if call_count >= budget or time_left() < 0.5:
                    break
                
                y = np.clip(x + sigma * rng.standard_normal(dim), -5, 5)
                fy = func(y)
                call_count += 1
                
                if fy < fx:
                    x = y
                    fx = fy
                    successes += 1
                    if fx < best_f:
                        best_f = fx
                        best_x = x.copy()
                else:
                    successes = max(0, successes - 1)
                
                if step > 0 and step % window == 0:
                    if successes > window / 2:
                        sigma *= 1.5
                    elif successes < window / 4:
                        sigma *= 0.7
                    successes = 0
                sigma = np.clip(sigma, 0.01, 5.0)

        return best_x, best_f
# EVOLVE-BLOCK-END
