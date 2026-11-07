"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """
    Structural change: Replace the stochastic (1+1)-ES with a deterministic
    multi-scale coordinate descent / grid refinement approach.
    
    Mechanism:
    1. Deterministic Initialization: Evaluate the origin and a set of symmetric
       points to establish a baseline.
    2. Multi-Scale Coordinate Descent: Systematically explore the neighborhood
       of the best point found so far using a fixed set of step sizes, starting
       from large steps (e.g., 5.0) down to fine steps (e.g., 0.01).
    3. Greedy Axis Selection: At each scale, iterate through dimensions and
       try moving +step and -step along that axis. Accept if better.
    4. Restart Logic: If progress stalls at fine scales, jump back to a
       coarser scale or restart from a slightly perturbed version of the best
       point to escape local minima, leveraging the multi-modal nature of f21.
    5. Deadline Guard: Explicit time tracking to ensure safe termination.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 320.0  # Safety margin before 330s kill
        dim = self.dim
        budget = self.budget
        
        def time_left():
            return deadline - time.time()

        best_x = None
        best_f = float('inf')
        call_count = 0

        def eval_point(x):
            nonlocal best_x, best_f, call_count
            if call_count >= budget or time_left() < 0.5:
                return None
            # Ensure x is a valid numpy array
            x = np.asarray(x, dtype=float)
            # Clip to domain [-5, 5]
            x = np.clip(x, -5.0, 5.0)
            try:
                fx = float(func(x))
            except Exception:
                return None
            call_count += 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            return fx

        # --- Phase 1: Deterministic Coarse Initialization ---
        # Evaluate origin
        eval_point(np.zeros(dim))
        
        # Evaluate midpoints of each axis (+/- 2.5)
        for d in range(dim):
            if time_left() < 1.0:
                break
            # Positive midpoint
            x_pos = np.zeros(dim)
            x_pos[d] = 2.5
            eval_point(x_pos)
            
            # Negative midpoint
            x_neg = np.zeros(dim)
            x_neg[d] = -2.5
            eval_point(x_neg)
            
            if time_left() < 1.0:
                break

        # Evaluate corners of a smaller box (e.g., +/- 1.0) to catch peaks closer to center
        # Only if budget allows. 2^5 = 32 points.
        if budget - call_count > 40:
            for mask in range(2 ** dim):
                x = np.zeros(dim)
                for d in range(dim):
                    if (mask >> d) & 1:
                        x[d] = 1.0
                    else:
                        x[d] = -1.0
                eval_point(x)
                if time_left() < 1.0:
                    break
                if call_count >= budget:
                    break
                if time_left() < 5.0: # Stop if very little time left
                    break

        if best_x is None:
            # Fallback if something went wrong
            best_x = np.zeros(dim)
            best_f = eval_point(best_x) or 0.0
            return best_x, best_f

        # --- Phase 2: Multi-Scale Coordinate Descent ---
        # Step sizes to try, from coarse to fine.
        # We will loop through these scales multiple times (restarts) if stuck.
        step_sizes = [5.0, 2.5, 1.25, 0.625, 0.3125, 0.15625, 0.078125, 0.0390625, 0.01953125, 0.01]
        
        current_x = best_x.copy()
        current_f = best_f
        
        # To handle multi-modality, we will perform several "rounds" of descent.
        # In each round, we start from the current best and try to improve.
        # If we make no progress in a full sweep of scales, we try a random perturbation
        # at a coarser scale to jump to a new basin.
        
        max_rounds = 10  # Limit restarts to prevent infinite loops
        rng = np.random.default_rng(42) # Deterministic "randomness" for restarts
        
        for round_num in range(max_rounds):
            if call_count >= budget or time_left() < 2.0:
                break
                
            # Reset current state to global best for this round
            current_x = best_x.copy()
            current_f = best_f
            improved_in_round = False
            
            # Iterate through scales from coarse to fine
            for step in step_sizes:
                if call_count >= budget or time_left() < 0.5:
                    break
                
                # For each dimension
                for d in range(dim):
                    if call_count >= budget or time_left() < 0.5:
                        break
                    
                    # Try +step
                    x_try = current_x.copy()
                    x_try[d] += step
                    if -5.0 <= x_try[d] <= 5.0:
                        f_try = eval_point(x_try)
                        if f_try is not None and f_try < current_f - 1e-12:
                            current_x = x_try
                            current_f = f_try
                            improved_in_round = True
                            if f_try < best_f:
                                best_f = f_try
                                best_x = current_x.copy()
                    
                    if call_count >= budget or time_left() < 0.5:
                        break

                    # Try -step
                    x_try = current_x.copy()
                    x_try[d] -= step
                    if -5.0 <= x_try[d] <= 5.0:
                        f_try = eval_point(x_try)
                        if f_try is not None and f_try < current_f - 1e-12:
                            current_x = x_try
                            current_f = f_try
                            improved_in_round = True
                            if f_try < best_f:
                                best_f = f_try
                                best_x = current_x.copy()
                    
                    if call_count >= budget or time_left() < 0.5:
                        break

            # If we didn't improve in this round, and we have budget left,
            # perform a random restart from a perturbed version of the best point
            # to escape local optima.
            if not improved_in_round:
                if call_count >= budget or time_left() < 2.0:
                    break
                
                # Perturb the best point with a relatively large step to jump to a new basin
                # Use a few random restarts
                n_restarts = 3
                for r in range(n_restarts):
                    if call_count >= budget or time_left() < 1.0:
                        break
                    
                    # Random step size for restart
                    restart_step = 2.0 * (0.5 ** r) # 2.0, 1.0, 0.5
                    x_new = best_x + restart_step * rng.standard_normal(dim)
                    x_new = np.clip(x_new, -5.0, 5.0)
                    
                    f_new = eval_point(x_new)
                    if f_new is not None:
                        # If this new point is better than current best, update
                        # Otherwise, we can optionally try a short local search from it
                        # But for now, just accepting the best point found by random restart
                        # is a valid structural change: "Jump and Climb"
                        pass 

        return best_x, best_f
# EVOLVE-BLOCK-END
