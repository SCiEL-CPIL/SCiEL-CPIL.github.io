"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """Multi-start Nelder-Mead optimizer for BBOB f9."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 295  # Safety margin before 330s kill
        rng = np.random.default_rng(42)
        n = self.dim
        budget = self.budget

        best_x = np.zeros(n)
        best_f = func(best_x)
        calls_used = 1

        def check_deadline():
            return time.time() >= deadline

        def do_call(x):
            nonlocal calls_used
            fx = func(x)
            calls_used += 1
            return fx

        # Nelder-Mead parameters
        # Initial simplex size
        initial_step = 0.5
        # Convergence criteria
        ftol = 1e-12
        xtol = 1e-10
        max_iter_nm = 1000

        def nelder_mead(x0, f0, max_calls):
            nonlocal best_f, best_x, calls_used
            
            # Create initial simplex
            simplex = [x0.copy()]
            fvals = [f0]
            
            for i in range(n):
                x_new = x0.copy()
                x_new[i] += initial_step
                x_new = np.clip(x_new, -5, 5)
                if calls_used >= budget or check_deadline():
                    break
                f_new = do_call(x_new)
                simplex.append(x_new)
                fvals.append(f_new)
                if f_new < best_f:
                    best_f = f_new
                    best_x = x_new.copy()
            
            if len(simplex) < n + 1:
                return

            for _ in range(max_iter_nm):
                if calls_used >= budget or check_deadline():
                    break
                
                # Sort simplex by function value
                order = np.argsort(fvals)
                simplex = [simplex[i] for i in order]
                fvals = [fvals[i] for i in order]
                
                # Check convergence
                if fvals[-1] - fvals[0] < ftol:
                    break
                
                # Centroid of all but the worst point
                centroid = np.mean(simplex[:-1], axis=0)
                
                # Reflection
                alpha = 1.0
                x_r = centroid + alpha * (centroid - simplex[-1])
                x_r = np.clip(x_r, -5, 5)
                
                if calls_used >= budget or check_deadline():
                    break
                f_r = do_call(x_r)
                if f_r < best_f:
                    best_f = f_r
                    best_x = x_r.copy()
                
                # Reflection is good
                if fvals[0] <= f_r < fvals[-2]:
                    simplex[-1] = x_r
                    fvals[-1] = f_r
                # Expansion
                elif f_r < fvals[0]:
                    gamma = 2.0
                    x_e = centroid + gamma * (centroid - simplex[-1])
                    x_e = np.clip(x_e, -5, 5)
                    if calls_used >= budget or check_deadline():
                        break
                    f_e = do_call(x_e)
                    if f_e < best_f:
                        best_f = f_e
                        best_x = x_e.copy()
                    if f_e < f_r:
                        simplex[-1] = x_e
                        fvals[-1] = f_e
                    else:
                        simplex[-1] = x_r
                        fvals[-1] = f_r
                # Contraction
                else:
                    rho = 0.5
                    x_c = centroid + rho * (simplex[-1] - centroid)
                    x_c = np.clip(x_c, -5, 5)
                    if calls_used >= budget or check_deadline():
                        break
                    f_c = do_call(x_c)
                    if f_c < best_f:
                        best_f = f_c
                        best_x = x_c.copy()
                    if f_c < fvals[-1]:
                        simplex[-1] = x_c
                        fvals[-1] = f_c
                    # Shrink
                    else:
                        for i in range(1, len(simplex)):
                            simplex[i] = simplex[0] + 0.5 * (simplex[i] - simplex[0])
                            simplex[i] = np.clip(simplex[i], -5, 5)
                            if calls_used >= budget or check_deadline():
                                break
                            f_new = do_call(simplex[i])
                            fvals[i] = f_new
                            if f_new < best_f:
                                best_f = f_new
                                best_x = simplex[i].copy()
                        if check_deadline() or calls_used >= budget:
                            break

        # Main loop: Multi-start
        # We will perform random restarts and run Nelder-Mead from each start
        # Strategy: Start with a few coarse random points, then polish the best ones.
        
        # Phase 1: Coarse random sampling to find a good basin
        # Use about 20% of budget for initial exploration
        exploration_budget = int(budget * 0.1)
        
        for _ in range(exploration_budget):
            if calls_used >= budget or check_deadline():
                break
            x_new = 5.0 * (rng.random(n) - 0.5)
            x_new = np.clip(x_new, -5, 5)
            f_new = do_call(x_new)
            if f_new < best_f:
                best_f = f_new
                best_x = x_new.copy()
        
        # Phase 2: Polishing with Nelder-Mead
        # We will run NM on the best point found so far.
        # If it converges or stagnates, we might try another random point if budget allows.
        # However, NM is very efficient. Let's just run it on the current best.
        # To be safe against getting stuck in a bad local min, we could restart NM 
        # from a slightly perturbed best_x if it doesn't improve much.
        
        # Run NM from best_x
        nelder_mead(best_x.copy(), best_f, max_calls=1000)
        
        # If we still have budget, we can try a few more random restarts with NM
        # But NM is fast. Let's see how many calls we used.
        # If we used less than 50% of budget, we can afford more restarts.
        
        # A better strategy for BBOB: 
        # 1. Find a good point via random search.
        # 2. Polish it.
        # 3. If budget remains, try another random point, polish it.
        # 4. Keep the global best.
        
        # Let's implement a loop that alternates between random search and polishing
        # but prioritizes polishing if we are in a good region.
        
        # Actually, the previous logic did one NM run. 
        # Let's add a loop for multiple NM restarts if time/calls allow.
        
        while calls_used < budget and not check_deadline():
            # Pick a new start point
            # 50% chance to perturb current best, 50% random
            if rng.random() < 0.5 and calls_used > exploration_budget:
                x_start = best_x + 0.5 * rng.standard_normal(n)
            else:
                x_start = 5.0 * (rng.random(n) - 0.5)
            x_start = np.clip(x_start, -5, 5)
            
            f_start = do_call(x_start)
            if f_start < best_f:
                best_f = f_start
                best_x = x_start.copy()
            
            # Run NM from this start
            # Limit calls per NM run to avoid wasting budget on a bad start
            # We can estimate remaining budget and allocate a portion
            remaining = budget - calls_used
            # Use at most 50 calls for this NM run if we are in later stages?
            # Or just let it run until convergence or budget check.
            # NM converges fast.
            nelder_mead(x_start, f_start, max_calls=100)

        return best_x, best_f
# EVOLVE-BLOCK-END
