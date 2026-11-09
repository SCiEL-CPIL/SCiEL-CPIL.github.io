"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-start Nelder-Mead simplex method."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        rng = np.random.default_rng(42)
        dim = self.dim
        budget = self.budget
        start_time = time.time()
        deadline = start_time + 270  # Safety margin

        best_x = None
        best_f = np.inf
        used_evals = 0

        def nelder_mead(x0, fx0, max_evals, rng):
            """
            Perform Nelder-Mead optimization starting from x0.
            Returns (best_x, best_f, evals_used)
            """
            n = dim
            # Initialize simplex
            simplex = np.zeros((n + 1, n))
            fvals = np.zeros(n + 1)
            
            # Vertex 0 is x0
            simplex[0] = x0
            fvals[0] = fx0
            evals_used = 1

            # Initialize other vertices
            for i in range(1, n + 1):
                dx = np.zeros(n)
                if i - 1 < n:
                    dx[i - 1] = 1.0
                simplex[i] = np.clip(x0 + dx, -5, 5)
                fvals[i] = func(simplex[i])
                evals_used += 1
                
                if fvals[i] < best_f:
                    # Update global best
                    pass # We track global best outside or return it

            # Sort initial simplex
            order = np.argsort(fvals)
            simplex = simplex[order]
            fvals = fvals[order]

            alpha = 1.0  # Reflection
            gamma = 2.0  # Expansion
            rho = 0.5    # Contraction
            sigma = 0.5  # Shrink

            while evals_used < max_evals:
                if time.time() > deadline:
                    break

                # Calculate centroid of all but worst
                centroid = np.mean(simplex[:-1], axis=0)

                # 1. Reflection
                xr = centroid + alpha * (centroid - simplex[-1])
                xr = np.clip(xr, -5, 5)
                fxr = func(xr)
                evals_used += 1

                if fxr < fvals[0]:
                    # 2. Expansion
                    if fxr < fvals[1]:
                        xe = centroid + gamma * (centroid - simplex[-1])
                        xe = np.clip(xe, -5, 5)
                        fxe = func(xe)
                        evals_used += 1
                        if fxe < fxr:
                            simplex[-1] = xe
                            fvals[-1] = fxe
                        else:
                            simplex[-1] = xr
                            fvals[-1] = fxr
                    else:
                        simplex[-1] = xr
                        fvals[-1] = fxr
                elif fxr < fvals[-2]:
                    # Accept reflection
                    simplex[-1] = xr
                    fvals[-1] = fxr
                else:
                    # 3. Contraction
                    if fxr < fvals[-1]:
                        # Outside contraction
                        xc = centroid + rho * (xr - centroid)
                        xc = np.clip(xc, -5, 5)
                    else:
                        # Inside contraction
                        xc = centroid - rho * (xr - centroid)
                        xc = np.clip(xc, -5, 5)
                    
                    fxc = func(xc)
                    evals_used += 1

                    if fxc < min(fxr, fvals[-1]):
                        simplex[-1] = xc
                        fvals[-1] = fxc
                    else:
                        # 4. Shrink
                        for i in range(1, n + 1):
                            simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                            simplex[i] = np.clip(simplex[i], -5, 5)
                            fvals[i] = func(simplex[i])
                            evals_used += 1
                            if evals_used >= max_evals:
                                break

                # Re-sort simplex
                order = np.argsort(fvals)
                simplex = simplex[order]
                fvals = fvals[order]

            # Return best in this local search
            best_idx = np.argmin(fvals)
            return simplex[best_idx], fvals[best_idx], evals_used

        # Strategy: 
        # NM requires n+1 evals to setup + 1 per step.
        # For dim=5, setup is 6 evals. 
        # We can do multiple restarts.
        # Early restarts are crucial for AOCC.
        
        # Phase 1: 10 restarts, 20 evals each (approx 14 steps each)
        # Phase 2: 5 restarts, 30 evals each
        # Phase 3: Refine best with 100 evals
        # Phase 4: Refine best with remaining
        
        restarts_config = [
            (10, 20),  # 10 restarts, 20 evals each -> 200 evals
            (5, 30),   # 5 restarts, 30 evals each -> 150 evals
            (1, 100),  # 1 deep refinement -> 100 evals
            (1, 50)    # Final push -> 50 evals
        ]
        
        # Track best point for refinement
        current_best_x = None
        current_best_f = np.inf

        for num_restarts, evals_per_restart in restarts_config:
            for i in range(num_restarts):
                if used_evals >= budget:
                    break
                if time.time() > deadline:
                    break
                
                # Determine starting point
                if i == 0 and current_best_x is not None:
                    # First restart of a phase refines the previous best slightly
                    # Or just use random for diversity? 
                    # Let's use random for new restarts, but for the "refinement" phases, 
                    # we should start from the best found so far.
                    # The config above mixes them. Let's adjust logic:
                    # If it's the last phase, start from best.
                    # For now, let's just do random restarts for first two phases, 
                    # and specific refinement for later ones.
                    pass
                
                # Simplified logic:
                # Phases 1 & 2: Random restarts
                # Phase 3: Refine best found in 1&2
                # Phase 4: Refine best found in 3
                
                if i == 0 and current_best_x is not None and num_restarts == 1:
                    # This is a refinement phase
                    x0 = current_best_x.copy()
                    fx0 = current_best_f
                    # Note: We already evaluated fx0? No, we need to pass it.
                    # But if we restart, we might re-evaluate. 
                    # To save budget, we assume we don't re-evaluate if we have the value.
                    # However, NM setup evaluates neighbors. 
                    # If we start from best, we have fx0.
                    # But NM creates new vertices. 
                    # Let's just treat it as a new local search.
                    
                    # Actually, to be safe and simple, we always generate a new x0 
                    # unless it's a dedicated refinement step.
                    
                    # Let's re-structure:
                    # Phase 1: 10 random restarts, 20 evals
                    # Phase 2: 5 random restarts, 30 evals
                    # Phase 3: 1 restart from best, 100 evals
                    # Phase 4: 1 restart from best, remaining evals
                    
                    pass
                
                # Implementation of the loop logic:
                
                if num_restarts == 1 and current_best_x is not None:
                    # Refinement phase
                    x0 = current_best_x
                    fx0 = current_best_f
                    # We need to be careful: if we pass fx0, we don't call func(x0) again in setup?
                    # My nelder_mead implementation takes fx0 and uses it for vertex 0.
                    # So we save 1 eval.
                    max_e = min(evals_per_restart, budget - used_evals)
                    if max_e < 2: break
                    
                    x, fx, e_used = nelder_mead(x0, fx0, max_e, rng)
                    used_evals += e_used
                    if fx < current_best_f:
                        current_best_f = fx
                        current_best_x = x.copy()
                else:
                    # Exploration phase
                    x0 = rng.uniform(-5, 5, dim)
                    fx0 = func(x0)
                    used_evals += 1
                    if fx0 < current_best_f:
                        current_best_f = fx0
                        current_best_x = x0.copy()
                    
                    max_e = min(evals_per_restart, budget - used_evals)
                    if max_e < 1: break
                    
                    # Pass remaining budget for this restart
                    # Note: nelder_mead will use up to max_e evals.
                    # But we already used 1 for x0.
                    # So we give it max_e - 1? 
                    # No, nelder_mead takes max_evals as total budget for the *local search*.
                    # But it doesn't know about the initial eval if we pass fx0?
                    # In my implementation:
                    # fvals[0] = fx0, evals_used = 1.
                    # Then loop while evals_used < max_evals.
                    # So if I want to spend K total evals, I pass K.
                    
                    x, fx, e_used = nelder_mead(x0, fx0, max_e, rng)
                    used_evals += e_used - 1 # Subtract the 1 we already counted for x0?
                    # Wait, nelder_mead returns e_used which includes the initial 1?
                    # In my code: evals_used starts at 1.
                    # So if I pass max_e=20, it will use up to 20 evals total.
                    # I already called func(x0) once.
                    # So I should pass max_e = remaining_budget.
                    # And then used_evals += (e_used - 1) because the first eval was counted outside?
                    # Let's trace:
                    # 1. func(x0) called. used_evals += 1.
                    # 2. nelder_mead(x0, fx0, max_e).
                    #    Inside: evals_used = 1.
                    #    Loop: while evals_used < max_e.
                    #    It will call func() for vertices 1..n (n calls). evals_used becomes 1+n.
                    #    Then iterations.
                    #    Total evals used by NM = e_used.
                    #    But NM assumes it "owns" the first eval.
                    #    Since I already did it, I should tell NM to start with that?
                    #    Yes, I passed fx0.
                    #    So NM thinks it has used 1 eval.
                    #    So if I want to spend 19 MORE evals, I pass max_e = 20.
                    #    Total cost = 1 (outside) + (e_used - 1) (inside)?
                    #    No. If NM runs for max_e=20, it performs 19 additional function calls.
                    #    Total function calls = 1 (outside) + 19 (inside) = 20.
                    #    e_used will be 20.
                    #    So used_evals should increase by e_used - 1? 
                    #    No, used_evals was incremented by 1 for x0.
                    #    NM uses e_used total, of which 1 is x0.
                    #    So new used_evals = old_used_evals + (e_used - 1).
                    
                    if fx < current_best_f:
                        current_best_f = fx
                        current_best_x = x.copy()
                        
        return current_best_x, current_best_f
# EVOLVE-BLOCK-END
