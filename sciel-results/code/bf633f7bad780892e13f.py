# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """Nelder-Mead optimizer with multi-restart strategy for BBOB f18."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        deadline = time.time() + 270.0
        
        calls = 0
        best_x = None
        best_f = np.inf
        
        def eval_pt(x):
            nonlocal calls, best_x, best_f
            if calls >= budget or time.time() > deadline:
                return best_f
            x = np.clip(np.asarray(x, dtype=float), -5.0, 5.0)
            fx = float(func(x))
            calls += 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            return fx

        def nelder_mead(start_x, max_calls):
            # Local state for this restart
            # Simplex initialization
            step = 0.5
            simplex = [start_x.copy()]
            for i in range(dim):
                pt = start_x.copy()
                pt[i] += step
                simplex.append(pt)
            
            # Evaluate initial simplex
            fvals = [eval_pt(pt) for pt in simplex]
            
            # NM Parameters
            alpha = 1.0
            gamma = 2.0
            rho = 0.5
            sigma = 0.5
            
            no_improve_count = 0
            local_best_f = best_f # Track global best to detect stagnation
            
            while calls < budget and time.time() < deadline and (calls < max_calls or max_calls == -1):
                # Sort simplex by function value
                order = np.argsort(fvals)
                simplex = [simplex[i] for i in order]
                fvals = [fvals[i] for i in order]
                
                current_best_f = fvals[0]
                
                # Stagnation check: if no improvement in global best for 20 iterations
                if current_best_f >= local_best_f - 1e-12:
                    no_improve_count += 1
                else:
                    no_improve_count = 0
                    local_best_f = current_best_f
                
                if no_improve_count > 20:
                    break # Trigger restart
                
                # Centroid of all but worst
                centroid = np.mean(simplex[:-1], axis=0)
                worst_pt = simplex[-1]
                worst_f = fvals[-1]
                
                # Reflection
                x_r = centroid + alpha * (centroid - worst_pt)
                f_r = eval_pt(x_r)
                
                if f_r < fvals[0]:
                    # Expansion
                    x_e = centroid + gamma * (centroid - worst_pt)
                    f_e = eval_pt(x_e)
                    if f_e < f_r:
                        simplex[-1] = x_e
                        fvals[-1] = f_e
                    else:
                        simplex[-1] = x_r
                        fvals[-1] = f_r
                elif f_r < fvals[-2]:
                    # Reflection accepted
                    simplex[-1] = x_r
                    fvals[-1] = f_r
                else:
                    # Contraction
                    if f_r < worst_f:
                        # Outside contraction
                        x_c = centroid + rho * (x_r - centroid)
                    else:
                        # Inside contraction
                        x_c = centroid - rho * (centroid - worst_pt)
                    
                    f_c = eval_pt(x_c)
                    
                    if f_c < worst_f:
                        simplex[-1] = x_c
                        fvals[-1] = f_c
                    else:
                        # Shrink
                        for i in range(1, len(simplex)):
                            simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                            fvals[i] = eval_pt(simplex[i])
            
            return best_x, best_f

        # Multi-restart strategy
        # Divide budget into chunks.
        # First run gets more budget to find a good basin.
        # Subsequent runs get less.
        rng = np.random.default_rng(42)
        
        # Start at center
        start_x = np.zeros(dim)
        
        # Run 1: Large budget
        # We don't know how many calls NM will take, so we give it a large chunk
        # and rely on the internal stagnation check to break early if stuck.
        # If it uses all budget, we are done.
        
        # To be safe, we run in a loop of restarts.
        # Each restart starts from a random point if the previous one stagnated or finished.
        
        # Actually, the internal `nelder_mead` function breaks on stagnation.
        # We can call it multiple times.
        
        # Strategy:
        # 1. Start at 0.
        # 2. If it stagnates (breaks early), start at a random point.
        # 3. Repeat until budget is exhausted.
        
        while calls < budget and time.time() < deadline:
            if best_x is None:
                start_x = np.zeros(dim)
            else:
                # Perturb best_x or pick random?
                # Random is better for multimodal.
                start_x = rng.uniform(-4.0, 4.0, dim)
            
            # Give this run a chance to use remaining budget
            # But we want to ensure we try a few restarts if budget allows.
            # Let's allocate 50% of remaining budget to each restart?
            # Or just let it run until stagnation.
            
            nelder_mead(start_x, -1) # -1 means unlimited, relies on stagnation or budget
            
            # If nelder_mead returned because of stagnation, we loop and restart.
            # If it returned because of budget, the while condition will fail.
            # If it returned because of time, the while condition will fail.
            
            # Small check: if we just started and used 0 calls (shouldn't happen), break to avoid infinite loop
            # nelder_mead always consumes at least dim+1 calls.

        return best_x, best_f
# EVOLVE-BLOCK-END
