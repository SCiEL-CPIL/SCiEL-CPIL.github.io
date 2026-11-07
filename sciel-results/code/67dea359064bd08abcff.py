"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """Hybrid Multi-Start Nelder-Mead optimizer."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 295.0  # Safety margin: 300s limit
        
        dim = self.dim
        budget = self.budget
        
        # 1. Initialization: Multi-start Nelder-Mead
        # We divide the budget into phases.
        # Phase 1: Global exploration via multiple local searches.
        # Phase 2: Refinement of the best point found.
        
        # For 5D, Nelder-Mead is very fast. 
        # We can afford many restarts.
        # Let's aim for ~50-100 restarts in the first phase.
        
        # To maximize AOCC, we want to drop the value quickly.
        # A uniform random start might take a while to hit a good basin.
        # However, f7 (Step Ellipsoidal) is unimodal (mostly), but with a step.
        # Actually, BBOB f7 is "Step Ellipsoidal". It has a global minimum and 
        # a "step" that creates a plateau or discontinuity in the gradient sense, 
        # but it's continuous. The function value drops sharply near the optimum.
        
        # Strategy:
        # 1. Generate a set of diverse starting points.
        # 2. Run a limited-budget Nelder-Mead on each.
        # 3. Keep the best one.
        # 4. Run a final, longer Nelder-Mead on the best point.
        
        rng = np.random.default_rng(42)
        
        best_x = None
        best_f = np.inf
        calls_made = 0
        
        # Helper to run Nelder-Mead with a strict call limit
        def nelder_mead(x0, f0, max_calls):
            nonlocal calls_made
            if calls_made >= budget:
                return x0, f0
            
            # Simplex setup
            # x0 is (dim,)
            # Initial simplex: x0 and x0 + delta * e_i
            delta = 0.5  # Initial step size
            simplex = np.empty((dim + 1, dim))
            simplex[0] = x0
            for i in range(dim):
                pt = x0.copy()
                pt[i] += delta
                simplex[i+1] = np.clip(pt, -5, 5)
            
            # Evaluate initial simplex
            fvals = np.empty(dim + 1)
            fvals[0] = f0
            for i in range(dim):
                if calls_made >= budget:
                    return x0, f0
                fvals[i+1] = func(simplex[i+1])
                calls_made += 1
                
            # Sort simplex by fitness (ascending)
            order = np.argsort(fvals)
            simplex = simplex[order]
            fvals = fvals[order]
            
            # Nelder-Mead parameters
            alpha = 1.0  # Reflection
            gamma = 2.0  # Expansion
            rho = 0.5    # Contraction
            sigma = 0.5  # Shrink
            
            # Iteration limit per restart is small
            # We don't want to spend too much on one bad start
            max_iter = max_calls # Roughly, each iter costs 1-2 calls
            # Actually, standard NM costs 1 call per iter usually (reflection), 
            # sometimes 2 (expansion/contraction).
            
            current_best_x = simplex[0].copy()
            current_best_f = fvals[0]
            
            for _ in range(max_iter):
                if calls_made >= budget:
                    break
                if time.time() > deadline:
                    break
                    
                # 1. Reflection
                centroid = np.mean(simplex[1:], axis=0)
                x_r = centroid + alpha * (centroid - simplex[0])
                x_r = np.clip(x_r, -5, 5)
                
                if calls_made >= budget:
                    break
                f_r = func(x_r)
                calls_made += 1
                
                if f_r < fvals[-1]:
                    # Reflection is better than worst
                    if f_r < fvals[0]:
                        # Try expansion
                        x_e = centroid + gamma * (x_r - centroid)
                        x_e = np.clip(x_e, -5, 5)
                        if calls_made >= budget:
                            # Can't expand, just accept reflection
                            simplex[0] = x_r
                            fvals[0] = f_r
                            # Re-sort
                            order = np.argsort(fvals)
                            simplex = simplex[order]
                            fvals = fvals[order]
                            if fvals[0] < current_best_f:
                                current_best_f = fvals[0]
                                current_best_x = simplex[0].copy()
                            continue
                        f_e = func(x_e)
                        calls_made += 1
                        if f_e < f_r:
                            simplex[0] = x_e
                            fvals[0] = f_e
                        else:
                            simplex[0] = x_r
                            fvals[0] = f_r
                    else:
                        # Accept reflection
                        simplex[0] = x_r
                        fvals[0] = f_r
                else:
                    # Reflection is not better than worst
                    # Try contraction
                    if f_r < fvals[-1]:
                        # Outside contraction
                        x_c = centroid + rho * (x_r - centroid)
                    else:
                        # Inside contraction
                        x_c = centroid - rho * (x_r - centroid)
                    
                    x_c = np.clip(x_c, -5, 5)
                    if calls_made >= budget:
                        break
                    f_c = func(x_c)
                    calls_made += 1
                    
                    if f_c < min(f_r, fvals[-1]):
                        simplex[0] = x_c
                        fvals[0] = f_c
                    else:
                        # Shrink
                        for i in range(1, dim + 1):
                            simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                            if calls_made >= budget:
                                break
                            fvals[i] = func(simplex[i])
                            calls_made += 1
                        
                # Re-sort
                order = np.argsort(fvals)
                simplex = simplex[order]
                fvals = fvals[order]
                
                # Update local best
                if fvals[0] < current_best_f:
                    current_best_f = fvals[0]
                    current_best_x = simplex[0].copy()
                    
                # Convergence check
                if np.max(fvals) - np.min(fvals) < 1e-8:
                    break
                    
            return current_best_x, current_best_f

        # Phase 1: Multi-start
        # Let's use 20% of budget for initial exploration
        # If budget=500, that's 100 calls.
        # If we do 20 starts, each gets 5 calls? That's too few for NM.
        # NM needs at least dim+1 = 6 calls to start.
        # Let's do 10 starts, each with ~10-15 calls budget.
        
        num_starts = 10
        calls_per_start = max(10, (budget // 5) // num_starts) # Roughly 20% budget total
        
        for i in range(num_starts):
            if calls_made >= budget:
                break
            if time.time() > deadline:
                break
                
            # Generate diverse start
            # Mix of uniform and some structured
            if i == 0:
                x0 = np.zeros(dim) # Center is often good for BBOB
            else:
                x0 = rng.uniform(-5, 5, dim)
            
            x0 = np.clip(x0, -5, 5)
            
            if calls_made >= budget:
                break
            f0 = func(x0)
            calls_made += 1
            
            if f0 < best_f:
                best_f = f0
                best_x = x0.copy()
            
            # Run NM
            x_opt, f_opt = nelder_mead(x0, f0, calls_per_start)
            
            if f_opt < best_f:
                best_f = f_opt
                best_x = x_opt.copy()
                
            # If we found a very good point, we might want to stop early?
            # No, let's keep exploring to find potentially better ones.
            
        # Phase 2: Refinement
        # Use remaining budget on the best point found
        if best_x is not None and calls_made < budget:
            remaining = budget - calls_made
            # Run a longer NM
            x_final, f_final = nelder_mead(best_x, best_f, remaining)
            if f_final < best_f:
                best_f = f_final
                best_x = x_final.copy()

        return best_x, best_f
# EVOLVE-BLOCK-END
