"""Nelder-Mead Simplex Optimizer for BBOB f7 (Step ellipsoidal)."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """Nelder-Mead Simplex optimizer for BBOB f7."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        # 300s limit. 280s is a safe margin.
        deadline = start_time + 280.0
        
        d = self.dim
        budget = self.budget
        rng = np.random.default_rng(42)
        
        best_x_global = None
        best_f_global = np.inf
        calls_used = 0
        
        def evaluate(x):
            nonlocal calls_used, best_f_global, best_x_global
            if calls_used >= budget:
                return np.inf
            if time.time() > deadline:
                return np.inf
            fx = func(x)
            calls_used += 1
            if fx < best_f_global:
                best_f_global = fx
                best_x_global = x.copy()
            return fx

        # Initial point: Try center and a few random points to pick a good start
        # Nelder-Mead is sensitive to the starting region.
        # For f7, the minimum is in [-4, 4]^5.
        
        candidates = [np.zeros(d)]
        for _ in range(5):
            candidates.append(rng.uniform(-4.0, 4.0, d))
        
        # Evaluate candidates to find a good starting point for the simplex
        # This uses 6 calls.
        # If we run out of budget, we stop.
        start_x = None
        min_f_start = np.inf
        
        for c in candidates:
            f_c = evaluate(c)
            if f_c < min_f_start:
                min_f_start = f_c
                start_x = c.copy()
                
        if start_x is None:
            # Fallback
            start_x = np.zeros(d)
            f_c = evaluate(start_x)
            if f_c < min_f_start:
                min_f_start = f_c

        # Nelder-Mead Parameters
        alpha = 1.0    # Reflection
        gamma = 2.0    # Expansion
        rho = 0.5      # Contraction
        sigma = 0.5    # Shrinkage
        
        # Initialize Simplex
        # Start at start_x, expand along axes
        simplex = [start_x.copy()]
        fvals = [min_f_start]
        
        step_size = 0.5
        for i in range(d):
            x_new = start_x.copy()
            x_new[i] += step_size
            simplex.append(x_new)
            f_new = evaluate(x_new)
            fvals.append(f_new)
            
        # Main Loop
        while calls_used < budget:
            if time.time() > deadline:
                break
            
            # Sort simplex by function value
            indices = np.argsort(fvals)
            simplex = [simplex[i] for i in indices]
            fvals = [fvals[i] for i in indices]
            
            # Check for convergence: if the spread of function values is tiny
            # or the simplex size is tiny
            f_spread = fvals[-1] - fvals[0]
            if f_spread < 1e-12:
                break
                
            # Calculate centroid of all points except the worst
            centroid = np.mean(simplex[:-1], axis=0)
            
            worst_x = simplex[-1]
            worst_f = fvals[-1]
            
            # 1. Reflection
            x_r = centroid + alpha * (centroid - worst_x)
            f_r = evaluate(x_r)
            
            if f_r < fvals[0]:
                # 2. Expansion
                x_e = centroid + gamma * (centroid - worst_x)
                f_e = evaluate(x_e)
                if f_e < f_r:
                    simplex[-1] = x_e
                    fvals[-1] = f_e
                else:
                    simplex[-1] = x_r
                    fvals[-1] = f_r
            elif f_r < fvals[-2]:
                # Reflection was good but not better than best
                simplex[-1] = x_r
                fvals[-1] = f_r
            else:
                # 3. Contraction
                # Outside contraction if worst_f < f_r, else inside
                if worst_f < f_r:
                    x_c = centroid + rho * (worst_x - centroid) # Outside contraction? No, standard is:
                    # Actually, if f_r >= f_w, we do contraction.
                    # If f_w < f_r, we do outside contraction?
                    # Standard Nelder-Mead:
                    # If f_r >= f_w:
                    #   If f_r < f_{n-1}: Inside contraction
                    #   Else: Outside contraction? No, usually Shrink.
                    # Let's stick to a robust variant:
                    # If f_r >= f_w:
                    #   Try contraction towards centroid.
                    #   If contraction fails, shrink.
                    
                    # Inside contraction
                    x_c = centroid + rho * (worst_x - centroid)
                    f_c = evaluate(x_c)
                    if f_c < worst_f:
                        simplex[-1] = x_c
                        fvals[-1] = f_c
                    else:
                        # 4. Shrink
                        for i in range(1, d + 1):
                            simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                            fvals[i] = evaluate(simplex[i])
                        
        return best_x_global, best_f_global
# EVOLVE-BLOCK-END
