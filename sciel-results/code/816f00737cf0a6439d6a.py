"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """Nelder-Mead optimizer with adaptive restarts for BBOB f1 (Sphere)."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        deadline = time.time() + 280  # Safety margin before 300s limit
        dim = self.dim
        budget = self.budget
        
        # State
        best_x = np.zeros(dim)
        best_f = float('inf')
        
        # Nelder-Mead parameters
        alpha = 1.0   # Reflection
        gamma = 2.0   # Expansion
        rho = 0.5     # Contraction
        sigma = 0.5   # Shrinkage
        
        def evaluate(x):
            nonlocal best_x, best_f
            fx = func(x)
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            return fx

        # Initialize simplex
        # Start at 0,0,0,0,0
        x0 = np.zeros(dim)
        f0 = evaluate(x0)
        calls_made = 1
        
        # Initial simplex size. Since minimizer is in [-4,4], 
        # a step of 1.0 is reasonable.
        delta = 1.0
        
        simplex = [x0]
        fvals = [f0]
        
        for i in range(dim):
            xi = x0.copy()
            xi[i] += delta
            simplex.append(xi)
            fvals.append(evaluate(xi))
            calls_made += 1
            
            if calls_made >= budget:
                break
            if time.time() > deadline:
                break

        # Main loop
        while calls_made < budget:
            if time.time() > deadline:
                break
                
            # Sort simplex by function value
            indices = np.argsort(fvals)
            simplex = [simplex[i] for i in indices]
            fvals = [fvals[i] for i in indices]
            
            x_worst = simplex[-1]
            f_worst = fvals[-1]
            
            # Calculate centroid of all but worst point
            x_centroid = np.mean(simplex[:-1], axis=0)
            
            # 1. Reflection
            x_ref = x_centroid + alpha * (x_centroid - x_worst)
            f_ref = evaluate(x_ref)
            calls_made += 1
            
            if f_ref < fvals[0]:
                # 2. Expansion
                x_exp = x_centroid + gamma * (x_ref - x_centroid)
                f_exp = evaluate(x_exp)
                calls_made += 1
                
                if f_exp < f_ref:
                    simplex[-1] = x_exp
                    fvals[-1] = f_exp
                else:
                    simplex[-1] = x_ref
                    fvals[-1] = f_ref
            elif f_ref < fvals[-2]:
                # Accept reflection
                simplex[-1] = x_ref
                fvals[-1] = f_ref
            else:
                # 3. Contraction
                x_con = x_centroid + rho * (x_worst - x_centroid)
                f_con = evaluate(x_con)
                calls_made += 1
                
                if f_con < f_worst:
                    simplex[-1] = x_con
                    fvals[-1] = f_con
                else:
                    # 4. Shrinkage
                    for i in range(1, len(simplex)):
                        simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                        fvals[i] = evaluate(simplex[i])
                        calls_made += 1
                        
                        if calls_made >= budget:
                            break
                        if time.time() > deadline:
                            break
                    if calls_made >= budget or time.time() > deadline:
                        break

            # Check for convergence (simplex size very small)
            # If converged, we might break early to save time, 
            # but since we have a budget, we might just stop.
            # For Sphere, once converged, further calls don't help much.
            # Let's check if the range of fvals is tiny.
            if max(fvals) - min(fvals) < 1e-10:
                break

        return best_x, best_f
# EVOLVE-BLOCK-END
