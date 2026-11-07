"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-start Nelder-Mead optimizer."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290  # 10s safety margin before 300s limit
        
        best_x = np.zeros(self.dim)
        best_f = float('inf')
        
        # Use fixed seed for reproducibility
        rng = np.random.default_rng(42)
        
        # 10 independent starts
        # Schwefel is multimodal, so spreading starts is crucial
        for i in range(10):
            # Generate initial point
            # Mix of random and slightly structured starts
            x0 = rng.uniform(-4.5, 4.5, self.dim)
            
            # Nelder-Mead setup
            # Initial simplex
            simplex = [x0.copy()]
            for j in range(self.dim):
                xi = x0.copy()
                # Small perturbation in dimension j
                # Scale perturbation based on box size
                xi[j] += 0.5 if xi[j] < 4.0 else -0.5
                simplex.append(xi)
            
            # Evaluate initial simplex
            fvals = [func(np.clip(x, -5, 5)) for x in simplex]
            
            # Check if we already beat global best
            min_idx = np.argmin(fvals)
            if fvals[min_idx] < best_f:
                best_f = fvals[min_idx]
                best_x = simplex[min_idx].copy()
            
            # Nelder-Mead iterations
            # Allocate ~50 iterations per start (total 500 calls approx, 
            # but we stop early if time is up or budget is used)
            # Actually, we have a global budget of 500. 
            # 10 starts * (1 simplex + N iters)
            # Let's aim for ~45 iters per start to stay under 500 total calls
            # 10 * (1 + 45) = 460 calls.
            
            for _ in range(45):
                # Check time
                if time.time() > deadline:
                    return best_x, best_f
                
                # Sort simplex
                idx = np.argsort(fvals)
                simplex = [simplex[i] for i in idx]
                fvals = [fvals[i] for i in idx]
                
                # Update global best
                if fvals[0] < best_f:
                    best_f = fvals[0]
                    best_x = simplex[0].copy()
                    
                # Centroid of all but worst
                centroid = np.mean(simplex[:-1], axis=0)
                worst_x = simplex[-1]
                worst_f = fvals[-1]
                
                # Reflection
                alpha = 1.0
                x_ref = centroid + alpha * (centroid - worst_x)
                x_ref = np.clip(x_ref, -5, 5)
                f_ref = func(x_ref)
                
                if f_ref < fvals[0]:
                    # Expansion
                    gamma = 2.0
                    x_exp = centroid + gamma * (centroid - worst_x)
                    x_exp = np.clip(x_exp, -5, 5)
                    f_exp = func(x_exp)
                    
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
                    # Contraction
                    rho = 0.5
                    x_con = centroid + rho * (worst_x - centroid)
                    x_con = np.clip(x_con, -5, 5)
                    f_con = func(x_con)
                    
                    if f_con < worst_f:
                        simplex[-1] = x_con
                        fvals[-1] = f_con
                    else:
                        # Shrink
                        for i in range(1, len(simplex)):
                            simplex[i] = simplex[0] + 0.5 * (simplex[i] - simplex[0])
                            simplex[i] = np.clip(simplex[i], -5, 5)
                            fvals[i] = func(simplex[i])
                            
                        # Update global best after shrink
                        min_idx = np.argmin(fvals)
                        if fvals[min_idx] < best_f:
                            best_f = fvals[min_idx]
                            best_x = simplex[min_idx].copy()
                
                # Check for convergence (small simplex)
                if np.max(fvals) - np.min(fvals) < 1e-8:
                    break

        return best_x, best_f
# EVOLVE-BLOCK-END
