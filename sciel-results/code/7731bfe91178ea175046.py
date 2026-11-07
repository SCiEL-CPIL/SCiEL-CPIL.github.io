"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Nelder-Mead simplex optimizer with multi-restart for Rastrigin rotated."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        rng = np.random.default_rng(42)
        calls_made = 0
        best_x = None
        best_f = float('inf')
        deadline = time.time() + 285

        def count_eval(x):
            nonlocal calls_made
            val = func(x)
            calls_made += 1
            return val

        # Number of restarts to attempt
        # Each restart uses a chunk of the budget
        # Nelder-Mead in 5D typically needs ~50-200 evals to converge locally
        # Rastrigin has many local minima, so restarts are key
        
        # Strategy: 
        # 1. First, do a coarse random search to find a few promising regions (10-20 evals)
        # 2. Then, run Nelder-Mead from the best few points found
        # 3. If time/budget remains, do another round with more points
        
        # Phase 1: Coarse random sampling
        n_coarse = 15
        coarse_x = []
        coarse_f = []
        for _ in range(n_coarse):
            if calls_made >= self.budget or time.time() > deadline:
                break
            x = rng.uniform(-5, 5, self.dim)
            fx = count_eval(x)
            coarse_x.append(x)
            coarse_f.append(fx)
        
        if not coarse_x:
            # Fallback if even coarse search fails
            x = rng.uniform(-5, 5, self.dim)
            fx = count_eval(x)
            return x, fx

        # Sort coarse results
        order = np.argsort(coarse_f)
        
        # Phase 2: Nelder-Mead from top candidates
        # We'll allocate remaining budget among top candidates
        # Let's pick top 3-5 candidates
        n_top = min(5, len(coarse_x))
        top_indices = order[:n_top]
        
        # Allocate budget: give more to better candidates
        # Simple approach: equal split of remaining budget
        remaining_budget = self.budget - calls_made
        if remaining_budget <= 0:
            # Return best from coarse
            best_idx = order[0]
            return coarse_x[best_idx], coarse_f[best_idx]
        
        budget_per_nelder = max(10, remaining_budget // n_top)
        
        for idx in top_indices:
            if calls_made >= self.budget or time.time() > deadline:
                break
            
            x0 = coarse_x[idx].copy()
            f0 = coarse_f[idx]
            
            # Nelder-Mead parameters
            alpha = 1.0  # reflection
            gamma = 2.0  # expansion
            rho = 0.5    # contraction
            sigma = 0.5  # shrink
            max_iter = budget_per_nelder
            
            # Initialize simplex
            simplex = [x0.copy()]
            simplex_f = [f0]
            
            # Add other points in simplex
            for i in range(self.dim):
                x_new = x0.copy()
                x_new[i] += 0.5 if x0[i] < 4.5 else -0.5
                x_new = np.clip(x_new, -5, 5)
                f_new = count_eval(x_new)
                simplex.append(x_new)
                simplex_f.append(f_new)
                
                if calls_made >= self.budget or time.time() > deadline:
                    break
            
            if len(simplex) < 2:
                continue
            
            # Nelder-Mead iterations
            for _ in range(max_iter):
                if calls_made >= self.budget or time.time() > deadline:
                    break
                
                # Sort simplex by function value
                order_simplex = np.argsort(simplex_f)
                simplex = [simplex[i] for i in order_simplex]
                simplex_f = [simplex_f[i] for i in order_simplex]
                
                # Centroid of all but worst point
                centroid = np.mean(simplex[:-1], axis=0)
                
                # Worst point
                x_worst = simplex[-1]
                f_worst = simplex_f[-1]
                
                # Reflection
                x_ref = centroid + alpha * (centroid - x_worst)
                x_ref = np.clip(x_ref, -5, 5)
                f_ref = count_eval(x_ref)
                
                if calls_made >= self.budget or time.time() > deadline:
                    break
                
                if f_ref < simplex_f[0]:
                    # Expansion
                    x_exp = centroid + gamma * (x_ref - centroid)
                    x_exp = np.clip(x_exp, -5, 5)
                    f_exp = count_eval(x_exp)
                    
                    if calls_made >= self.budget or time.time() > deadline:
                        break
                    
                    if f_exp < f_ref:
                        x_new, f_new = x_exp, f_exp
                    else:
                        x_new, f_new = x_ref, f_ref
                elif f_ref < f_worst:
                    x_new, f_new = x_ref, f_ref
                else:
                    # Contraction
                    x_con = centroid + rho * (x_worst - centroid)
                    x_con = np.clip(x_con, -5, 5)
                    f_con = count_eval(x_con)
                    
                    if calls_made >= self.budget or time.time() > deadline:
                        break
                    
                    if f_con < f_worst:
                        x_new, f_new = x_con, f_con
                    else:
                        # Shrink
                        x_new = x0 + sigma * (x_worst - x0)
                        x_new = np.clip(x_new, -5, 5)
                        f_new = count_eval(x_new)
                        
                        if calls_made >= self.budget or time.time() > deadline:
                            break
                
                # Replace worst point
                simplex[-1] = x_new
                simplex_f[-1] = f_new
                
                # Update best
                min_idx = np.argmin(simplex_f)
                if simplex_f[min_idx] < best_f:
                    best_f = simplex_f[min_idx]
                    best_x = simplex[min_idx].copy()
                    
                # Check convergence
                if len(simplex) >= 2:
                    f_range = max(simplex_f) - min(simplex_f)
                    if f_range < 1e-6:
                        break

        if best_x is None:
            # Fallback to best coarse point
            best_idx = order[0]
            best_x = coarse_x[best_idx]
            best_f = coarse_f[best_idx]

        return best_x, best_f
# EVOLVE-BLOCK-END
