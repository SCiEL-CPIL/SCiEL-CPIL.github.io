"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """Nelder-Mead optimizer for BBOB-style Sphere f1."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        total_budget = self.budget
        t0 = time.time()
        deadline = t0 + 280  # Safety margin before 300s limit

        best_x = None
        best_f = np.inf
        calls_made = 0

        def safe_call(x):
            nonlocal calls_made, best_f, best_x
            if calls_made >= total_budget:
                return np.inf
            fx = func(x)
            calls_made += 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            return fx

        # 1. Global Exploration Phase
        # Use a small budget for random sampling to find a good basin.
        # Since the domain is [-5, 5]^5 and minimizer is in [-4, 4]^5,
        # random sampling in [-4, 4]^5 is effective.
        n_global = min(20, total_budget // 5)
        
        # Generate random points
        # We need to be careful not to exceed budget
        for _ in range(n_global):
            if time.time() > deadline or calls_made >= total_budget:
                break
            x = np.random.uniform(-4.0, 4.0, dim)
            safe_call(x)
            
        if best_x is None:
            # Fallback if no calls were made (should not happen)
            x0 = np.zeros(dim)
            safe_call(x0)
            if best_x is None:
                return None, np.inf

        # 2. Local Refinement Phase with Nelder-Mead
        # Nelder-Mead parameters
        alpha = 1.0  # Reflection
        gamma = 2.0  # Expansion
        rho = 0.5    # Contraction
        sigma = 0.5  # Shrinkage
        
        # Initialize simplex
        # Start from best_x found in global phase
        x_center = best_x.copy()
        f_center = best_f
        
        # Create simplex vertices
        # The initial simplex should be centered at x_center with a reasonable step size.
        # Since we don't know the scale, a step of 1.0 is reasonable for [-5,5].
        step = 1.0
        simplex = []
        simplex_f = []
        
        # Vertex 0 is the best point found
        simplex.append(x_center.copy())
        simplex_f.append(f_center)
        
        # Other vertices
        for i in range(dim):
            x_v = x_center.copy()
            x_v[i] += step
            simplex.append(x_v)
            f_v = safe_call(x_v)
            simplex_f.append(f_v)
            
        # If we ran out of budget during simplex init, return best
        if calls_made >= total_budget:
            return best_x, best_f

        # Nelder-Mead Loop
        # Max iterations to prevent infinite loops if convergence is slow
        max_iter = 200
        
        for _ in range(max_iter):
            if time.time() > deadline or calls_made >= total_budget:
                break
                
            # Sort simplex by function value
            idx = np.argsort(simplex_f)
            simplex = [simplex[i] for i in idx]
            simplex_f = [simplex_f[i] for i in idx]
            
            # Check convergence
            # If the difference between best and worst is small, stop
            if abs(simplex_f[-1] - simplex_f[0]) < 1e-8:
                break
                
            # Centroid of the worst point excluded
            centroid = np.mean(simplex[:-1], axis=0)
            x_worst = simplex[-1]
            f_worst = simplex_f[-1]
            
            # 1. Reflection
            x_ref = centroid + alpha * (centroid - x_worst)
            f_ref = safe_call(x_ref)
            
            if calls_made >= total_budget:
                break
                
            if f_ref < simplex_f[0]:
                # 2. Expansion
                x_exp = centroid + gamma * (centroid - x_worst)
                f_exp = safe_call(x_exp)
                
                if calls_made >= total_budget:
                    break
                    
                if f_exp < f_ref:
                    simplex[-1] = x_exp
                    simplex_f[-1] = f_exp
                else:
                    simplex[-1] = x_ref
                    simplex_f[-1] = f_ref
            elif f_ref < simplex_f[-2]:
                # Accept reflection
                simplex[-1] = x_ref
                simplex_f[-1] = f_ref
            else:
                # 3. Contraction
                # External contraction if f_ref < f_worst
                if f_ref < f_worst:
                    x_con = centroid + rho * (x_ref - centroid)
                else:
                    # Internal contraction
                    x_con = centroid - rho * (centroid - x_worst)
                    
                f_con = safe_call(x_con)
                
                if calls_made >= total_budget:
                    break
                    
                if f_con < min(f_ref, f_worst):
                    simplex[-1] = x_con
                    simplex_f[-1] = f_con
                else:
                    # 4. Shrinkage
                    # Shrink all points except the best towards the best
                    for i in range(1, dim + 1):
                        simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                        simplex_f[i] = safe_call(simplex[i])
                        
                        if calls_made >= total_budget:
                            break
                    if calls_made >= total_budget:
                        break

        return best_x, best_f

# EVOLVE-BLOCK-END
