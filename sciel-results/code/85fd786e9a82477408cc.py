"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """Global Screening + Nelder-Mead Simplex optimizer."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 295.0  # Safety margin: 300s limit
        
        dim = self.dim
        budget = self.budget
        
        # 1. Coarse Global Screening
        # BBOB f7 (Step Ellipsoidal) has a step. We need to find the basin.
        # A coarse grid of 3^5 = 243 points is a good balance.
        # If budget allows, we evaluate this grid.
        grid_levels = 3
        grid_points = grid_levels ** dim
        
        calls_made = 0
        best_x = None
        best_f = np.inf
        
        if budget >= grid_points + 10:
            coords = np.linspace(-5, 5, grid_levels)
            grid = np.array(np.meshgrid(*([coords]*dim)))
            grid = grid.reshape(dim, -1).T
            
            # Shuffle to randomize order (though symmetric, good practice)
            rng = np.random.default_rng(42)
            perm = rng.permutation(len(grid))
            grid = grid[perm]
            
            for x in grid:
                if calls_made >= budget or time.time() > deadline:
                    break
                f_val = func(x)
                calls_made += 1
                if f_val < best_f:
                    best_f = f_val
                    best_x = x.copy()
        else:
            # Fallback: Random initialization
            rng = np.random.default_rng(42)
            pop_size = min(budget // 2, 20)
            population = rng.uniform(-5, 5, (pop_size, dim))
            for x in population:
                if calls_made >= budget or time.time() > deadline:
                    break
                f_val = func(x)
                calls_made += 1
                if f_val < best_f:
                    best_f = f_val
                    best_x = x.copy()

        if best_x is None:
            # Should not happen, but safety
            best_x = np.zeros(dim)
            best_f = func(best_x)
            calls_made += 1

        # 2. Local Refinement using Nelder-Mead Simplex
        # This is the core of the new strategy.
        # Nelder-Mead is a derivative-free method that works well on smooth
        # landscapes and can handle some non-smoothness (like the step in f7)
        # if initialized in the correct basin.
        
        # Initialize simplex around best_x
        # Simplex has dim+1 vertices
        simplex = np.zeros((dim + 1, dim))
        for i in range(dim + 1):
            simplex[i] = best_x.copy()
        
        # Set initial step size for simplex expansion
        # Start with a moderate step to allow movement within the basin
        step_size = 0.5
        for i in range(dim):
            simplex[i + 1][i] += step_size
        
        # Evaluate initial simplex
        f_simplex = np.zeros(dim + 1)
        for i in range(dim + 1):
            if calls_made >= budget or time.time() > deadline:
                break
            f_simplex[i] = func(simplex[i])
            calls_made += 1
            if f_simplex[i] < best_f:
                best_f = f_simplex[i]
                best_x = simplex[i].copy()
        
        # Nelder-Mead parameters
        alpha = 1.0  # Reflection
        gamma = 2.0  # Expansion
        rho = 0.5    # Contraction
        sigma = 0.5  # Shrinkage
        
        # Main Nelder-Mead loop
        max_iter = 2000
        for iter in range(max_iter):
            if calls_made >= budget or time.time() > deadline:
                break
            
            # Sort simplex by function value
            order = np.argsort(f_simplex)
            simplex = simplex[order]
            f_simplex = f_simplex[order]
            
            # Check for convergence
            if np.max(f_simplex) - np.min(f_simplex) < 1e-8:
                break
            
            # Centroid of all points except the worst
            centroid = np.mean(simplex[:-1], axis=0)
            worst = simplex[-1]
            
            # Reflection
            x_ref = centroid + alpha * (centroid - worst)
            x_ref = np.clip(x_ref, -5, 5)
            f_ref = func(x_ref)
            calls_made += 1
            if f_ref < best_f:
                best_f = f_ref
                best_x = x_ref.copy()
            
            if f_ref < f_simplex[0]:
                # Expansion
                x_exp = centroid + gamma * (x_ref - centroid)
                x_exp = np.clip(x_exp, -5, 5)
                f_exp = func(x_exp)
                calls_made += 1
                if f_exp < best_f:
                    best_f = f_exp
                    best_x = x_exp.copy()
                if f_exp < f_ref:
                    simplex[-1] = x_exp
                    f_simplex[-1] = f_exp
                else:
                    simplex[-1] = x_ref
                    f_simplex[-1] = f_ref
            elif f_ref < f_simplex[-2]:
                # Simple reflection
                simplex[-1] = x_ref
                f_simplex[-1] = f_ref
            else:
                # Contraction
                if f_ref < f_simplex[-1]:
                    # Outside contraction
                    x_con = centroid + rho * (x_ref - centroid)
                    f_con = f_ref
                else:
                    # Inside contraction
                    x_con = centroid - rho * (worst - centroid)
                    f_con = f_simplex[-1]
                
                x_con = np.clip(x_con, -5, 5)
                f_con = func(x_con)
                calls_made += 1
                if f_con < best_f:
                    best_f = f_con
                    best_x = x_con.copy()
                
                if f_con < min(f_ref, f_simplex[-1]):
                    simplex[-1] = x_con
                    f_simplex[-1] = f_con
                else:
                    # Shrinkage
                    for i in range(1, dim + 1):
                        simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                        simplex[i] = np.clip(simplex[i], -5, 5)
                        if calls_made >= budget or time.time() > deadline:
                            break
                        f_simplex[i] = func(simplex[i])
                        calls_made += 1
                        if f_simplex[i] < best_f:
                            best_f = f_simplex[i]
                            best_x = simplex[i].copy()
                    
                    # Re-sort after shrinkage
                    order = np.argsort(f_simplex)
                    simplex = simplex[order]
                    f_simplex = f_simplex[order]
            
            # Update best
            if f_simplex[0] < best_f:
                best_f = f_simplex[0]
                best_x = simplex[0].copy()
        
        return best_x, best_f
# EVOLVE-BLOCK-END
