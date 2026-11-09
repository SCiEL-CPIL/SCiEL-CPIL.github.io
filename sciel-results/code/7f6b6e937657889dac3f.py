"""Nelder-Mead Simplex for Schaffers F7."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290  # Safety margin for 300s limit
        
        d = self.dim
        budget = self.budget
        
        # Nelder-Mead Parameters
        # For d=5, a simplex has d+1 = 6 vertices.
        # Initial step size: 0.05 * range (range is 10, so 0.5)
        # However, for ill-conditioned functions, a slightly larger initial step might help escape flat regions,
        # but NM is sensitive. Let's start with 0.1 * range = 1.0? 
        # Standard NM often uses 0.05 or 0.1. Let's use 0.1 * 10 = 1.0 for initial spread.
        # Wait, if the function is very ill-conditioned, a large step might jump out of the basin.
        # Let's try 0.05 * 10 = 0.5 for initial simplex size.
        
        alpha = 1.0  # Reflection
        gamma = 2.0  # Expansion
        rho = 0.5    # Contraction
        sigma = 0.5  # Shrink
        
        # Initialize simplex at center
        x_center = np.zeros(d)
        f_center = func(x_center)
        calls = 1
        best_f = f_center
        best_x = x_center.copy()
        
        # Create initial simplex
        # Vertex 0 is the center.
        # Vertices 1..d are center + step * e_i
        step = 0.5 
        simplex = [x_center.copy()]
        for i in range(d):
            x_i = x_center.copy()
            x_i[i] += step
            simplex.append(x_i)
        
        f_simplex = [f_center]
        for i in range(1, d + 1):
            f_i = func(simplex[i])
            calls += 1
            f_simplex.append(f_i)
            if f_i < best_f:
                best_f = f_i
                best_x = simplex[i].copy()
                
        # Sort simplex by fitness
        order = np.argsort(f_simplex)
        simplex = [simplex[i] for i in order]
        f_simplex = [f_simplex[i] for i in order]
        
        # Main Loop
        while calls < budget:
            if time.time() > deadline:
                break
            
            # Current worst and best
            x_worst = simplex[-1]
            f_worst = f_simplex[-1]
            x_best = simplex[0]
            f_best = f_simplex[0]
            
            # Check for convergence
            # If the difference between best and worst is very small, we might be stuck or converged.
            # Also check if the simplex volume is small.
            if f_worst - f_best < 1e-10:
                # Try to restart with a larger step if we have budget
                # Or just break if we are very small
                pass
            
            # Calculate centroid of all but the worst point
            x_centroid = np.mean(simplex[:-1], axis=0)
            
            # Reflection
            x_r = x_centroid + alpha * (x_centroid - x_worst)
            x_r = np.clip(x_r, -5, 5)
            f_r = func(x_r)
            calls += 1
            
            if f_r < f_best:
                # Expansion
                x_e = x_centroid + gamma * (x_r - x_centroid)
                x_e = np.clip(x_e, -5, 5)
                f_e = func(x_e)
                calls += 1
                
                if f_e < f_r:
                    new_x = x_e
                    new_f = f_e
                else:
                    new_x = x_r
                    new_f = f_r
                    
                simplex[-1] = new_x
                f_simplex[-1] = new_f
                
            elif f_r < f_worst:
                # Accept reflection
                simplex[-1] = x_r
                f_simplex[-1] = f_r
                
            else:
                # Contraction
                # If f_r > f_worst, contract towards the worst point (outside) or centroid (inside)?
                # Standard NM: If f_r > f_worst, it's a "failure".
                # If f_r < f_worst but > f_best, it's a "success" but we accepted it above.
                # Wait, the logic above:
                # if f_r < f_best: try expansion.
                # elif f_r < f_worst: accept reflection.
                # else: f_r >= f_worst. This means reflection is worse than the current worst.
                # In this case, we should contract.
                
                # Inside contraction (towards centroid)
                x_c = x_centroid + rho * (x_worst - x_centroid)
                x_c = np.clip(x_c, -5, 5)
                f_c = func(x_c)
                calls += 1
                
                if f_c < f_worst:
                    simplex[-1] = x_c
                    f_simplex[-1] = f_c
                else:
                    # Shrink
                    # Move all points except the best towards the best
                    for i in range(1, d + 1):
                        simplex[i] = x_best + sigma * (simplex[i] - x_best)
                        simplex[i] = np.clip(simplex[i], -5, 5)
                        f_simplex[i] = func(simplex[i])
                        calls += 1
                        if calls >= budget:
                            break
                    
                    if calls >= budget:
                        break
            
            # Update best
            for i in range(d + 1):
                if f_simplex[i] < best_f:
                    best_f = f_simplex[i]
                    best_x = simplex[i].copy()
                    
            # Re-sort simplex
            order = np.argsort(f_simplex)
            simplex = [simplex[i] for i in order]
            f_simplex = [f_simplex[i] for i in order]
            
            # Optional: Restart if simplex is too small
            # Calculate diameter
            diam = np.linalg.norm(simplex[0] - simplex[-1])
            if diam < 1e-6:
                # If stuck, perturb the worst point with a random step
                # This acts as a local restart
                if calls < budget:
                    # Generate a random point near the best
                    perturbation = 0.5 * np.random.randn(d)
                    x_new = best_x + perturbation
                    x_new = np.clip(x_new, -5, 5)
                    f_new = func(x_new)
                    calls += 1
                    if f_new < best_f:
                        best_f = f_new
                        best_x = x_new.copy()
                    # Replace the worst point in the simplex
                    simplex[-1] = x_new
                    f_simplex[-1] = f_new
                    order = np.argsort(f_simplex)
                    simplex = [simplex[i] for i in order]
                    f_simplex = [f_simplex[i] for i in order]
                    
        return best_x, best_f
# EVOLVE-BLOCK-END
