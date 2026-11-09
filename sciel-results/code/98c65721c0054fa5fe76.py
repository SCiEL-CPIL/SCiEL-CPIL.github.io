"""Adaptive Nelder-Mead optimizer with greedy restarts for BBOB-style functions."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """
    Optimizer using Nelder-Mead Simplex method.
    
    Strategy:
    1. Start with a random simplex.
    2. Run Nelder-Mead to minimize.
    3. If the simplex converges (volume shrinks) or stagnates (no improvement for N iterations),
       restart from the best point found so far with a new, slightly perturbed simplex.
    4. This balances exploration (new starts) and exploitation (deep local search).
    
    Why this is different:
    - Current program uses fixed-step coordinate descent with rigid 30x10 partition.
    - This program uses adaptive step sizes (simplex geometry) and dynamic restarts based on progress.
    - Nelder-Mead is derivative-free and works well on smooth functions like Sphere and Rosenbrock.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        time_limit = 320.0  # Safety margin before 330s kill
        
        dim = self.dim
        budget = self.budget
        
        best_x_global = None
        best_f_global = np.inf
        
        # Nelder-Mead parameters
        alpha = 1.0   # Reflection
        gamma = 2.0   # Expansion
        rho = 0.5     # Contraction
        sigma = 0.5   # Shrinkage
        tol = 1e-8    # Convergence tolerance
        
        # Restart parameters
        max_iter_per_restart = 50  # Max iterations of NM before forcing restart
        stagnation_threshold = 5   # If no improvement for this many iterations, restart
        
        # Initial simplex size
        initial_sigma = 1.0
        
        while budget > 0:
            if time.time() - start_time > time_limit:
                break
            
            # 1. Initialize Simplex
            # Start from best point if we have one, otherwise random
            if best_x_global is None:
                center = np.random.uniform(-4, 4, dim)
            else:
                # Perturb the best point slightly to escape local minima
                center = best_x_global + np.random.normal(0, 0.5, dim)
                center = np.clip(center, -5, 5)
            
            # Create simplex vertices
            v0 = center.copy()
            f0 = func(v0)
            budget -= 1
            if f0 < best_f_global:
                best_f_global = f0
                best_x_global = v0.copy()
            
            simplex = [v0]
            fvals = [f0]
            
            for i in range(1, dim + 1):
                vi = v0.copy()
                # Perturb dimension i
                vi[i-1] += initial_sigma
                vi = np.clip(vi, -5, 5)
                fi = func(vi)
                budget -= 1
                if fi < best_f_global:
                    best_f_global = fi
                    best_x_global = vi.copy()
                simplex.append(vi)
                fvals.append(fi)
            
            # 2. Nelder-Mead Iterations
            iter_count = 0
            stagnation_count = 0
            best_f_in_restart = min(fvals)
            
            while iter_count < max_iter_per_restart and budget > 0:
                if time.time() - start_time > time_limit:
                    break
                
                # Sort simplex by function value
                idx = np.argsort(fvals)
                simplex = [simplex[i] for i in idx]
                fvals = [fvals[i] for i in idx]
                
                # Check convergence
                if abs(fvals[0] - fvals[-1]) < tol:
                    break
                
                # Check stagnation
                if fvals[0] < best_f_in_restart - 1e-6:
                    best_f_in_restart = fvals[0]
                    stagnation_count = 0
                else:
                    stagnation_count += 1
                    if stagnation_count > stagnation_threshold:
                        break
                
                # Centroid of all points except the worst
                centroid = np.mean(simplex[:-1], axis=0)
                
                # Reflection
                xr = centroid + alpha * (centroid - simplex[-1])
                xr = np.clip(xr, -5, 5)
                fr = func(xr)
                budget -= 1
                if fr < best_f_global:
                    best_f_global = fr
                    best_x_global = xr.copy()
                
                if fr < fvals[0]:
                    # Expansion
                    xe = centroid + gamma * (centroid - simplex[-1])
                    xe = np.clip(xe, -5, 5)
                    fe = func(xe)
                    budget -= 1
                    if fe < best_f_global:
                        best_f_global = fe
                        best_x_global = xe.copy()
                    
                    if fe < fr:
                        simplex[-1] = xe
                        fvals[-1] = fe
                    else:
                        simplex[-1] = xr
                        fvals[-1] = fr
                elif fr < fvals[-2]:
                    # Accept reflection
                    simplex[-1] = xr
                    fvals[-1] = fr
                else:
                    # Contraction
                    # Outside contraction if worst is worse than reflected
                    if fr < fvals[-1]:
                        xc = centroid + rho * (xr - centroid)
                    else:
                        xc = centroid - rho * (simplex[-1] - centroid)
                    
                    xc = np.clip(xc, -5, 5)
                    fc = func(xc)
                    budget -= 1
                    if fc < best_f_global:
                        best_f_global = fc
                        best_x_global = xc.copy()
                    
                    if fc < min(fr, fvals[-1]):
                        simplex[-1] = xc
                        fvals[-1] = fc
                    else:
                        # Shrink
                        for i in range(1, dim + 1):
                            simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                            simplex[i] = np.clip(simplex[i], -5, 5)
                            fvals[i] = func(simplex[i])
                            budget -= 1
                            if fvals[i] < best_f_global:
                                best_f_global = fvals[i]
                                best_x_global = simplex[i].copy()
                        break # Shrinkage usually indicates convergence or bad region
                
                iter_count += 1
            
            # If we exited the while loop due to budget or time, stop
            if budget <= 0 or time.time() - start_time > time_limit:
                break

            # If we exited due to convergence or stagnation, loop back to restart
            # The loop condition will check budget again

        return best_x_global, best_f_global
# EVOLVE-BLOCK-END
