"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import qmc


class Optimizer:
    """Deterministic multi-start Nelder-Mead optimizer."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0  # Safety margin for 300s limit
        dim = self.dim
        budget = self.budget
        
        # 1. Initialize Sobol sequence for global exploration
        # We need a good number of initial points to cover [-5, 5]^dim
        # For dim=5, 100 points is a reasonable starting grid
        n_init = 100
        if budget < n_init:
            n_init = max(2, budget)
            
        sampler = qmc.Sobol(d=dim, scramble=True, seed=42)
        # Generate points in [0, 1]^dim
        sobol_points = sampler.random(n_init)
        
        # Map to [-5, 5]^dim
        x_init = sobol_points * 10.0 - 5.0
        
        # 2. Evaluate initial points and track best
        best_x = None
        best_f = np.inf
        candidates = []
        
        calls_made = 0
        for i in range(n_init):
            if calls_made >= budget:
                break
            if time.time() > deadline:
                break
                
            x = x_init[i]
            fx = func(x)
            calls_made += 1
            
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            
            # Keep top candidates for local refinement
            # We want to start Nelder-Mead from the best few points
            candidates.append((fx, x))
        
        if calls_made >= budget or time.time() > deadline:
            return best_x, best_f

        # Sort candidates by fitness
        candidates.sort(key=lambda t: t[0])
        
        # 3. Refine the best candidate using Nelder-Mead
        # We will spend the remaining budget on this single best basin.
        # If time permits, we might try the second best, but NM is fast.
        
        x0 = candidates[0][1]
        f0 = candidates[0][0]
        
        # Initialize simplex
        # Standard NM initialization: x0, and x0 + delta * e_i
        delta = 0.1
        simplex = [x0]
        for i in range(dim):
            xi = x0.copy()
            xi[i] += delta
            simplex.append(xi)
        
        simplex = np.array(simplex)
        f_simplex = np.array([f0])
        
        # Evaluate rest of initial simplex
        for i in range(1, dim + 1):
            if calls_made >= budget:
                break
            if time.time() > deadline:
                break
            fx = func(simplex[i])
            calls_made += 1
            f_simplex = np.append(f_simplex, fx)
        
        # Nelder-Mead Loop
        alpha = 1.0   # reflection
        gamma = 2.0   # expansion
        rho = 0.5     # contraction
        sigma = 0.5   # shrink
        
        while calls_made < budget:
            if time.time() > deadline:
                break
            
            # Sort simplex by fitness
            order = np.argsort(f_simplex)
            simplex = simplex[order]
            f_simplex = f_simplex[order]
            
            # If converged, stop
            if np.max(f_simplex) - np.min(f_simplex) < 1e-10:
                break
            
            x_best = simplex[0]
            f_best = f_simplex[0]
            x_worst = simplex[-1]
            f_worst = f_simplex[-1]
            
            # Centroid of all but worst
            x_centroid = np.mean(simplex[:-1], axis=0)
            
            # Reflection
            x_r = x_centroid + alpha * (x_centroid - x_worst)
            x_r = np.clip(x_r, -5, 5)
            
            if calls_made >= budget or time.time() > deadline:
                break
            f_r = func(x_r)
            calls_made += 1
            
            if f_r < f_simplex[0]:
                # Expansion
                x_e = x_centroid + gamma * (x_r - x_centroid)
                x_e = np.clip(x_e, -5, 5)
                
                if calls_made >= budget or time.time() > deadline:
                    break
                f_e = func(x_e)
                calls_made += 1
                
                if f_e < f_r:
                    simplex[-1] = x_e
                    f_simplex[-1] = f_e
                else:
                    simplex[-1] = x_r
                    f_simplex[-1] = f_r
            elif f_r < f_simplex[-2]:
                # Accept reflection
                simplex[-1] = x_r
                f_simplex[-1] = f_r
            else:
                # Contraction
                if f_r < f_worst:
                    # Outside contraction
                    x_c = x_centroid + rho * (x_r - x_centroid)
                else:
                    # Inside contraction
                    x_c = x_centroid - rho * (x_r - x_centroid)
                
                x_c = np.clip(x_c, -5, 5)
                
                if calls_made >= budget or time.time() > deadline:
                    break
                f_c = func(x_c)
                calls_made += 1
                
                if f_c <= min(f_r, f_worst):
                    simplex[-1] = x_c
                    f_simplex[-1] = f_c
                else:
                    # Shrink
                    for i in range(1, dim + 1):
                        simplex[i] = x_best + sigma * (simplex[i] - x_best)
                        
                        if calls_made >= budget or time.time() > deadline:
                            break
                        f_simplex[i] = func(simplex[i])
                        calls_made += 1
                    
                    # If we broke out of the shrink loop early, we need to handle state
                    # But since we break the main loop immediately after, it's fine.
                    # However, if we didn't break, we continue.
                    if time.time() > deadline or calls_made >= budget:
                        break
            
            # Update global best
            current_best_in_simplex = np.min(f_simplex)
            if current_best_in_simplex < best_f:
                best_f = current_best_in_simplex
                best_idx = np.argmin(f_simplex)
                best_x = simplex[best_idx].copy()
            
            # Check for stagnation or small simplex
            if np.linalg.norm(simplex[-1] - simplex[0]) < 1e-8:
                break

        return best_x, best_f
# EVOLVE-BLOCK-END
