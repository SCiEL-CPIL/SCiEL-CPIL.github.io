"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import qmc

class Optimizer:
    """Nelder-Mead simplex with robust global search and adaptive restarts."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        rng = np.random.default_rng(42)
        
        # Phase 1: Global Search using Latin Hypercube (Quasi-Monte Carlo)
        # This provides better space-filling properties than uniform random sampling.
        n_global = 100
        sampler = qmc.LatinHypercube(d=dim, seed=42)
        # Scale to [-5, 5]
        global_points = sampler.random(n=n_global) * 10 - 5
        
        # Add some structured points (corners, center) to ensure coverage
        structured = np.array([
            [0.0]*dim,
            [5.0]*dim,
            [-5.0]*dim,
            [2.5]*dim,
            [-2.5]*dim
        ])
        
        all_points = np.vstack([global_points, structured])
        
        best_x = None
        best_f = np.inf
        calls_made = 0
        
        # Evaluate global points
        for x in all_points:
            if calls_made >= budget:
                break
            x = np.clip(x, -5, 5)
            fx = func(x)
            calls_made += 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
                
        if best_x is None:
            return np.zeros(dim), np.inf

        # Phase 2: Nelder-Mead with Restarts
        # Standard parameters
        alpha = 1.0
        gamma = 2.0
        rho = 0.5
        sigma = 0.5
        
        # Initial step size for simplex
        # Heuristic: start with a moderate step, adapt if stuck
        current_step = 1.0 
        max_restarts = 20
        restart_count = 0
        
        while calls_made < budget and restart_count <= max_restarts:
            # Initialize simplex around best_x
            simplex = np.zeros((dim + 1, dim))
            simplex[0] = best_x
            for i in range(1, dim + 1):
                simplex[i] = best_x.copy()
                simplex[i, i - 1] += current_step
                # Ensure points stay within bounds or clip
                simplex[i] = np.clip(simplex[i], -5, 5)
            
            simplex_f = np.zeros(dim + 1)
            for i in range(dim + 1):
                if calls_made >= budget:
                    simplex_f[i] = np.inf
                else:
                    simplex_f[i] = func(simplex[i])
                    calls_made += 1
                    if simplex_f[i] < best_f:
                        best_f = simplex_f[i]
                        best_x = simplex[i].copy()
                        
            no_improve = 0
            max_no_improve = 20 # Allow more iterations before restarting
            
            while calls_made < budget:
                # Order simplex by function value
                order = np.argsort(simplex_f)
                simplex = simplex[order]
                simplex_f = simplex_f[order]
                
                # Convergence check
                if np.max(simplex_f) - np.min(simplex_f) < 1e-10:
                    break
                    
                centroid = simplex[:-1].mean(axis=0)
                
                # Reflection
                if calls_made >= budget: break
                x_r = centroid + alpha * (centroid - simplex[-1])
                x_r = np.clip(x_r, -5, 5)
                f_r = func(x_r)
                calls_made += 1
                if f_r < best_f:
                    best_f = f_r
                    best_x = x_r.copy()
                    
                if f_r < simplex_f[0]:
                    # Expansion
                    if calls_made >= budget: break
                    x_e = centroid + gamma * (centroid - simplex[-1])
                    x_e = np.clip(x_e, -5, 5)
                    f_e = func(x_e)
                    calls_made += 1
                    if f_e < best_f:
                        best_f = f_e
                        best_x = x_e.copy()
                    
                    if f_e < f_r:
                        simplex[-1] = x_e
                        simplex_f[-1] = f_e
                    else:
                        simplex[-1] = x_r
                        simplex_f[-1] = f_r
                    no_improve = 0
                elif f_r < simplex_f[-2]:
                    # Accept reflection
                    simplex[-1] = x_r
                    simplex_f[-1] = f_r
                    no_improve = 0
                else:
                    # Contraction
                    if calls_made >= budget: break
                    x_c = centroid + rho * (simplex[-1] - centroid)
                    x_c = np.clip(x_c, -5, 5)
                    f_c = func(x_c)
                    calls_made += 1
                    if f_c < best_f:
                        best_f = f_c
                        best_x = x_c.copy()
                    
                    if f_c < f_r:
                        simplex[-1] = x_c
                        simplex_f[-1] = f_c
                        no_improve = 0
                    else:
                        # Shrink
                        no_improve += 1
                        for i in range(1, dim + 1):
                            if calls_made >= budget: break
                            simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                            simplex[i] = np.clip(simplex[i], -5, 5)
                            simplex_f[i] = func(simplex[i])
                            calls_made += 1
                            if simplex_f[i] < best_f:
                                best_f = simplex_f[i]
                                best_x = simplex[i].copy()
                                
                if no_improve > max_no_improve:
                    # Restart: Perturb best point and reduce step size
                    restart_count += 1
                    current_step = max(0.05, current_step * 0.7)
                    # Perturb to escape local minima
                    perturbation = rng.normal(0, current_step, size=dim)
                    best_x = np.clip(best_x + perturbation, -5, 5)
                    break # Break inner while, go to outer while
            
        return best_x, best_f
# EVOLVE-BLOCK-END
