"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import scipy.stats.qmc as qmc


class Optimizer:
    """Derivative-free Nelder-Mead with Sobol global search."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        
        # Phase 1: Global Search using Sobol sequence for uniform coverage
        # Sobol points are low-discrepancy, good for identifying basins
        n_global = 30  # Use ~30 points for global search
        sampler = qmc.Sobol(d=dim, scramble=True, seed=42)
        # generate points in [0, 1]^dim
        global_points_unit = sampler.random(n_global)
        # scale to [-5, 5]^dim
        global_points = global_points_unit * 10.0 - 5.0
        
        best_x = None
        best_f = np.inf
        calls_made = 0
        
        for i in range(n_global):
            if calls_made >= budget:
                break
            x = np.clip(global_points[i], -5, 5)
            fx = func(x)
            calls_made += 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
                
        if best_x is None:
            return np.zeros(dim), np.inf

        # Phase 2: Nelder-Mead Local Optimization
        # We will run NM multiple times if it stagnates, restarting from the best known point
        # with a smaller initial step size.
        
        alpha = 1.0
        gamma = 2.0
        rho = 0.5
        sigma = 0.5
        
        current_step = 1.0
        max_restarts = 10
        restart_count = 0
        
        # We keep the global best_x and best_f updated throughout
        while calls_made < budget and restart_count <= max_restarts:
            # Initialize simplex around best_x
            simplex = np.zeros((dim + 1, dim))
            simplex[0] = best_x
            for i in range(1, dim + 1):
                simplex[i] = best_x.copy()
                simplex[i, i - 1] += current_step
                # Ensure we stay within bounds, though NM will clip
                simplex[i] = np.clip(simplex[i], -5, 5)
                
            simplex_f = np.zeros(dim + 1)
            for i in range(dim + 1):
                if calls_made >= budget:
                    simplex_f[i] = np.inf
                    continue
                simplex_f[i] = func(simplex[i])
                calls_made += 1
                if simplex_f[i] < best_f:
                    best_f = simplex_f[i]
                    best_x = simplex[i].copy()
                    
            no_improve = 0
            max_no_improve = 10  # Tighten stagnation detection
            
            while calls_made < budget:
                order = np.argsort(simplex_f)
                simplex = simplex[order]
                simplex_f = simplex_f[order]
                
                # Check for convergence
                if np.max(simplex_f) - np.min(simplex_f) < 1e-6:
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
                    # Restart
                    restart_count += 1
                    current_step *= 0.5
                    if current_step < 0.01:
                        break # Too small, stop
                    # Perturb best_x slightly to escape local minima if any, or just restart NM
                    # Since we are at a local min, restarting NM with smaller step is better
                    break
            
        return best_x, best_f
# EVOLVE-BLOCK-END
