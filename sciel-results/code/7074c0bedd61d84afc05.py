"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-start Nelder-Mead optimizer for sharp ridge landscapes."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280
        rng = np.random.default_rng(42)
        dim = self.dim
        budget = self.budget
        
        global_best_x = np.zeros(dim)
        global_best_fx = np.inf
        calls_used = 0
        
        # Helper to run Nelder-Mead
        def nelder_mead(x0, max_iter, budget_remaining, deadline):
            nonlocal calls_used, global_best_x, global_best_fx
            
            if budget_remaining <= 0 or time.time() >= deadline:
                return x0, func(x0) if calls_used < budget else global_best_fx

            # Initialize simplex
            simplex = [x0.copy()]
            f_simplex = [func(x0)]
            calls_used += 1
            if f_simplex[0] < global_best_fx:
                global_best_fx = f_simplex[0]
                global_best_x = x0.copy()
                
            # Perturbation step
            # Use a step size that covers a reasonable portion of the domain initially
            step = 1.0 
            
            for i in range(dim):
                if calls_used >= budget or time.time() >= deadline:
                    break
                x_pert = x0.copy()
                x_pert[i] += step
                x_pert = np.clip(x_pert, -5, 5)
                simplex.append(x_pert)
                f_simplex.append(func(x_pert))
                calls_used += 1
                if f_simplex[-1] < global_best_fx:
                    global_best_fx = f_simplex[-1]
                    global_best_x = x_pert.copy()
            
            if len(simplex) < dim + 1:
                return x0, f_simplex[0]

            # Nelder-Mead parameters
            alpha = 1.0
            gamma = 2.0
            rho = 0.5
            sigma_nm = 0.5
            
            for _ in range(max_iter):
                if calls_used >= budget or time.time() >= deadline:
                    break
                
                # Sort
                order = np.argsort(f_simplex)
                simplex = [simplex[i].copy() for i in order]
                f_simplex = [f_simplex[i] for i in order]
                
                # Centroid of best dim points
                centroid = np.mean(simplex[:-1], axis=0)
                
                # Reflect
                x_r = centroid + alpha * (centroid - simplex[-1])
                x_r = np.clip(x_r, -5, 5)
                
                if calls_used >= budget or time.time() >= deadline:
                    break
                f_r = func(x_r)
                calls_used += 1
                if f_r < global_best_fx:
                    global_best_fx = f_r
                    global_best_x = x_r.copy()
                    
                if f_r < f_simplex[0]:
                    # Expand
                    x_e = centroid + gamma * (x_r - centroid)
                    x_e = np.clip(x_e, -5, 5)
                    if calls_used >= budget or time.time() >= deadline:
                        break
                    f_e = func(x_e)
                    calls_used += 1
                    if f_e < global_best_fx:
                        global_best_fx = f_e
                        global_best_x = x_e.copy()
                    if f_e < f_r:
                        simplex[-1] = x_e
                        f_simplex[-1] = f_e
                    else:
                        simplex[-1] = x_r
                        f_simplex[-1] = f_r
                elif f_r < f_simplex[-2]:
                    simplex[-1] = x_r
                    f_simplex[-1] = f_r
                else:
                    # Contract
                    x_c = centroid + rho * (simplex[-1] - centroid)
                    x_c = np.clip(x_c, -5, 5)
                    if calls_used >= budget or time.time() >= deadline:
                        break
                    f_c = func(x_c)
                    calls_used += 1
                    if f_c < global_best_fx:
                        global_best_fx = f_c
                        global_best_x = x_c.copy()
                    if f_c < f_simplex[-1]:
                        simplex[-1] = x_c
                        f_simplex[-1] = f_c
                    else:
                        # Shrink
                        for i in range(1, dim + 1):
                            if calls_used >= budget or time.time() >= deadline:
                                break
                            simplex[i] = simplex[0] + sigma_nm * (simplex[i] - simplex[0])
                            simplex[i] = np.clip(simplex[i], -5, 5)
                            f_simplex[i] = func(simplex[i])
                            calls_used += 1
                            if f_simplex[i] < global_best_fx:
                                global_best_fx = f_simplex[i]
                                global_best_x = simplex[i].copy()
            
            return simplex[0].copy(), f_simplex[0]

        # Phase 1: Multi-start
        # We have 500 budget.
        # Let's do 5 starts with 50 iterations of NM each? 5 * (1 + 5 + 50) = 280 calls.
        # Remaining 220 calls for a final long NM run on the best point.
        
        n_starts = 5
        calls_per_start = 50
        
        for i in range(n_starts):
            if time.time() >= deadline or calls_used >= budget:
                break
            
            # Random start
            x0 = rng.uniform(-5, 5, dim)
            # Run NM
            nelder_mead(x0, calls_per_start, budget - calls_used, deadline)
        
        # Phase 2: Final Polish
        # Use remaining budget for a long NM run on the best point found
        if time.time() < deadline and calls_used < budget:
            remaining = budget - calls_used
            # Cap iterations to avoid infinite loops, but NM usually converges fast
            # Or just let it run until budget/time is out
            nelder_mead(global_best_x, 1000, remaining, deadline)

        return global_best_x, global_best_fx
# EVOLVE-BLOCK-END
