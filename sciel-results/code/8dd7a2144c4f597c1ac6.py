"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import math


class Optimizer:
    """
    Nelder-Mead simplex optimizer with multi-start and restarts for Rastrigin rotated (f15).
    
    Strategy:
    1. Use Nelder-Mead for fast local convergence in smooth regions between Rastrigin minima.
    2. Multi-start from diverse points to explore different basins.
    3. Restart when simplex volume collapses or stagnation is detected.
    4. Combine with periodic jumps to explore adjacent basins.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290  # safety margin before 330s kill
        rng = np.random.default_rng(42)
        dim = self.dim
        budget = self.budget

        best_x = np.zeros(dim)
        best_f = np.inf
        calls_used = 0

        def evaluate(x):
            nonlocal calls_used, best_x, best_f
            x = np.asarray(x, dtype=np.float64)
            x = np.clip(x, -5, 5)
            fx = float(func(x))
            calls_used += 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            return fx

        def time_ok():
            return time.time() < deadline and calls_used < budget

        def nelder_mead(x0, max_evals, initial_alpha=0.5, beta=0.5, gamma=2.0, rho=1.0):
            """
            Nelder-Mead simplex method.
            Returns best_x, best_f found during this run.
            """
            nonlocal calls_used, best_x, best_f
            
            dim = self.dim
            # Initialize simplex
            simplex = np.zeros((dim + 1, dim))
            simplex[0] = x0
            fvals = np.zeros(dim + 1)
            fvals[0] = evaluate(x0)
            
            # Initialize other vertices with small offsets
            step = initial_alpha
            for i in range(1, dim + 1):
                v = x0.copy()
                v[i - 1] += step
                simplex[i] = v
                fvals[i] = evaluate(v)
            
            local_best_x = x0.copy()
            local_best_f = fvals[0]
            
            evals_in_run = 1 + dim  # initial evaluations
            
            while evals_in_run < max_evals and time_ok():
                # Sort simplex by function value
                order = np.argsort(fvals)
                simplex = simplex[order]
                fvals = fvals[order]
                
                # Check for convergence (simplex too small)
                centroid = simplex[:-1].mean(axis=0)
                spread = np.max(np.abs(simplex - centroid))
                if spread < 1e-6:
                    break
                
                # Check for stagnation
                if fvals[-1] - fvals[0] < 1e-10:
                    break
                
                # Compute centroid of all but worst point
                centroid = simplex[:-1].mean(axis=0)
                
                # Reflection
                xr = centroid + rho * (centroid - simplex[-1])
                xr = np.clip(xr, -5, 5)
                fr = evaluate(xr)
                evals_in_run += 1
                
                if fr < fvals[0]:
                    # Expansion
                    xe = centroid + gamma * (centroid - simplex[-1])
                    xe = np.clip(xe, -5, 5)
                    fe = evaluate(xe)
                    evals_in_run += 1
                    if fe < fr:
                        simplex[-1] = xe
                        fvals[-1] = fe
                    else:
                        simplex[-1] = xr
                        fvals[-1] = fr
                elif fr < fvals[-2]:
                    simplex[-1] = xr
                    fvals[-1] = fr
                else:
                    # Contraction
                    xc = centroid + beta * (simplex[-1] - centroid)
                    xc = np.clip(xc, -5, 5)
                    fc = evaluate(xc)
                    evals_in_run += 1
                    if fc < fvals[-1]:
                        simplex[-1] = xc
                        fvals[-1] = fc
                    else:
                        # Shrink
                        for i in range(1, dim + 1):
                            simplex[i] = simplex[0] + 0.5 * (simplex[i] - simplex[0])
                            fvals[i] = evaluate(simplex[i])
                            evals_in_run += 1
                        if evals_in_run >= max_evals:
                            break
                
                # Update local best
                min_idx = np.argmin(fvals)
                if fvals[min_idx] < local_best_f:
                    local_best_f = fvals[min_idx]
                    local_best_x = simplex[min_idx].copy()
            
            return local_best_x, local_best_f

        # Phase 1: Initial random sampling to find promising basins
        n_random = 15
        for _ in range(n_random):
            if not time_ok():
                break
            x = rng.uniform(-4.5, 4.5, dim)
            evaluate(x)
        
        if not time_ok():
            return (best_x, best_f)

        # Phase 2: Multi-start Nelder-Mead with restarts
        num_starts = 8
        calls_per_start = (budget - calls_used) // num_starts
        remaining = budget - calls_used
        
        for start_idx in range(num_starts):
            if not time_ok():
                break
            
            # Choose starting point: mix of random and near-best
            if start_idx < 3:
                # First few starts: random
                x0 = rng.uniform(-4.5, 4.5, dim)
            elif start_idx < 6:
                # Next few: perturb current best
                x0 = best_x + 1.0 * rng.standard_normal(dim)
                x0 = np.clip(x0, -4.5, 4.5)
            else:
                # Later: smaller perturbations
                x0 = best_x + 0.5 * rng.standard_normal(dim)
                x0 = np.clip(x0, -4.5, 4.5)
            
            max_evals = min(calls_per_start, budget - calls_used)
            if max_evals < 10:
                break
            
            x_best_local, f_best_local = nelder_mead(x0, max_evals)
            
            # If we found something better, update global
            if f_best_local < best_f:
                best_f = f_best_local
                best_x = x_best_local.copy()
        
        # Phase 3: Final refinement around global best
        if best_x is not None and time_ok():
            final_budget = min(30, budget - calls_used)
            if final_budget > 10:
                nelder_mead(best_x, final_budget, initial_alpha=0.1)
        
        return (best_x, best_f)
# EVOLVE-BLOCK-END
