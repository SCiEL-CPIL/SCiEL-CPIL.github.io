"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-start Nelder-Mead with deadline guard for BBOB f21."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 325.0  # Safety margin before 330s kill
        rng = np.random.default_rng(42)
        dim = self.dim
        budget = self.budget

        best_x = None
        best_f = np.inf
        calls_used = 0

        def time_left():
            return deadline - time.time()

        def eval_point(x):
            nonlocal calls_used, best_f, best_x
            if calls_used >= budget:
                return None
            fx = func(x)
            calls_used += 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            return fx

        def nelder_mead(x0, max_iter=200):
            """
            Lightweight Nelder-Mead implementation.
            Returns the best point found during this local search.
            """
            nonlocal calls_used, best_f, best_x
            
            # Initialize simplex
            simplex = [x0.copy()]
            for i in range(dim):
                x_new = x0.copy()
                x_new[i] += 0.5 if x0[i] < 5 else -0.5
                x_new = np.clip(x_new, -5, 5)
                simplex.append(x_new)
            
            # Evaluate initial simplex
            simplex_f = []
            for x in simplex:
                fx = eval_point(x)
                if fx is None:
                    return x0, best_f
                simplex_f.append(fx)
            
            # Sort simplex by fitness
            simplex = np.array(simplex)
            simplex_f = np.array(simplex_f)
            order = np.argsort(simplex_f)
            simplex = simplex[order]
            simplex_f = simplex_f[order]
            
            alpha = 1.0  # Reflection
            gamma = 2.0  # Expansion
            rho = 0.5    # Contraction
            sigma = 0.5  # Shrink
            
            for _ in range(max_iter):
                if calls_used >= budget or time_left() < 5.0:
                    break
                
                # Check convergence
                if np.std(simplex_f) < 1e-8:
                    break
                
                centroid = np.mean(simplex[:-1], axis=0)
                
                # Reflection
                xr = centroid + alpha * (centroid - simplex[-1])
                xr = np.clip(xr, -5, 5)
                fxr = eval_point(xr)
                if fxr is None:
                    break
                
                if simplex_f[0] <= fxr < simplex_f[-2]:
                    simplex[-1] = xr
                    simplex_f[-1] = fxr
                elif fxr < simplex_f[0]:
                    # Expansion
                    xe = centroid + gamma * (xr - centroid)
                    xe = np.clip(xe, -5, 5)
                    fxe = eval_point(xe)
                    if fxe is None:
                        break
                    if fxe < fxr:
                        simplex[-1] = xe
                        simplex_f[-1] = fxe
                    else:
                        simplex[-1] = xr
                        simplex_f[-1] = fxr
                else:
                    # Contraction
                    if fxr < simplex_f[-1]:
                        # Outside contraction
                        xc = centroid + rho * (simplex[-1] - centroid)
                        xc = np.clip(xc, -5, 5)
                        fxc = eval_point(xc)
                        if fxc is None:
                            break
                        if fxc <= min(fxr, simplex_f[-1]):
                            simplex[-1] = xc
                            simplex_f[-1] = fxc
                        else:
                            # Shrink
                            for i in range(1, dim + 1):
                                simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                                simplex[i] = np.clip(simplex[i], -5, 5)
                                fxi = eval_point(simplex[i])
                                if fxi is None:
                                    return simplex[0], best_f
                                simplex_f[i] = fxi
                            break
                    else:
                        # Inside contraction
                        xc = centroid - rho * (centroid - simplex[-1])
                        xc = np.clip(xc, -5, 5)
                        fxc = eval_point(xc)
                        if fxc is None:
                            break
                        if fxc < simplex_f[-1]:
                            simplex[-1] = xc
                            simplex_f[-1] = fxc
                        else:
                            # Shrink
                            for i in range(1, dim + 1):
                                simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                                simplex[i] = np.clip(simplex[i], -5, 5)
                                fxi = eval_point(simplex[i])
                                if fxi is None:
                                    return simplex[0], best_f
                                simplex_f[i] = fxi
                            break
                
                # Re-sort simplex
                order = np.argsort(simplex_f)
                simplex = simplex[order]
                simplex_f = simplex_f[order]
            
            return simplex[0], simplex_f[0]

        # Strategy: Multi-start Nelder-Mead
        # 1. Generate a diverse set of starting points using a grid/random mix
        # 2. Evaluate each briefly to identify promising regions
        # 3. Run Nelder-Mead on the top candidates
        
        # Phase 1: Broad Exploration
        # Use a stratified random sampling approach
        n_initial = 50
        initial_points = []
        
        # Include origin and some structured points
        initial_points.append(np.zeros(dim))
        
        # Add random points
        for _ in range(n_initial - 1):
            p = rng.uniform(-5, 5, dim)
            initial_points.append(p)
            
        # Sort by function value
        initial_results = []
        for p in initial_points:
            if calls_used >= budget or time_left() < 10.0:
                break
            p = np.clip(p, -5, 5)
            fp = eval_point(p)
            if fp is None:
                break
            initial_results.append((p, fp))
            
        if not initial_results:
            return best_x, best_f
            
        # Sort by fitness
        initial_results.sort(key=lambda x: x[1])
        
        # Phase 2: Local Refinement with Nelder-Mead
        # Pick top K candidates to refine
        # Budget remaining is roughly budget - n_initial
        # Nelder-Mead in 5D takes ~50-100 calls to converge well
        # We can afford roughly (budget - n_initial) / 50 restarts
        remaining_budget = budget - calls_used
        max_refinements = max(1, int(remaining_budget / 60))
        
        candidates = [p for p, f in initial_results[:max_refinements]]
        
        for p in candidates:
            if calls_used >= budget or time_left() < 10.0:
                break
            
            # Run Nelder-Mead starting from this point
            nelder_mead(p, max_iter=100)
            
            # If we found a new global best, we might want to restart NM from there
            # but for now, we just proceed to the next candidate
            
        return best_x, best_f

# EVOLVE-BLOCK-END
