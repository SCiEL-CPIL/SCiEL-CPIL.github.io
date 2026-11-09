import numpy as np
import time
from scipy.optimize import minimize

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 295.0  # Safety margin before 330s kill
        budget = self.budget
        dim = self.dim
        rng = np.random.default_rng(12345)

        best_x = None
        best_f = np.inf
        calls = 0

        def try_call(x):
            nonlocal calls, best_x, best_f
            if calls >= budget:
                return None
            x = np.clip(np.asarray(x, dtype=float), -5.0, 5.0)
            try:
                fx = float(func(x))
            except Exception:
                return None
            calls += 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            return fx

        # Phase 1: Multi-start Nelder-Mead with adaptive step sizes
        # We use a set of diverse starting points to cover the search space
        # For dim=5, we'll use multiple restarts with different scales
        
        # Generate diverse starting points
        # Mix of: origin, random points, points on a grid-like structure
        n_starts = 12
        
        # Starting points: combine structured and random
        start_points = []
        
        # Origin
        start_points.append(np.zeros(dim))
        
        # Random points with varying scales
        for i in range(n_starts - 1):
            scale = 2.0 * (0.3 + 0.7 * (i / max(1, n_starts - 2)))
            p = rng.standard_normal(dim) * scale
            p = np.clip(p, -4.5, 4.5)  # Keep within box with margin
            start_points.append(p)
        
        # Add some structured points
        for i in range(dim):
            p = np.zeros(dim)
            p[i] = 3.0 * (1 if i % 2 == 0 else -1)
            start_points.append(p)
        
        # Shuffle start points
        idx = rng.permutation(len(start_points))
        start_points = [start_points[i] for i in idx]
        
        # For each start point, run a local Nelder-Mead optimization
        # Budget allocation: give each start a fraction of the budget
        # But we need to be careful - if we use all budget on early starts,
        # we won't have time for later ones
        
        # Strategy: allocate budget proportionally, but with diminishing returns
        # First few starts get more budget, later ones get less
        # Also, stop early if we're converging well
        
        # Use scipy's Nelder-Mead for robustness
        # But we need to track calls carefully
        
        # Alternative: implement our own Nelder-Mead with call tracking
        # This gives us more control over the budget
        
        def nelder_mead_local(x0, max_calls, timeout_remaining):
            """Run Nelder-Mead from x0 with at most max_calls function evaluations."""
            nonlocal calls, best_x, best_f
            
            if calls >= budget or time.time() >= deadline:
                return
            
            # Initialize simplex
            simplex = [x0.copy()]
            for i in range(dim):
                perturbation = x0.copy()
                # Adaptive initial step size
                step = 0.5
                perturbation[i] += step
                simplex.append(perturbation)
            
            simplex_f = []
            for i in range(dim + 1):
                fx_i = try_call(simplex[i])
                if fx_i is None:
                    return
                simplex_f.append(fx_i)
            
            # Nelder-Mead parameters
            alpha = 1.0   # Reflection
            gamma = 2.0   # Expansion
            rho = 0.5     # Contraction
            sigma_nm = 0.5  # Shrinkage
            
            # Run Nelder-Mead
            while calls < budget and time.time() < deadline:
                # Check if we've exceeded local budget
                local_calls = calls
                if local_calls - (calls - max_calls) >= max_calls:
                    break
                
                # Sort simplex by fitness
                sorted_idx = np.argsort(simplex_f)
                simplex = [simplex[i].copy() for i in sorted_idx]
                simplex_f = [simplex_f[i] for i in sorted_idx]
                
                # Update best
                if simplex_f[0] < best_f:
                    best_f = simplex_f[0]
                    best_x = simplex[0].copy()
                
                # Check convergence
                spread = max(simplex_f) - min(simplex_f)
                if spread < 1e-10 or all(abs(simplex_f[i] - simplex_f[0]) < 1e-12 for i in range(1, dim + 1)):
                    break
                
                # Centroid of all points except worst
                centroid = np.mean(simplex[:-1], axis=0)
                
                # Reflection
                x_r = centroid + alpha * (centroid - simplex[-1])
                f_r = try_call(x_r)
                if f_r is None:
                    break
                
                if f_r < simplex_f[0]:
                    # Try expansion
                    x_e = centroid + gamma * (x_r - centroid)
                    f_e = try_call(x_e)
                    if f_e is None:
                        break
                    
                    if f_e < f_r:
                        simplex[-1] = x_e
                        simplex_f[-1] = f_e
                    else:
                        simplex[-1] = x_r
                        simplex_f[-1] = f_r
                elif f_r < simplex_f[-2]:
                    simplex[-1] = x_r
                    simplex_f[-1] = f_r
                else:
                    # Try contraction
                    x_c = centroid + rho * (simplex[-1] - centroid)
                    f_c = try_call(x_c)
                    if f_c is None:
                        break
                    
                    if f_c < simplex_f[-1]:
                        simplex[-1] = x_c
                        simplex_f[-1] = f_c
                    else:
                        # Shrink
                        for i in range(1, dim + 1):
                            simplex[i] = simplex[0] + sigma_nm * (simplex[i] - simplex[0])
                            f_i = try_call(simplex[i])
                            if f_i is None:
                                return
                            simplex_f[i] = f_i
        
        # Allocate budget across starts
        # Use a geometric decay: first start gets more, subsequent get less
        # Total budget: 500 calls
        # Reserve some for final polishing
        
        # Let's use a strategy where we try starts in order, 
        # allocating a fraction of remaining budget to each
        remaining_budget = budget
        for i, sp in enumerate(start_points):
            if calls >= budget or time.time() >= deadline:
                break
            
            # Allocate budget: diminishing, but ensure at least 10 calls per start
            # First few starts get more
            if i == 0:
                alloc = min(remaining_budget, 80)
            elif i < 4:
                alloc = min(remaining_budget, 50)
            elif i < 8:
                alloc = min(remaining_budget, 30)
            else:
                alloc = min(remaining_budget, 20)
            
            alloc = max(alloc, 10)
            alloc = min(alloc, remaining_budget)
            
            if alloc < 10:
                break
            
            nelder_mead_local(sp, alloc, deadline - time.time())
            remaining_budget = budget - calls
            
            # If we've made significant progress, we can be more aggressive
            # If best_f is very good, spend more on polishing
        
        # Phase 2: Final polishing with more intensive local search
        if best_x is not None and calls < budget and time.time() < deadline:
            # Run a few more local optimizations from the best point
            # with smaller initial steps
            
            # Try different step sizes for the simplex
            for step_scale in [0.1, 0.05, 0.01]:
                if calls >= budget or time.time() >= deadline:
                    break
                
                x0 = best_x.copy()
                
                # Initialize simplex with smaller steps
                simplex = [x0.copy()]
                for i in range(dim):
                    perturbation = x0.copy()
                    perturbation[i] += step_scale
                    simplex.append(perturbation)
                
                simplex_f = []
                for i in range(dim + 1):
                    fx_i = try_call(simplex[i])
                    if fx_i is None:
                        break
                    simplex_f.append(fx_i)
                
                if len(simplex_f) < dim + 1:
                    continue
                
                # Run a few iterations of Nelder-Mead
                alpha = 1.0
                gamma = 2.0
                rho = 0.5
                sigma_nm = 0.5
                
                max_iters = 30
                for _ in range(max_iters):
                    if calls >= budget or time.time() >= deadline:
                        break
                    
                    # Sort simplex by fitness
                    sorted_idx = np.argsort(simplex_f)
                    simplex = [simplex[i].copy() for i in sorted_idx]
                    simplex_f = [simplex_f[i] for i in sorted_idx]
                    
                    # Update best
                    if simplex_f[0] < best_f:
                        best_f = simplex_f[0]
                        best_x = simplex[0].copy()
                    
                    # Check convergence
                    spread = max(simplex_f) - min(simplex_f)
                    if spread < 1e-12:
                        break
                    
                    # Centroid of all points except worst
                    centroid = np.mean(simplex[:-1], axis=0)
                    
                    # Reflection
                    x_r = centroid + alpha * (centroid - simplex[-1])
                    f_r = try_call(x_r)
                    if f_r is None:
                        break
                    
                    if f_r < simplex_f[0]:
                        # Try expansion
                        x_e = centroid + gamma * (x_r - centroid)
                        f_e = try_call(x_e)
                        if f_e is None:
                            break
                        
                        if f_e < f_r:
                            simplex[-1] = x_e
                            simplex_f[-1] = f_e
                        else:
                            simplex[-1] = x_r
                            simplex_f[-1] = f_r
                    elif f_r < simplex_f[-2]:
                        simplex[-1] = x_r
                        simplex_f[-1] = f_r
                    else:
                        # Try contraction
                        x_c = centroid + rho * (simplex[-1] - centroid)
                        f_c = try_call(x_c)
                        if f_c is None:
                            break
                        
                        if f_c < simplex_f[-1]:
                            simplex[-1] = x_c
                            simplex_f[-1] = f_c
                        else:
                            # Shrink
                            for i in range(1, dim + 1):
                                simplex[i] = simplex[0] + sigma_nm * (simplex[i] - simplex[0])
                                f_i = try_call(simplex[i])
                                if f_i is None:
                                    break
                                simplex_f[i] = f_i
                            if f_i is None:
                                break
        
        # Phase 3: If we have budget left, do a final global exploration
        # using random restarts with very small steps around the best point
        if best_x is not None and calls < budget and time.time() < deadline:
            # Try random perturbations around the best point
            for _ in range(50):
                if calls >= budget or time.time() >= deadline:
                    break
                
                # Random direction with small step
                direction = rng.standard_normal(dim)
                direction /= np.linalg.norm(direction)
                step_size = 0.01 * (1 + rng.random())
                
                x_try = best_x + step_size * direction
                x_try = np.clip(x_try, -5.0, 5.0)
                
                fx = try_call(x_try)
                if fx is None:
                    break
                
                # If we found a better point, do a quick local refinement
                if fx < best_f - 1e-8:
                    # Quick Nelder-Mead from this point
                    x0 = x_try.copy()
                    simplex = [x0.copy()]
                    for i in range(dim):
                        perturbation = x0.copy()
                        perturbation[i] += 0.01
                        simplex.append(perturbation)
                    
                    simplex_f = []
                    for i in range(dim + 1):
                        fx_i = try_call(simplex[i])
                        if fx_i is None:
                            break
                        simplex_f.append(fx_i)
                    
                    if len(simplex_f) < dim + 1:
                        continue
                    
                    # Run a few iterations
                    for _ in range(10):
                        if calls >= budget or time.time() >= deadline:
                            break
                        
                        sorted_idx = np.argsort(simplex_f)
                        simplex = [simplex[i].copy() for i in sorted_idx]
                        simplex_f = [simplex_f[i] for i in sorted_idx]
                        
                        if simplex_f[0] < best_f:
                            best_f = simplex_f[0]
                            best_x = simplex[0].copy()
                        
                        spread = max(simplex_f) - min(simplex_f)
                        if spread < 1e-12:
                            break
                        
                        centroid = np.mean(simplex[:-1], axis=0)
                        x_r = centroid + 1.0 * (centroid - simplex[-1])
                        f_r = try_call(x_r)
                        if f_r is None:
                            break
                        
                        if f_r < simplex_f[0]:
                            x_e = centroid + 2.0 * (x_r - centroid)
                            f_e = try_call(x_e)
                            if f_e is None:
                                break
                            if f_e < f_r:
                                simplex[-1] = x_e
                                simplex_f[-1] = f_e
                            else:
                                simplex[-1] = x_r
                                simplex_f[-1] = f_r
                        elif f_r < simplex_f[-2]:
                            simplex[-1] = x_r
                            simplex_f[-1] = f_r
                        else:
                            x_c = centroid + 0.5 * (simplex[-1] - centroid)
                            f_c = try_call(x_c)
                            if f_c is None:
                                break
                            if f_c < simplex_f[-1]:
                                simplex[-1] = x_c
                                simplex_f[-1] = f_c
                            else:
                                for i in range(1, dim + 1):
                                    simplex[i] = simplex[0] + 0.5 * (simplex[i] - simplex[0])
                                    f_i = try_call(simplex[i])
                                    if f_i is None:
                                        break
                                    simplex_f[i] = f_i
                                if f_i is None:
                                    break
        
        if best_x is None:
            best_x = np.zeros(dim)
            best_f = 0.0
        
        return best_x, best_f
