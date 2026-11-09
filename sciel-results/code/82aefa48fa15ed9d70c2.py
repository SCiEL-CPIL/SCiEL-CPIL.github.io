"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


class Optimizer:
    """Multi-start Nelder-Mead with adaptive step sizes, coordinate descent polishing, and BFGS refinement."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 285  # safety margin before 300s limit

        total_budget = self.budget
        dim = self.dim

        best_x = None
        best_f = np.inf
        calls_used = 0

        def safe_func(x):
            nonlocal calls_used
            if calls_used >= total_budget:
                return np.inf
            x = np.clip(x, -5, 5)
            calls_used += 1
            try:
                f = float(func(x))
                if np.isfinite(f):
                    return f
                return np.inf
            except Exception:
                return np.inf

        def bfgs_refine(x0, budget_for_this_run):
            """Use BFGS to refine a point, respecting the call budget."""
            nonlocal calls_used, best_f, best_x
            if calls_used >= total_budget or time.time() > deadline:
                return x0, best_f
            
            x_start = x0.copy()
            f_start = safe_func(x_start)
            
            # Use scipy BFGS with a wrapper that respects our budget
            def wrapped_func(x):
                nonlocal calls_used
                if calls_used >= total_budget or time.time() > deadline:
                    return 1e10
                return safe_func(x)
            
            try:
                result = minimize(wrapped_func, x_start, method='BFGS', 
                                  options={'maxiter': budget_for_this_run, 'gtol': 1e-6})
                if result.success or result.fun < f_start:
                    if result.fun < best_f:
                        best_f = result.fun
                        best_x = result.x.copy()
            except Exception:
                pass
            
            return best_x if best_x is not None else x_start, best_f

        def nelder_mead(x0, step_size, budget_for_this_run, rng):
            """Adaptive Nelder-Mead with coordinate descent polishing."""
            n = dim
            x_best = x0.copy().astype(float)
            f_best = safe_func(x_best)
            
            # Create initial simplex
            simplex = [x_best.copy()]
            for i in range(n):
                x_new = x_best.copy()
                x_new[i] += step_size
                simplex.append(x_new)
            
            fs = [safe_func(s) for s in simplex]
            
            # Track best
            idx_min = np.argmin(fs)
            if fs[idx_min] < f_best:
                f_best = fs[idx_min]
                x_best = simplex[idx_min].copy()
            
            # Nelder-Mead parameters
            alpha = 1.0  # reflection
            gamma = 2.0  # expansion
            rho = 0.5    # contraction
            sigma = 0.5  # shrinkage
            
            # Adaptive step size
            current_step = step_size
            no_improve_count = 0
            max_no_improve = 8
            
            for gen in range(budget_for_this_run):
                if time.time() > deadline:
                    break
                if calls_used >= total_budget:
                    break
                
                # Sort simplex by fitness
                order = np.argsort(fs)
                simplex = [simplex[i] for i in order]
                fs = [fs[i] for i in order]
                
                # Update best
                if fs[0] < f_best:
                    f_best = fs[0]
                    x_best = simplex[0].copy()
                    no_improve_count = 0
                else:
                    no_improve_count += 1
                
                # If no improvement for several iterations, try coordinate descent
                if no_improve_count >= max_no_improve:
                    # Coordinate descent polishing
                    x_cd = x_best.copy()
                    f_cd = f_best
                    for _ in range(2):  # fewer passes to save budget
                        for d in range(n):
                            for direction in [1, -1]:
                                x_try = x_cd.copy()
                                x_try[d] += direction * current_step * 0.1
                                x_try = np.clip(x_try, -5, 5)
                                f_try = safe_func(x_try)
                                if f_try < f_cd:
                                    x_cd = x_try
                                    f_cd = f_try
                        if calls_used >= total_budget or time.time() > deadline:
                            break
                    if f_cd < f_best:
                        f_best = f_cd
                        x_best = x_cd
                    no_improve_count = 0
                    current_step *= 0.5
                    if current_step < 1e-5:
                        break
                
                # Check if we should stop
                if len(set(fs)) == 1 or (fs[-1] - fs[0] < 1e-12):
                    break
                
                # Compute centroid of all but worst point
                centroid = np.mean(simplex[:-1], axis=0)
                
                # Reflection
                x_r = centroid + alpha * (centroid - simplex[-1])
                x_r = np.clip(x_r, -5, 5)
                f_r = safe_func(x_r)
                
                if f_r < fs[0]:
                    # Expansion
                    x_e = centroid + gamma * (centroid - simplex[-1])
                    x_e = np.clip(x_e, -5, 5)
                    f_e = safe_func(x_e)
                    if f_e < f_r:
                        simplex[-1] = x_e
                        fs[-1] = f_e
                    else:
                        simplex[-1] = x_r
                        fs[-1] = f_r
                elif f_r < fs[-2]:
                    simplex[-1] = x_r
                    fs[-1] = f_r
                else:
                    # Contraction
                    if f_r < fs[-1]:
                        # Outside contraction
                        x_c = centroid + 0.5 * (x_r - centroid)
                    else:
                        # Inside contraction
                        x_c = centroid - 0.5 * (simplex[-1] - centroid)
                    x_c = np.clip(x_c, -5, 5)
                    f_c = safe_func(x_c)
                    
                    if f_c < min(f_r, fs[-1]):
                        simplex[-1] = x_c
                        fs[-1] = f_c
                    else:
                        # Shrink
                        for i in range(1, n + 1):
                            simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                            fs[i] = safe_func(simplex[i])
                        current_step *= 0.5
                        if current_step < 1e-5:
                            break
            
            return x_best, f_best

        # Strategy: Multi-start with diverse initializations
        # Phase 1: Grid sampling for early progress (12 calls)
        rng = np.random.default_rng(42)
        
        # Create a small grid of starting points
        grid_points = []
        # Center
        grid_points.append(np.zeros(dim))
        # Axis-aligned points
        for i in range(dim):
            for val in [-2.0, 2.0]:
                x = np.zeros(dim)
                x[i] = val
                grid_points.append(x)
        # Random points
        for _ in range(7):
            grid_points.append(rng.uniform(-4, 4, dim))
        
        # Evaluate grid points (up to 12 calls)
        grid_results = []
        for x in grid_points:
            if calls_used >= total_budget or time.time() > deadline:
                break
            f = safe_func(x)
            grid_results.append((x, f))
            if f < best_f:
                best_f = f
                best_x = x.copy()
        
        # Phase 2: Run Nelder-Mead from top 3 grid points
        # Sort grid results by fitness
        grid_results.sort(key=lambda t: t[1])
        
        # Allocate remaining budget to top 3 starting points
        remaining_budget = total_budget - calls_used
        if remaining_budget > 30:
            # Top 3 starting points
            top_starts = grid_results[:3]
            budget_per_nm = remaining_budget // 4  # reserve 1/4 for BFGS
            
            for x_start, f_start in top_starts:
                if calls_used >= total_budget or time.time() > deadline:
                    break
                if budget_per_nm < 15:
                    break
                
                # Use adaptive step size based on distance to center
                dist = np.linalg.norm(x_start)
                step_size = max(0.5, min(2.0, dist * 0.5 + 0.5))
                
                x_nm, f_nm = nelder_mead(x_start, step_size, budget_per_nm, rng)
                if f_nm < best_f:
                    best_f = f_nm
                    best_x = x_nm
        
        # Phase 3: BFGS refinement from best point
        if calls_used < total_budget and time.time() < deadline and best_x is not None:
            remaining = total_budget - calls_used
            if remaining > 20:
                x_bfgs, f_bfgs = bfgs_refine(best_x, remaining)
                if f_bfgs < best_f:
                    best_f = f_bfgs
                    best_x = x_bfgs
        
        # Phase 4: Final Nelder-Mead polish if budget remains
        if calls_used < total_budget and time.time() < deadline and best_x is not None:
            remaining = total_budget - calls_used
            if remaining > 15:
                step_size = 0.2
                x_polish, f_polish = nelder_mead(best_x, step_size, remaining, rng)
                if f_polish < best_f:
                    best_f = f_polish
                    best_x = x_polish
        
        if best_x is None:
            best_x = np.zeros(dim)
        
        return best_x, best_f

# EVOLVE-BLOCK-END
