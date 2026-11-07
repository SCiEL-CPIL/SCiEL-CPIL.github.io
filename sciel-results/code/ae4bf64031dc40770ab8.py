"""
Hybrid Global Lattice Search + Nelder-Mead Simplex Optimizer.
Strategy:
1. Perform a sparse global lattice search over [-5, 5]^5 to identify the basin of attraction.
   This avoids the slow convergence of CMA-ES and the gradient approximation errors of L-BFGS-B.
2. Select the best point from the lattice search.
3. Run a Nelder-Mead simplex method starting from this point to exploit the function's smoothness
   and rapidly converge to the local minimum.
4. If the simplex method stagnates or the budget allows, restart Nelder-Mead from the next best
   lattice point to escape potential local minima.
"""
# EVOLVE-BLOCK-START
import numpy as np
import time
import math


class Optimizer:
    """Hybrid Global Lattice Search + Nelder-Mead Simplex Optimizer."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280  # Safety margin before 330s kill

        dim = self.dim
        budget = self.budget
        calls_made = 0

        # Stage 1: Global Lattice Search
        # We want to cover the space reasonably well without exhausting the budget.
        # Let's try a 3^5 = 243 point lattice? That's almost half the budget.
        # Maybe 2^5 = 32 points (corners) + center? Too coarse.
        # Let's use a stratified sampling approach or a smaller lattice.
        # A 4x4x4x4x4 lattice is 1024 points, too many.
        # A 3x3x3x3x3 lattice is 243 points.
        # Let's reserve ~50 calls for global search and ~450 for local.
        
        # Generate a set of diverse starting points.
        # Use a Halton sequence for low-discrepancy sampling to ensure good coverage.
        def halton_sequence(n, dim):
            """Generate n points in [0, 1]^dim using Halton sequence."""
            points = []
            for i in range(1, n + 1):
                point = []
                for d in range(1, dim + 1):
                    f = 1.0
                    i_val = i
                    while i_val >= d:
                        i_val //= d
                        f /= d
                    point.append(f)
                points.append(point)
            return np.array(points)

        n_global = 50
        halton_points = halton_sequence(n_global, dim)
        
        # Map to [-5, 5]^5
        # Note: The minimizer is in [-4, 4]^5, so we can map to [-4.5, 4.5] or similar to be safe
        # but the domain is [-5, 5]. Let's map to [-4.8, 4.8] to stay within bounds and cover likely area.
        start_pts = halton_points * 9.6 - 4.8
        
        # Evaluate all global points
        global_results = []
        for x in start_pts:
            if time.time() > deadline:
                break
            if calls_made >= budget:
                break
            f_val = func(x)
            calls_made += 1
            global_results.append((f_val, x.copy()))
            
        if not global_results:
            # Fallback if we ran out of time/budget immediately (unlikely)
            x0 = np.zeros(dim)
            f0 = func(x0)
            calls_made += 1
            return x0, f0

        # Sort by function value
        global_results.sort(key=lambda t: t[0])
        
        best_f = global_results[0][0]
        best_x = global_results[0][1].copy()
        
        # Stage 2: Nelder-Mead Local Search
        # We will iterate through the top K global points and run Nelder-Mead on each.
        # We allocate a portion of the remaining budget to each attempt.
        # Let's say we try the top 5 points.
        # Remaining budget: budget - calls_made
        # Budget per NM attempt: (budget - calls_made) // 5
        
        # Implement a simple Nelder-Mead from scratch to avoid SciPy dependency issues or version diffs,
        # although SciPy is allowed. Using a custom implementation allows better control over budget.
        
        def nelder_mead(func, x0, budget_nm, dim, deadline, current_calls):
            """
            Custom Nelder-Mead implementation.
            Returns (best_x, best_f, calls_used)
            """
            # Initialize simplex
            # x0 is the centroid. We create dim+1 points.
            # Initial simplex size
            simplex = [x0.copy()]
            for i in range(dim):
                p = x0.copy()
                p[i] += 0.5  # Initial step
                simplex.append(p)
            
            # Evaluate initial simplex
            f_values = []
            for i in range(dim + 1):
                if time.time() > deadline:
                    return x0, func(x0), 0 # Should not happen, but safe
                if current_calls[0] >= budget:
                    return x0, func(x0), 0
                f_val = func(simplex[i])
                current_calls[0] += 1
                f_values.append(f_val)
            
            calls_used = dim + 1
            best_idx = np.argmin(f_values)
            best_f_nm = f_values[best_idx]
            best_x_nm = simplex[best_idx].copy()
            
            # Nelder-Mead parameters
            alpha = 1.0
            gamma = 2.0
            rho = 0.5
            sigma = 0.5
            
            max_iter = budget_nm
            iter_count = 0
            
            while iter_count < max_iter:
                if time.time() > deadline:
                    break
                if current_calls[0] >= budget:
                    break
                    
                # Sort simplex by function value
                order = np.argsort(f_values)
                simplex = [simplex[i].copy() for i in order]
                f_values = [f_values[i] for i in order]
                
                # Centroid of all points except the worst
                centroid = np.mean(simplex[:-1], axis=0)
                
                # Reflection
                x_r = centroid + alpha * (centroid - simplex[-1])
                x_r = np.clip(x_r, -5, 5)
                if current_calls[0] >= budget:
                    break
                f_r = func(x_r)
                current_calls[0] += 1
                calls_used += 1
                iter_count += 1
                
                if f_r < f_values[0]:
                    # Expansion
                    x_e = centroid + gamma * (x_r - centroid)
                    x_e = np.clip(x_e, -5, 5)
                    if current_calls[0] >= budget:
                        break
                    f_e = func(x_e)
                    current_calls[0] += 1
                    calls_used += 1
                    iter_count += 1
                    
                    if f_e < f_r:
                        simplex[-1] = x_e
                        f_values[-1] = f_e
                    else:
                        simplex[-1] = x_r
                        f_values[-1] = f_r
                elif f_r < f_values[-2]:
                    # Accept reflection
                    simplex[-1] = x_r
                    f_values[-1] = f_r
                else:
                    # Contraction
                    # Outside contraction if f_r >= f_values[-1], inside otherwise
                    if f_r < f_values[-1]:
                        # Inside contraction
                        x_c = centroid + rho * (simplex[-1] - centroid)
                    else:
                        # Outside contraction
                        x_c = centroid - rho * (centroid - simplex[-1])
                    
                    x_c = np.clip(x_c, -5, 5)
                    if current_calls[0] >= budget:
                        break
                    f_c = func(x_c)
                    current_calls[0] += 1
                    calls_used += 1
                    iter_count += 1
                    
                    if f_c < f_r:
                        simplex[-1] = x_c
                        f_values[-1] = f_c
                    else:
                        # Shrink
                        for i in range(1, dim + 1):
                            simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                            simplex[i] = np.clip(simplex[i], -5, 5)
                            if current_calls[0] >= budget:
                                break
                            f_shrink = func(simplex[i])
                            current_calls[0] += 1
                            f_values[i] = f_shrink
                            calls_used += 1
                            iter_count += 1
                            if iter_count >= max_iter:
                                break
                        
                        if iter_count >= max_iter:
                            break
                        
                        # Re-sort after shrink
                        order = np.argsort(f_values)
                        simplex = [simplex[i].copy() for i in order]
                        f_values = [f_values[i] for i in order]
                
                # Update best
                if f_values[0] < best_f_nm:
                    best_f_nm = f_values[0]
                    best_x_nm = simplex[0].copy()
                    
                # Convergence check: if simplex is very small, stop
                spread = np.max(f_values) - np.min(f_values)
                if spread < 1e-8:
                    break
                    
            return best_x_nm, best_f_nm, calls_used

        # Try top 5 starting points from global search
        top_k = 5
        remaining_budget = budget - calls_made
        if remaining_budget <= 0:
            return best_x, best_f
            
        budget_per_nm = max(10, remaining_budget // top_k)
        
        current_calls_tracker = [calls_made] # Use list to allow modification inside function
        
        for i in range(min(top_k, len(global_results))):
            if time.time() > deadline:
                break
            if current_calls_tracker[0] >= budget:
                break
            
            start_x = global_results[i][1]
            start_f = global_results[i][0]
            
            # Only run NM if this start is better than current global best?
            # Or just run it anyway?
            # If start_f is already worse than best_f, it might still lead to a better local min.
            # But to save budget, maybe only run if start_f is competitive?
            # Let's run for all top 5.
            
            x_nm, f_nm, _ = nelder_mead(func, start_x, budget_per_nm, dim, deadline, current_calls_tracker)
            
            if f_nm < best_f:
                best_f = f_nm
                best_x = x_nm.copy()
                
            # If we found a very good point (close to 0, which is typical for Rosenbrock), we might stop early?
            # But we don't know f_opt.
            # Let's continue to use the budget.
            
        return best_x, best_f

# EVOLVE-BLOCK-END
