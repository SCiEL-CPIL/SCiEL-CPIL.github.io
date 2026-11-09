# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

class Optimizer:
    """
    Hybrid Global-Local Optimizer for BBOB-style functions.
    Uses Halton sequence for global exploration and Nelder-Mead for local refinement.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        start_time = time.time()
        deadline = start_time + 295  # Safety margin before 300s limit

        # Global state
        best_x = None
        best_f = np.inf
        calls_made = 0

        def wrapped_func(x):
            nonlocal calls_made, best_f, best_x
            if calls_made >= budget:
                raise StopIteration("Budget exceeded")
            # Ensure x is a numpy array and within bounds (though scipy might handle it, 
            # the problem statement says func clips, but we should be safe)
            x_arr = np.asarray(x)
            if x_arr.shape != (dim,):
                x_arr = x_arr.reshape(dim)
            
            f_val = float(func(x_arr))
            calls_made += 1
            
            if f_val < best_f:
                best_f = f_val
                best_x = x_arr.copy()
            
            return f_val

        # Phase 1: Global Exploration using Halton Sequence
        # We will sample a fixed number of points to cover the space.
        # 5 dimensions. Let's use 60 points for initial global search.
        n_global = 60
        
        def halton_sequence(index, base):
            q = 1.0
            r = 0.0
            while index > 0:
                q /= base
                r += q * (index % base)
                index //= base
            return r

        def halton_points(n, dim):
            points = np.zeros((n, dim))
            for i in range(1, n + 1):
                for j in range(dim):
                    # Use first dim prime numbers as bases
                    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29]
                    base = primes[j]
                    points[i-1, j] = halton_sequence(i, base)
            return points

        halton_pts = halton_points(n_global, dim)
        # Scale to [-5, 5]
        halton_pts = halton_pts * 10.0 - 5.0
        
        # Evaluate global points
        for i in range(n_global):
            if calls_made >= budget:
                break
            if time.time() > deadline:
                break
            
            pt = halton_pts[i]
            try:
                wrapped_func(pt)
            except StopIteration:
                break

        # Also evaluate origin and a few other strategic points if not already covered well
        strategic_points = [
            np.zeros(dim),
            np.ones(dim) * 2.0,
            np.ones(dim) * -2.0,
            np.ones(dim) * 4.0,
            np.ones(dim) * -4.0,
        ]
        for pt in strategic_points:
            if calls_made >= budget:
                break
            if time.time() > deadline:
                break
            try:
                wrapped_func(pt)
            except StopIteration:
                break

        # Phase 2: Local Refinement using Nelder-Mead
        # We need to select the best points from Phase 1 to start local searches.
        # However, we didn't store all points and values separately in a list easily accessible 
        # without modifying wrapped_func to store history. 
        # Let's re-evaluate the best candidates? No, that wastes calls.
        # Instead, we can just pick a few diverse points from the Halton set that were likely good,
        # or simply run local optimization from the current best_x and a few other random-ish points.
        
        # A better approach: We didn't store the history of (x, f) pairs. 
        # Let's modify the strategy: 
        # 1. Store history during Phase 1.
        
        # Reset for a cleaner implementation if needed, but we already used calls.
        # To keep it simple and robust, we will just run local optimization from:
        # 1. The current global best.
        # 2. A few points from the Halton sequence that are far from the best (to explore other basins).
        
        # Since we don't have the list of (x, f) from Phase 1 readily available in a list,
        # we will just pick a few random points from the Halton set (indices) to restart from.
        # Indices 0, 10, 20, 30, 40, 50 from the Halton set.
        
        local_starts = []
        # Add current best
        if best_x is not None:
            local_starts.append(best_x.copy())
        
        # Add a few diverse points from the Halton sequence
        # We can just re-use the halton_pts array
        diverse_indices = [0, n_global//4, n_global//2, 3*n_global//4, n_global-1]
        for idx in diverse_indices:
            if idx < len(halton_pts):
                local_starts.append(halton_pts[idx].copy())
        
        # Deduplicate starts (approx)
        unique_starts = []
        for s in local_starts:
            is_unique = True
            for u in unique_starts:
                if np.linalg.norm(s - u) < 1e-5:
                    is_unique = False
                    break
            if is_unique:
                unique_starts.append(s)

        # Allocate remaining budget among local starts
        remaining_budget = budget - calls_made
        if remaining_budget <= 0:
            return best_x, best_f

        n_starts = len(unique_starts)
        if n_starts == 0:
             return best_x, best_f
        
        # Limit the number of local starts to avoid using too many calls on poor starts
        # If we have many starts, we give fewer calls each.
        # Max 5 starts to ensure each gets enough budget.
        if n_starts > 5:
            # Sort by distance to best? No, we don't have f values for all.
            # Just take the first 5.
            unique_starts = unique_starts[:5]
            n_starts = 5

        calls_per_start = max(10, remaining_budget // n_starts)

        for i, start_pt in enumerate(unique_starts):
            if calls_made >= budget:
                break
            if time.time() > deadline:
                break
            
            # Define options for Nelder-Mead
            # maxiter is not strictly the number of function calls, but a good proxy.
            # We want to limit function calls. 
            # We can use a callback to check budget.
            
            def callback(xk):
                nonlocal calls_made
                if calls_made >= budget:
                    return True # Stop optimization
                return False

            try:
                res = minimize(
                    wrapped_func, 
                    start_pt, 
                    method='Nelder-Mead',
                    options={
                        'xatol': 1e-6,
                        'fatol': 1e-8,
                        'maxiter': 100, # Cap iterations to prevent infinite loops
                        'adaptive': True
                    },
                    callback=callback
                )
            except StopIteration:
                break
            except Exception:
                # If minimize fails, continue to next start
                continue

            # Update best if needed (wrapped_func already does this, but just in case)
            if res.fun < best_f:
                best_f = res.fun
                best_x = res.x.copy()

        return best_x, best_f

# EVOLVE-BLOCK-END
