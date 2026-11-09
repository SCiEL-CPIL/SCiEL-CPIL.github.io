"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import scipy.optimize

class Optimizer:
    """Hybrid Multi-Start Nelder-Mead optimizer for BBOB f20."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290
        
        n = self.dim
        budget = self.budget
        
        # State tracking
        best_x = None
        best_f = np.inf
        calls_made = 0
        
        # Helper to count calls and track best
        def safe_call(x):
            nonlocal calls_made, best_x, best_f
            if calls_made >= budget:
                return None
            if time.time() > deadline:
                return None
            
            # Clip to box
            x_clipped = np.clip(x, -5, 5)
            f_val = func(x_clipped)
            calls_made += 1
            
            if f_val < best_f:
                best_f = f_val
                best_x = x_clipped.copy()
            return f_val

        # Phase 1: Multi-Start Sampling to find a good basin quickly
        # We use a mix of random points and a coarse grid to ensure coverage.
        # For dim=5, a full grid is too large, so we sample strategically.
        
        # 1. Center
        safe_call(np.zeros(n))
        
        # 2. Random points
        # Use a fixed seed for reproducibility in the "black box" sense? 
        # The task says "hidden seeded shift", implying the function is deterministic per instance.
        # We can use np.random without seeding to get different starts if run multiple times, 
        # but here we just want a good start.
        
        rng = np.random.default_rng(42)
        
        # Generate initial candidates
        # We will spend about 20% of the budget on this phase
        num_starts = min(50, budget // 10)
        
        # Strategy: 
        # - 1 center
        # - 10 random uniform
        # - 10 random normal (sigma=2)
        # - 10 random on vertices of the box (corners)
        # - 10 random near the boundary
        
        starts = []
        
        # Random Uniform
        starts.append(rng.uniform(-5, 5, n))
        
        # Random Normal (centered)
        starts.append(rng.normal(0, 2, n))
        
        # Corners (sample a few)
        for _ in range(5):
            corner = rng.choice([-5, 5], n)
            starts.append(corner)
            
        # More randoms
        for _ in range(num_starts - 7):
            starts.append(rng.uniform(-5, 5, n))

        for s in starts:
            if calls_made >= budget or time.time() > deadline:
                break
            safe_call(s)
            
        if best_x is None:
            # Fallback if no calls made
            best_x = np.zeros(n)
            best_f = func(best_x)
            calls_made = 1
            return best_x, best_f

        # Phase 2: Local Refinement using Nelder-Mead
        # Nelder-Mead is fast. We can run it for the remaining budget.
        # We need to be careful with the number of function evaluations.
        # scipy.optimize.minimize has a maxiter parameter, but it doesn't strictly limit function calls.
        # We can use a callback to stop if we exceed budget or time.
        
        # However, scipy's Nelder-Mead can be slow if it takes tiny steps.
        # Let's implement a simple custom Nelder-Mead or use scipy with careful limits.
        
        # Using scipy is safer for correctness.
        # We will wrap the function to check limits.
        
        def wrapped_func(x):
            nonlocal calls_made
            if calls_made >= budget:
                return best_f # Return current best to signal stop? No, minimize expects a value.
                             # If we return a constant, it might stop or behave oddly.
                             # Better to raise an exception? No, that kills the run.
                             # The safest way is to just let it run, but we check time.
            if time.time() > deadline:
                return best_f
            return safe_call(x)

        # Options for Nelder-Mead
        # maxiter is not the same as maxfev. 
        # We can estimate maxfev. Each iteration of NM does roughly n+1 evaluations.
        # Remaining budget: budget - calls_made
        # Let's allow it to run until budget is almost exhausted.
        
        # To be safe and fast, we might want to do a few restarts if the first one stalls.
        # But NM is usually good enough.
        
        # Let's use a custom lightweight NM if scipy is too slow? 
        # No, scipy is C-optimized.
        
        # We'll run NM from the best point found in Phase 1.
        
        # Note: scipy.optimize.minimize might make many calls. 
        # We can't strictly limit fevals in the main loop easily without a custom implementation.
        # Let's use a simple custom Nelder-Mead to have strict control over calls.
        
        def nelder_mead(func, x0, maxfev, x_bounds=None):
            n = len(x0)
            # Initialize simplex
            simplex = [x0.copy() for _ in range(n + 1)]
            # Perturb the simplex points
            for i in range(1, n + 1):
                simplex[i][i-1] += 0.05 * (abs(x0[i-1]) + 1)
            
            f_simplex = [func(s) for s in simplex]
            calls = n + 1
            
            while calls < maxfev:
                # Sort by fitness
                order = np.argsort(f_simplex)
                simplex = [simplex[i] for i in order]
                f_simplex = [f_simplex[i] for i in order]
                
                # Check for convergence
                if np.allclose(f_simplex[0], f_simplex[-1]):
                    break
                
                # Centroid of all but worst
                centroid = np.mean(simplex[:-1], axis=0)
                
                # Reflection
                alpha = 1.0
                x_r = centroid + alpha * (centroid - simplex[-1])
                f_r = func(x_r)
                calls += 1
                
                if f_r < f_simplex[0]:
                    # Expansion
                    gamma = 2.0
                    x_e = centroid + gamma * (centroid - simplex[-1])
                    f_e = func(x_e)
                    calls += 1
                    
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
                    rho = 0.5
                    x_c = centroid + rho * (simplex[-1] - centroid)
                    f_c = func(x_c)
                    calls += 1
                    
                    if f_c < f_simplex[-1]:
                        simplex[-1] = x_c
                        f_simplex[-1] = f_c
                    else:
                        # Shrink
                        for i in range(1, n + 1):
                            simplex[i] = simplex[0] + 0.5 * (simplex[i] - simplex[0])
                            f_simplex[i] = func(simplex[i])
                            calls += 1
                            if calls >= maxfev:
                                break
            
            # Return best
            best_idx = np.argmin(f_simplex)
            return simplex[best_idx], f_simplex[best_idx]

        # We need a wrapper for the custom NM that respects the global call count
        def nm_func(x):
            nonlocal calls_made, best_f, best_x
            if calls_made >= budget:
                return best_f
            if time.time() > deadline:
                return best_f
            return safe_call(x)

        # Run NM
        # Reserve some calls for potential restarts? 
        # Let's use 80% of remaining budget for the first NM run.
        remaining = budget - calls_made
        nm_budget = int(remaining * 0.8)
        
        if nm_budget > 10:
            x_opt, f_opt = nelder_mead(nm_func, best_x, nm_budget)
            
            # Check if we found something better
            # nm_func updates best_x and best_f internally via safe_call
            
            # Phase 3: If we have budget left, do a second restart from a different point
            # or continue refining.
            if calls_made < budget and time.time() < deadline:
                # Try a restart from a random point if the first one converged too early?
                # Or just run NM again from the current best with remaining budget.
                remaining2 = budget - calls_made
                if remaining2 > 10:
                    x_opt2, f_opt2 = nelder_mead(nm_func, best_x, remaining2)

        return best_x, best_f
# EVOLVE-BLOCK-END
