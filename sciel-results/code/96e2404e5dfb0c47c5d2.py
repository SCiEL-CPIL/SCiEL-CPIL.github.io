"""
Multi-start Nelder-Mead with L-BFGS-B polish for BBOB f11 (Discus).

Strategy:
1. Phase 1 (Exploration): Evaluate a deterministic set of diverse starting points
   (origin, axis-aligned points, and some random-like deterministic points) to
   identify multiple promising basins. This provides strong early progress for AOCC.
   
2. Phase 2 (Exploitation): For each of the top few starting points, run a 
   Nelder-Mead simplex method to adapt to the local curvature of the Discus function.
   Nelder-Mead is effective for moderately non-convex, anisotropic functions.
   
3. Phase 3 (Polish): After Nelder-Mead converges or exhausts its iterations, 
   apply L-BFGS-B from the best point found. L-BFGS-B uses gradient information
   (via finite differences) and can converge very rapidly near the minimum,
   exploiting the known structure of f11 more efficiently than derivative-free methods.

This hybrid approach combines the robustness of Nelder-Mead for basin-finding
with the fast local convergence of L-BFGS-B for final polishing.
"""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

class Optimizer:
    """
    Multi-start Nelder-Mead + L-BFGS-B Optimizer for BBOB f11 (Discus).
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290  # Safety margin before 300s limit
        dim = self.dim
        budget = self.budget
        
        calls_made = 0
        best_x = None
        best_f = float('inf')
        
        # Helper to check budget and time
        def can_call():
            nonlocal calls_made
            if calls_made >= budget:
                return False
            if time.time() > deadline:
                return False
            return True

        def evaluate(x):
            nonlocal calls_made, best_x, best_f
            if not can_call():
                return float('inf')
            x = np.asarray(x, dtype=float)
            x = np.clip(x, -5, 5)
            fx = func(x)
            calls_made += 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            return fx

        # --- Phase 1: Diverse Initialization ---
        # Generate a set of diverse starting points to cover the search space.
        # The minimum is known to be in [-4, 4]^5.
        
        # Start with origin and axis-aligned points
        start_points = [np.zeros(dim)]
        
        # Add points at +/- 3 along each axis (within [-4,4])
        for i in range(dim):
            for sign in [-1, 1]:
                p = np.zeros(dim)
                p[i] = sign * 3.0
                start_points.append(p)
        
        # Add some deterministic pseudo-random points
        # Use a fixed seed for reproducibility
        rng = np.random.RandomState(42)
        for _ in range(10):
            p = rng.uniform(-4, 4, size=dim)
            start_points.append(p)
        
        # Evaluate all starting points
        start_f_values = []
        for p in start_points:
            if not can_call():
                break
            fx = evaluate(p)
            start_f_values.append((fx, p))
        
        if best_x is None:
            # Fallback
            x0 = np.zeros(dim)
            evaluate(x0)
            if best_x is None:
                return np.zeros(dim)
        
        # Sort starting points by function value, keep top few for multi-start
        start_f_values.sort(key=lambda x: x[0])
        top_starts = start_f_values[:min(3, len(start_f_values))]
        
        # --- Phase 2 & 3: Nelder-Mead + L-BFGS-B for each top start ---
        
        def nelder_mead_lbbfgsb(x_start, f_start, nm_max_iter=30):
            nonlocal calls_made, best_x, best_f
            
            n = dim
            
            # Initialize simplex for Nelder-Mead
            step = 0.5
            simplex = [x_start.copy()]
            f_values = [f_start]
            
            for i in range(n):
                v = x_start.copy()
                v[i] += step
                simplex.append(v)
                if not can_call():
                    return
                fx = evaluate(v)
                f_values.append(fx)
            
            # Main Nelder-Mead Loop
            for _ in range(nm_max_iter):
                if not can_call():
                    break
                    
                # Sort simplex by function value
                pairs = sorted(zip(f_values, simplex), key=lambda p: p[0])
                f_values = [p[0] for p in pairs]
                simplex = [p[1] for p in pairs]
                
                # Update best if improved
                if f_values[0] < best_f:
                    best_f = f_values[0]
                    best_x = simplex[0].copy()
                
                # Convergence check
                if abs(f_values[-1] - f_values[0]) < 1e-8:
                    break
                
                x_worst = simplex[-1]
                f_worst = f_values[-1]
                x_best = simplex[0]
                f_best = f_values[0]
                
                # Centroid of all points except worst
                centroid = np.mean(simplex[:-1], axis=0)
                
                # Reflection
                alpha = 1.0
                x_ref = centroid + alpha * (centroid - x_worst)
                x_ref = np.clip(x_ref, -5, 5)
                f_ref = evaluate(x_ref)
                
                if f_ref < f_best:
                    # Expansion
                    gamma = 2.0
                    x_exp = centroid + gamma * (centroid - x_worst)
                    x_exp = np.clip(x_exp, -5, 5)
                    f_exp = evaluate(x_exp)
                    if f_exp < f_ref:
                        simplex[-1] = x_exp
                        f_values[-1] = f_exp
                    else:
                        simplex[-1] = x_ref
                        f_values[-1] = f_ref
                elif f_ref < f_values[-2]:
                    # Accept reflection
                    simplex[-1] = x_ref
                    f_values[-1] = f_ref
                else:
                    # Contraction
                    rho = 0.5
                    x_con = centroid + rho * (x_worst - centroid)
                    x_con = np.clip(x_con, -5, 5)
                    f_con = evaluate(x_con)
                    
                    if f_con < f_worst:
                        simplex[-1] = x_con
                        f_values[-1] = f_con
                    else:
                        # Shrink
                        for i in range(1, n + 1):
                            simplex[i] = x_best + 0.5 * (simplex[i] - x_best)
                            if not can_call():
                                return
                            f_values[i] = evaluate(simplex[i])
                        
            # After Nelder-Mead, use L-BFGS-B for polishing
            if can_call():
                # Use the best point from NM as starting point for L-BFGS-B
                x_nm_best = simplex[0].copy()
                f_nm_best = f_values[0]
                
                # Define wrapper for L-BFGS-B that tracks calls
                def func_wrapper(x):
                    nonlocal calls_made, best_x, best_f
                    if not can_call():
                        return float('inf')
                    x = np.clip(x, -5, 5)
                    fx = func(x)
                    calls_made += 1
                    if fx < best_f:
                        best_f = fx
                        best_x = x.copy()
                    return fx
                
                # Run L-BFGS-B with limited iterations to conserve budget
                # maxiter controls the number of function evaluations
                # For n=5, L-BFGS-B typically needs ~20-50 evals for good convergence
                try:
                    res = minimize(
                        func_wrapper, 
                        x_nm_best, 
                        method='L-BFGS-B', 
                        bounds=[(-5, 5)] * dim,
                        options={'maxiter': 30, 'maxfun': 50}
                    )
                except Exception:
                    pass
                
                # Update best if L-BFGS-B found something better
                if hasattr(res, 'x') and hasattr(res, 'fun'):
                    if res.fun < best_f:
                        best_f = res.fun
                        best_x = res.x.copy()
        
        # Run multi-start: NM + L-BFGS-B for each top starting point
        for f_start, x_start in top_starts:
            if not can_call():
                break
            nelder_mead_lbbfgsb(x_start, f_start, nm_max_iter=20)
        
        return best_x

# EVOLVE-BLOCK-END
