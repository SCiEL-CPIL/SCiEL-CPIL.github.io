"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

class Optimizer:
    """
    Optimizer using:
    1. Latin Hypercube Sampling (LHS) for global exploration.
    2. Coordinate Descent with Golden Section Line Search for local refinement 
       (exploits separable structure of Griewank-Rosenbrock).
    3. L-BFGS-B for final high-precision polish.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0  # Safety margin for 300s limit
        dim = self.dim
        budget = self.budget
        
        calls_made = 0
        best_x = None
        best_f = np.inf

        def safe_call(x):
            nonlocal calls_made, best_x, best_f
            if calls_made >= budget:
                return np.inf
            x = np.clip(x, -5.0, 5.0)
            calls_made += 1
            try:
                f_val = float(func(x))
            except Exception:
                f_val = np.inf
            
            if f_val < best_f:
                best_f = f_val
                best_x = x.copy()
            return f_val

        def lhs_sample(n, dim, lower=-5.0, upper=5.0):
            """Generate n points using Latin Hypercube Sampling."""
            points = np.zeros((n, dim))
            for i in range(n):
                for j in range(dim):
                    points[i, j] = np.random.uniform(lower, upper)
            # Shuffle to break correlation between dimensions
            np.random.shuffle(points)
            return points

        def coordinate_descent(x_start, max_iter=100, tol=1e-8):
            """
            Perform coordinate-wise optimization using Golden Section Search.
            """
            nonlocal calls_made, best_x, best_f
            x = x_start.copy()
            f_current = safe_call(x)
            
            # If we hit budget or deadline immediately, return
            if calls_made >= budget or time.time() > deadline:
                return x, f_current

            for _ in range(max_iter):
                if time.time() > deadline or calls_made >= budget - 10:
                    break
                    
                improvement = 0.0
                for i in range(dim):
                    if time.time() > deadline or calls_made >= budget - 10:
                        break
                    
                    # Define 1D function along dimension i
                    def f_1d(t):
                        x_new = x.copy()
                        x_new[i] = t
                        return safe_call(x_new)

                    # Initial bracket for golden section
                    # Start with a reasonable step size
                    step = 1.0
                    t_center = x[i]
                    t_low = t_center - step
                    t_high = t_center + step
                    
                    # Expand bracket if necessary (simple version)
                    f_low = f_1d(t_low)
                    f_high = f_1d(t_high)
                    f_center = f_current
                    
                    # If center is not best, expand
                    if f_center < f_low and f_center < f_high:
                        # Already at local min in this coord? Try smaller step or break
                        # For GR function, gradient is rarely exactly 0 away from min.
                        # Let's try a smaller step to be sure
                        step = 0.1
                        t_low = t_center - step
                        t_high = t_center + step
                        f_low = f_1d(t_low)
                        f_high = f_1d(t_high)
                        
                        if f_center < f_low and f_center < f_high:
                            continue # No improvement in this coord

                    # Ensure we have a bracket where center is better than ends? 
                    # Actually, Golden Section requires f(a) > f(b) and f(c) > f(d) 
                    # or similar. Let's just use scipy's minimize_scalar for robustness 
                    # within the coordinate descent loop, or implement a simple 
                    # adaptive step halving.
                    
                    # Given the complexity of robust bracketing, let's use a simpler 
                    # adaptive step halving/doubling approach which is often faster 
                    # for smooth functions like GR.
                    
                    # Try positive step
                    best_t = t_center
                    best_f_1d = f_current
                    step = 0.5
                    for _ in range(20):
                        if calls_made >= budget - 10 or time.time() > deadline:
                            break
                        t_test = t_center + step
                        if t_test > 5.0 or t_test < -5.0:
                            break
                        f_test = f_1d(t_test)
                        if f_test < best_f_1d:
                            best_f_1d = f_test
                            best_t = t_test
                            step *= 0.5 # Move closer
                        else:
                            # Try negative direction
                            t_test_neg = t_center - step
                            if t_test_neg > -5.0 and t_test_neg < 5.0:
                                f_test_neg = f_1d(t_test_neg)
                                if f_test_neg < best_f_1d:
                                    best_f_1d = f_test_neg
                                    best_t = t_test_neg
                                    step *= 0.5
                                else:
                                    # If neither direction improves significantly, 
                                    # we might be at a local min or step is too large.
                                    # Shrink step to see if we missed a dip
                                    step *= 0.5
                            else:
                                step *= 0.5
                                
                        if step < 1e-6:
                            break

                    if best_t != t_center:
                        x[i] = best_t
                        f_current = best_f_1d
                        improvement += abs(f_current - f_center)
                        # Update global best
                        if f_current < best_f:
                            best_f = f_current
                            best_x = x.copy()

                if improvement < tol:
                    break

            return x, f_current

        # Phase 1: Global Exploration with LHS
        # Use ~20% of budget for exploration
        explore_budget = int(budget * 0.2)
        n_lhs = min(explore_budget, 50) # Don't use too many, keep some for refinement
        
        # Generate LHS points
        lhs_points = lhs_sample(n_lhs, dim)
        
        # Evaluate LHS points
        for p in lhs_points:
            if time.time() > deadline or calls_made >= budget - 50:
                break
            f_val = safe_call(p)
            if f_val < best_f:
                best_f = f_val
                best_x = p.copy()

        # Phase 2: Local Refinement using Coordinate Descent from best point
        # Also try a few other promising points from LHS
        # Sort LHS points by fitness (we don't have them all stored, so just use best)
        
        # Run CD from the global best
        if best_x is not None:
            x_ref, f_ref = coordinate_descent(best_x, max_iter=50)
        
        # Phase 3: Multi-start Refinement
        # If we have budget left, try refining a few other points from LHS
        # Since we didn't store all LHS fits, we'll just pick a few random points 
        # near the best or re-sample a bit.
        
        # Let's do a few random restarts + CD
        n_restarts = 5
        for i in range(n_restarts):
            if time.time() > deadline or calls_made >= budget - 20:
                break
            
            # Generate a point near the current best
            x_new = best_x + np.random.normal(0, 0.5, dim)
            x_new = np.clip(x_new, -5, 5)
            
            f_new = safe_call(x_new)
            if f_new < best_f:
                best_f = f_new
                best_x = x_new.copy()
            
            # Refine this point
            x_ref, f_ref = coordinate_descent(x_new, max_iter=30)

        # Phase 4: Final Polish with L-BFGS-B
        # L-BFGS-B uses gradient approximation internally or we can provide it.
        # Since func is black-box, L-BFGS-B will use finite differences.
        # This is expensive per call but very accurate.
        
        if calls_made < budget - 20 and time.time() < deadline:
            try:
                # Define a wrapper that counts calls
                # L-BFGS-B might make many calls. We need to limit them.
                # We'll just run it and hope it converges quickly or stops.
                # To be safe, we'll only run it if we have >50 calls left.
                
                def lbfgsb_func(x):
                    return safe_call(x)
                
                # Bounds
                bounds = [(-5.0, 5.0)] * dim
                
                # Run L-BFGS-B
                # maxiter is not the number of function calls, but iterations.
                # Each iteration does ~dim function calls for gradient.
                # So maxiter=10 might use ~50 calls.
                result = minimize(
                    lbfgsb_func, 
                    best_x, 
                    method='L-BFGS-B', 
                    bounds=bounds,
                    options={'maxiter': 10, 'ftol': 1e-10, 'gtol': 1e-8}
                )
                
                if result.fun < best_f:
                    best_f = result.fun
                    best_x = result.x.copy()
                    
            except Exception:
                pass

        return best_x, best_f

# EVOLVE-BLOCK-END
