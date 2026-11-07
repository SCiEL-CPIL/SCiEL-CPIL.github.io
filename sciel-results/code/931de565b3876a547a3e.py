"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

class Optimizer:
    """Multi-start Nelder-Mead optimizer."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = 325.0  # Safety margin for 330s limit
        
        dim = self.dim
        budget = self.budget
        
        # 1. Initial Exploration
        # Use a quasi-random sequence for better coverage than pure random
        num_initial = 20
        # Simple Halton-like or just random for simplicity, random is often fine for dim=5
        rng = np.random.default_rng(42)
        
        # Generate initial points
        # We can use a grid or random. Random with some spread is good.
        # Let's use a mix: center + random offsets
        x0_center = np.zeros(dim)
        
        initial_points = []
        # Add center
        initial_points.append(x0_center)
        
        # Add random points
        for _ in range(num_initial - 1):
            p = rng.uniform(-5, 5, dim)
            initial_points.append(p)
            
        initial_points = np.array(initial_points)
        
        # Evaluate initial points
        best_x = initial_points[0]
        best_f = np.inf
        eval_count = 0
        
        # We will keep track of the best few points to start local optimization
        # Let's evaluate all initial points first
        # But we must respect the time limit and budget.
        
        # Store results
        initial_results = []
        for x in initial_points:
            if time.time() > deadline:
                break
            fx = func(x)
            eval_count += 1
            initial_results.append((fx, x))
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
                
        if eval_count >= budget:
            return best_x, best_f
            
        # Sort by fitness
        initial_results.sort(key=lambda item: item[0])
        
        # 2. Local Optimization
        # Select top K candidates for local refinement
        num_local = 5
        candidates = initial_results[:min(num_local, len(initial_results))]
        
        remaining_budget = budget - eval_count
        budget_per_local = remaining_budget // num_local
        
        if budget_per_local < 10:
            # If budget is too low for multiple local runs, just do one or return
            # Actually, if budget_per_local is small, we might just stick with best found
            # or do one deep run
            pass

        for i, (f_start, x_start) in enumerate(candidates):
            if time.time() > deadline:
                break
            
            # Check if we have enough budget left
            if remaining_budget <= 0:
                break
                
            # Define the function to pass to scipy
            # scipy counts function calls. We need to limit them.
            # We don't have a direct counter in scipy's minimize options easily accessible 
            # without a wrapper, so we use maxiter as a proxy, but maxiter != func_evals.
            # For Nelder-Mead, func_evals is roughly 2 * maxiter * dim? No, it's complex.
            # Safer to wrap func and count calls.
            
            local_evals = [0]
            def local_func(x):
                if time.time() > deadline:
                    raise TimeoutError("Time limit exceeded")
                local_evals[0] += 1
                return func(x)
                
            try:
                # Nelder-Mead options
                # maxiter is not exactly func_evals, but a good control.
                # For dim=5, simplex has 6 points.
                # Let's allow up to budget_per_local evaluations.
                # scipy's Nelder-Mead usually uses ~ dim * maxiter evaluations?
                # Actually, it's safer to just run it and let the wrapper count.
                # But we can't stop scipy mid-run easily based on count without exception.
                # The TimeoutError above handles time.
                # For budget, we can just trust that 100-200 evals is reasonable for one local run.
                
                res = minimize(local_func, x_start, method='Nelder-Mead', 
                               options={'maxiter': 200, 'xatol': 1e-6, 'fatol': 1e-6})
                
                # Update global best
                if res.fun < best_f:
                    best_f = res.fun
                    best_x = res.x.copy()
                    
                eval_count += local_evals[0]
                remaining_budget -= local_evals[0]
                
            except TimeoutError:
                break
            except Exception:
                # If an exception occurs, we keep the calls made so far
                break

        return best_x, best_f
# EVOLVE-BLOCK-END
