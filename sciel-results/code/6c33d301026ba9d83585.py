import numpy as np
import time
from scipy.optimize import minimize

class Optimizer:
    """
    Grid-based multi-start strategy for BBOB f9 (Rosenbrock rotated).
    
    Strategy:
    1. Deterministic Sparse Grid: Evaluate a 3^5 grid (243 points) covering [-5, 5]^5.
       This ensures robust basin capture without stochastic variance.
    2. Basin Selection: Identify the top 5 lowest-fitness points from the grid.
    3. Local Refinement: Dedicate the remaining budget (257 calls) to independent,
       aggressive Nelder-Mead local optimizations initialized at these specific grid points.
    
    This avoids the overhead and potential instability of CMA-ES in low dimensions
    by leveraging deterministic global coverage and proven local heuristics.
    """
    
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        t_start = time.time()
        deadline = t_start + 320  # Safety margin for 330s kill switch
        
        n = self.dim
        best_f = float('inf')
        best_x = np.zeros(n)
        calls_made = 0
        
        # Phase 1: Sparse Grid Search (3^5 = 243 points)
        # Use 3 points per dimension: -4.0, 0.0, 4.0 (within [-5,5], avoids boundary clipping issues mostly)
        # Actually, to be safe and cover the box well, use -4, 0, 4.
        # The minimizer is in [-4, 4]^5, so this grid is well-centered.
        
        grid_points = []
        # Generate 3^5 grid
        coords = np.array([-4.0, 0.0, 4.0])
        for i in range(3):
            for j in range(3):
                for k in range(3):
                    for l in range(3):
                        for m in range(3):
                            grid_points.append([coords[i], coords[j], coords[k], coords[l], coords[m]])
        
        grid_points = np.array(grid_points)
        
        # Evaluate grid
        grid_fitness = []
        for p in grid_points:
            if calls_made >= self.budget or time.time() > deadline:
                break
            fp = func(p)
            calls_made += 1
            grid_fitness.append(fp)
            
            if fp < best_f:
                best_f = fp
                best_x = p.copy()
        
        if calls_made >= self.budget or time.time() > deadline:
            return best_x, best_f
            
        # Sort grid points by fitness
        grid_fitness = np.array(grid_fitness)
        order = np.argsort(grid_fitness)
        
        # Select top 5 distinct basins
        # Simple heuristic: take the top 5. If they are too close, we might waste time,
        # but Nelder-Mead is cheap. To ensure diversity, we could check distance,
        # but for 5D and 243 points, top 5 are likely in distinct or adjacent valleys.
        # Let's just take the top 5 unique indices.
        
        top_indices = order[:5]
        
        # Phase 2: Nelder-Mead Local Optimization
        # Remaining budget: self.budget - calls_made
        # Distribute remaining budget among the 5 starts.
        
        remaining_calls = self.budget - calls_made
        if remaining_calls <= 0:
            return best_x, best_f
            
        # Allocate roughly equal budget to each start, but cap per-start to allow
        # Nelder-Mead to converge. Nelder-Mead typically needs O(N^2) to O(N^3) evals.
        # For N=5, ~25-125 evals per start is reasonable.
        # 5 starts * 50 evals = 250 evals. We have ~257.
        
        calls_per_start = max(1, int(remaining_calls / 5))
        
        for idx in top_indices:
            if calls_made >= self.budget or time.time() > deadline:
                break
                
            x0 = grid_points[idx].copy()
            f0 = grid_fitness[idx]
            
            # If we already found a very good solution, we can be more aggressive or skip
            # if f0 is already close to expected optimum (but we don't know f_opt).
            
            # Use Nelder-Mead with a custom callback to track calls and enforce deadline
            nm_calls = [0]
            
            def nm_callback(xk):
                nonlocal best_f, best_x
                nm_calls[0] += 1
                fk = func(xk)
                if fk < best_f:
                    best_f = fk
                    best_x = xk.copy()
                
                # Check budget and time
                if calls_made + nm_calls[0] >= self.budget or time.time() > deadline:
                    raise ValueError("Budget or time exceeded")
            
            # Nelder-Mead options
            # maxiter is not strictly eval count, but maxfev is.
            # However, scipy's minimize with method='Nelder-Mead' uses maxiter.
            # We need to map our call budget to iterations.
            # Each iteration of NM in scipy does ~N+1 evaluations? No, it does a sequence of operations.
            # It's safer to use a maxfev-like constraint if available, but NM in scipy
            # doesn't expose maxfev directly in the same way.
            # We will rely on the callback to stop it.
            
            try:
                # Set initial simplex size to be small relative to grid spacing
                # Grid spacing is 4.0. Let's start with a step of 0.5
                res = minimize(
                    func, 
                    x0, 
                    method='Nelder-Mead', 
                    options={
                        'maxiter': 200, # High enough to be stopped by callback
                        'xatol': 1e-4,
                        'fatol': 1e-4,
                        'adaptive': True
                    },
                    callback=nm_callback
                )
                
                # Update best from result if it's better
                if res.fun < best_f:
                    best_f = res.fun
                    best_x = res.x.copy()
                    
            except ValueError:
                # Budget or time exceeded, break out
                break
            except Exception:
                # If NM fails for some reason, continue to next start
                continue
            
            # Update total calls
            calls_made += nm_calls[0]
            
        return best_x, best_f
