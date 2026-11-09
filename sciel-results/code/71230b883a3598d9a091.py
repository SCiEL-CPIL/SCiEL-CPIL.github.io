"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """Nelder-Mead Simplex optimizer for Sphere function."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290  # Safety margin well before 330s limit
        
        n = self.dim
        budget = self.budget
        
        # 1. Initial point
        # The problem states the minimizer is in [-4, 4]^5.
        # Starting at 0 is a reasonable prior, but we can use a slightly larger initial simplex
        # to ensure we "see" the gradient direction even if shifted.
        x0 = np.zeros(n)
        
        # Initial Simplex
        # The standard Nelder-Mead algorithm requires n+1 points.
        # We define the initial simplex with points at distance d from x0 along axes.
        # Since the box is [-5, 5] and min is in [-4, 4], a step of 1.0 is safe.
        # A larger step might find the basin faster if the function is very flat,
        # but Sphere is quadratic, so 1.0 is good.
        d = 1.0
        simplex = [x0.copy()]
        for i in range(n):
            p = x0.copy()
            p[i] += d
            simplex.append(p)
        
        # Evaluate initial simplex
        # Cost: n+1 calls
        if budget < n + 1:
            # Fallback if budget is too small (unlikely given budget=500)
            return x0, func(x0)
            
        fvals = np.array([func(p) for p in simplex])
        calls_made = n + 1
        
        # Best so far
        best_idx = np.argmin(fvals)
        x_best = simplex[best_idx].copy()
        f_best = fvals[best_idx]
        
        # Nelder-Mead Parameters
        alpha = 1.0  # Reflection
        gamma = 2.0  # Expansion
        rho = 0.5    # Contraction
        sigma = 0.5  # Shrink
        
        # Max iterations guard (not strictly needed if budget is the limit, but good practice)
        max_iter = 1000
        
        for _ in range(max_iter):
            # Time check
            if time.time() > deadline:
                break
            
            # Budget check
            # Each iteration consumes at least 1 call (reflection).
            # It can consume up to 4 calls (reflection, expansion, contraction, shrink).
            # To be safe, we check if we have at least 1 call left.
            if calls_made >= budget:
                break
                
            # 1. Sort simplex by function value
            # We maintain the list of points and their values.
            # Sorting n+1=6 elements is cheap.
            order = np.argsort(fvals)
            simplex = [simplex[i] for i in order]
            fvals = fvals[order]
            
            x_worst = simplex[-1]
            f_worst = fvals[-1]
            x_best_cur = simplex[0]
            
            # Update global best
            if f_best > fvals[0]:
                f_best = fvals[0]
                x_best = simplex[0].copy()
            
            # 2. Compute centroid of best n points
            centroid = np.sum(simplex[:-1], axis=0) / n
            
            # 3. Reflection
            x_r = centroid + alpha * (centroid - x_worst)
            x_r = np.clip(x_r, -5, 5)
            
            if calls_made >= budget: break
            f_r = func(x_r)
            calls_made += 1
            
            if f_r < fvals[0]:
                # 4. Expansion
                x_e = centroid + gamma * (centroid - x_worst)
                x_e = np.clip(x_e, -5, 5)
                
                if calls_made >= budget: 
                    # Update best if x_r is better
                    if f_r < f_best:
                        f_best = f_r
                        x_best = x_r
                    break
                    
                f_e = func(x_e)
                calls_made += 1
                
                if f_e < f_r:
                    # Replace worst with expansion
                    simplex[-1] = x_e
                    fvals[-1] = f_e
                else:
                    # Replace worst with reflection
                    simplex[-1] = x_r
                    fvals[-1] = f_r
            elif f_r < fvals[-2]:
                # 5. Accept reflection
                simplex[-1] = x_r
                fvals[-1] = f_r
            else:
                # 6. Contraction
                # Outside contraction if f_r >= f_worst, Inside otherwise
                if f_r < f_worst:
                    # Inside contraction
                    x_c = centroid + rho * (x_r - centroid)
                else:
                    # Outside contraction
                    x_c = centroid - rho * (x_worst - centroid)
                    
                x_c = np.clip(x_c, -5, 5)
                
                if calls_made >= budget:
                    # Update best if x_r is better
                    if f_r < f_best:
                        f_best = f_r
                        x_best = x_r
                    break
                    
                f_c = func(x_c)
                calls_made += 1
                
                if f_c < f_r:
                    # Accept contraction
                    simplex[-1] = x_c
                    fvals[-1] = f_c
                else:
                    # 7. Shrink
                    # Move all points except the best towards the best
                    # This requires n calls (n-1 updates + 1 evaluation? No, we just move points.
                    # We need to evaluate the new positions?
                    # Standard NM: We shrink the simplex towards x_best.
                    # We don't necessarily need to re-evaluate all points if we just move them,
                    # but we do need to evaluate the new points to update fvals.
                    # However, usually shrink is done by moving points and keeping fvals?
                    # No, fvals must correspond to the points.
                    # So we need n-1 new evaluations?
                    # That's expensive.
                    # Alternative: Just break if we are stuck?
                    # Or just move the worst point to midpoint with best?
                    # Let's implement standard shrink:
                    # x_i = x_best + sigma * (x_i - x_best) for i=1..n
                    # We need to evaluate these n-1 points.
                    
                    if calls_made + (n - 1) > budget:
                         # Not enough budget for full shrink
                         break
                         
                    for i in range(1, n + 1):
                        simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                        simplex[i] = np.clip(simplex[i], -5, 5)
                        fvals[i] = func(simplex[i])
                        calls_made += 1
                        
                    if calls_made > budget:
                        break

        return x_best, f_best

# EVOLVE-BLOCK-END
