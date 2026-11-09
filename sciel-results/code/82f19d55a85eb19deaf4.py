"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Nelder-Mead simplex method with coordinate descent polish."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0  # 10s safety margin before 300s limit
        dim = self.dim
        budget = self.budget
        
        # Initialize simplex
        # Start at a reasonable point, e.g., 0 or random
        # Rosenbrock minimum is usually at (1, 1, ...) but here it's shifted.
        # A uniform start in the box is safer.
        rng = np.random.default_rng(0)
        x0 = rng.uniform(-5, 5, dim)
        
        # Build initial simplex
        # Vertices: x0, x0 + delta * e_i
        delta = 0.5
        simplex = np.zeros((dim + 1, dim))
        simplex[0] = x0
        for i in range(dim):
            simplex[i + 1] = x0
            simplex[i + 1][i] += delta
            
        # Evaluate initial simplex
        fvals = np.array([func(simplex[i]) for i in range(dim + 1)])
        calls = dim + 1
        
        # Nelder-Mead parameters
        alpha = 1.0   # Reflection
        gamma = 2.0   # Expansion
        rho = 0.5     # Contraction
        sigma = 0.5   # Shrink
        
        # Sort initial simplex
        idx = np.argsort(fvals)
        simplex = simplex[idx]
        fvals = fvals[idx]
        
        while calls < budget:
            if time.time() > deadline:
                break
                
            # Check convergence
            # If the spread of function values is very small, we are likely converged
            if np.ptp(fvals) < 1e-10:
                break
                
            # Sort simplex by function value
            idx = np.argsort(fvals)
            simplex = simplex[idx]
            fvals = fvals[idx]
            
            x_worst = simplex[-1]
            f_worst = fvals[-1]
            x_best = simplex[0]
            f_best = fvals[0]
            
            # Centroid of all points except the worst
            centroid = np.mean(simplex[:-1], axis=0)
            
            # Reflection
            x_r = centroid + alpha * (centroid - x_worst)
            x_r = np.clip(x_r, -5, 5)
            f_r = func(x_r)
            calls += 1
            
            if f_r < f_best:
                # Try expansion
                x_e = centroid + gamma * (x_r - centroid)
                x_e = np.clip(x_e, -5, 5)
                f_e = func(x_e)
                calls += 1
                
                if f_e < f_r:
                    simplex[-1] = x_e
                    fvals[-1] = f_e
                else:
                    simplex[-1] = x_r
                    fvals[-1] = f_r
            elif f_r < fvals[-2]:
                # Accept reflection
                simplex[-1] = x_r
                fvals[-1] = f_r
            else:
                # Contraction
                # Outside contraction if f_r < f_worst, else inside
                if f_r < f_worst:
                    x_c = centroid + rho * (x_r - centroid)
                    type_c = 'outside'
                else:
                    x_c = centroid - rho * (x_worst - centroid)
                    type_c = 'inside'
                
                x_c = np.clip(x_c, -5, 5)
                f_c = func(x_c)
                calls += 1
                
                if f_c < f_worst:
                    simplex[-1] = x_c
                    fvals[-1] = f_c
                else:
                    # Shrink
                    for i in range(1, dim + 1):
                        simplex[i] = x_best + sigma * (simplex[i] - x_best)
                        simplex[i] = np.clip(simplex[i], -5, 5)
                        fvals[i] = func(simplex[i])
                        calls += 1
                        if calls >= budget:
                            break
                    if calls >= budget:
                        break
        
        # Polish phase: Coordinate Descent
        # Start from the best point found
        idx_best = np.argmin(fvals)
        x_best = simplex[idx_best].copy()
        f_best = fvals[idx_best]
        
        # Coordinate descent with adaptive step
        step = 1.0
        while calls < budget:
            if time.time() > deadline:
                break
            
            improved = False
            for i in range(dim):
                if calls >= budget:
                    break
                
                # Try positive step
                x_trial = x_best.copy()
                x_trial[i] += step
                x_trial = np.clip(x_trial, -5, 5)
                f_trial = func(x_trial)
                calls += 1
                
                if f_trial < f_best:
                    x_best = x_trial
                    f_best = f_trial
                    improved = True
                    # Try to continue in same direction (optional, but adds calls)
                    # For simplicity, just restart the loop or continue to next dim
                else:
                    # Try negative step
                    x_trial = x_best.copy()
                    x_trial[i] -= step
                    x_trial = np.clip(x_trial, -5, 5)
                    f_trial = func(x_trial)
                    calls += 1
                    
                    if f_trial < f_best:
                        x_best = x_trial
                        f_best = f_trial
                        improved = True
            
            if not improved:
                # Reduce step size
                step *= 0.5
                if step < 1e-6:
                    break
        
        return x_best, f_best
# EVOLVE-BLOCK-END
