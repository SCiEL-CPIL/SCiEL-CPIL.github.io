"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import numpy as np


class Optimizer:
    """(1+1)-ES with 1/5 success rule + Nelder-Mead polish."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start = time.time()
        deadline = start + 290.0  # safety margin before 300s limit
        rng = np.random.default_rng(0)
        
        # Phase 1: (1+1)-ES for global exploration and basin finding
        # Start at origin as suggested by f14 properties (min near origin)
        x = np.zeros(self.dim)
        fx = func(x)
        best_x = x.copy()
        best_f = fx
        sigma = 2.0
        budget = self.budget
        dim = self.dim
        
        # Run (1+1)-ES for the first ~70% of budget
        # This ensures early progress which is critical for AOCC
        phase1_budget = int(budget * 0.7)
        for i in range(phase1_budget):
            if time.time() > deadline:
                break
            y = np.clip(x + sigma * rng.standard_normal(dim), -5, 5)
            fy = func(y)
            if fy < fx:
                x, fx = y, fy
                sigma *= 1.2
                if fx < best_f:
                    best_f = fx
                    best_x = x.copy()
            else:
                sigma *= 0.9
                # Prevent sigma from becoming too small
                if sigma < 1e-8:
                    sigma = 1e-8

        # Phase 2: Nelder-Mead polish for local refinement
        # Use remaining budget for NM
        remaining_budget = budget - phase1_budget
        
        # Initialize simplex around the best point found by (1+1)-ES
        center = best_x.copy()
        simplex = [(center.copy(), best_f)]
        
        # Add vertices with a small step size
        # Since (1+1)-ES has likely converged to a small region, we use a small step
        step = 0.5 
        for i in range(dim):
            if time.time() > deadline:
                break
            pt = center.copy()
            pt[i] += step
            fval = func(pt)
            simplex.append((pt.copy(), fval))
            if fval < best_f:
                best_f = fval
                best_x = pt.copy()
        
        # Sort simplex
        simplex.sort(key=lambda p: p[1])
        
        # Nelder-Mead parameters
        alpha = 1.0
        gamma = 2.0
        rho = 0.5
        
        # Run NM for the remaining budget
        # We need to track evaluations carefully
        evals_used = phase1_budget + dim + 1  # +1 for initial center
        nm_evals = 0
        
        while nm_evals < remaining_budget:
            if time.time() > deadline:
                break
            
            # Get best, worst, second worst
            x_worst, f_worst = simplex[-1]
            x_second, f_second = simplex[-2]
            
            # Centroid of all but worst
            x_centroid = np.mean([p[0] for p in simplex[:-1]], axis=0)
            
            # Reflection
            x_ref = x_centroid + alpha * (x_centroid - x_worst)
            f_ref = func(x_ref)
            nm_evals += 1
            if f_ref < best_f:
                best_f = f_ref
                best_x = x_ref.copy()
                
            if f_ref < simplex[0][1]: # Better than best
                # Expansion
                x_exp = x_centroid + gamma * (x_centroid - x_worst)
                f_exp = func(x_exp)
                nm_evals += 1
                if f_exp < best_f:
                    best_f = f_exp
                    best_x = x_exp.copy()
                if f_exp < f_ref:
                    simplex[-1] = (x_exp, f_exp)
                else:
                    simplex[-1] = (x_ref, f_ref)
            elif f_ref < f_second: # Between best and second worst
                simplex[-1] = (x_ref, f_ref)
            else: # Worse than second worst
                # Contraction
                if f_ref < f_worst:
                    # Outer contraction
                    x_con = x_centroid + 0.5 * (x_ref - x_centroid)
                else:
                    # Inner contraction
                    x_con = x_centroid - 0.5 * (x_worst - x_centroid)
                    
                f_con = func(x_con)
                nm_evals += 1
                if f_con < best_f:
                    best_f = f_con
                    best_x = x_con.copy()
                    
                if f_con < min(f_ref, f_worst):
                    simplex[-1] = (x_con, f_con)
                else:
                    # Shrink
                    x_best, f_best_cur = simplex[0]
                    new_simplex = []
                    for i, (x_i, f_i) in enumerate(simplex):
                        if i == 0:
                            new_simplex.append((x_i, f_i))
                        else:
                            x_shrink = x_best + rho * (x_i - x_best)
                            f_shrink = func(x_shrink)
                            nm_evals += 1
                            if f_shrink < best_f:
                                best_f = f_shrink
                                best_x = x_shrink.copy()
                            new_simplex.append((x_shrink, f_shrink))
                    simplex = new_simplex
                    
            # Re-sort
            simplex.sort(key=lambda p: p[1])
            
            # Convergence check: if simplex is very small, break
            x_min = np.min([p[0] for p in simplex], axis=0)
            x_max = np.max([p[0] for p in simplex], axis=0)
            if np.max(x_max - x_min) < 1e-8:
                break

        return best_x, best_f
# EVOLVE-BLOCK-END
