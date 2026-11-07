"""Multi-start local search: Sobol global sampling + Nelder-Mead refinement for BBOB f21."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import math
from scipy.stats import qmc


class Optimizer:
    """Multi-start optimizer for BBOB f21 (Gallagher 101 peaks)."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 285.0  # Safety margin before 300s limit
        rng = np.random.default_rng(42)

        dim = self.dim
        budget = self.budget
        best_x = None
        best_f = float('inf')
        calls_made = 0

        def time_left():
            return deadline - time.time()

        def make_call(x):
            nonlocal calls_made, best_x, best_f
            if calls_made >= budget:
                return None
            f = func(x)
            calls_made += 1
            if f < best_f:
                best_f = f
                best_x = x.copy()
            return f

        # Phase 1: Global exploration with Sobol sequence
        # Sobol provides better space-filling properties than LHS or random
        n_global = 40
        if n_global > budget:
            n_global = budget
        
        sobol = qmc.Sobol(d=dim, scramble=True, seed=42)
        # Generate points in [0, 1]^dim
        # Use power of 2 for Sobol efficiency, but adjust to n_global
        m = 1
        while m < n_global:
            m *= 2
        sobol_points = sobol.random(m)
        # Scale to [-5, 5]^dim
        sobol_points = sobol_points * 10.0 - 5.0
        
        # Evaluate first n_global points
        for i in range(n_global):
            if calls_made >= budget or time_left() < 5:
                break
            x = sobol_points[i]
            make_call(x)
            
        if best_x is None:
            # Fallback
            make_call(np.zeros(dim))
            return best_x, best_f

        # Phase 2: Local refinement using Nelder-Mead on top candidates
        # We need to track the best points from Phase 1
        # Since make_call doesn't return history, we re-evaluate or just use the best found
        # A better approach: Run Nelder-Mead starting from the best point found in Phase 1
        # And potentially from a few other good points if budget allows
        
        # For simplicity and robustness, we run Nelder-Mead on the current best_x
        # Nelder-Mead is a simplex method that works well for smooth functions
        
        def nelder_mead(start_x, budget_nm):
            nonlocal calls_made, best_x, best_f
            
            # Initialize simplex
            # Simplex has dim+1 vertices
            # Start with start_x and perturbations
            simplex = [start_x.copy()]
            # Add vertices with small perturbations
            for i in range(dim):
                vertex = start_x.copy()
                # Use a reasonable initial step size
                step = 0.5
                vertex[i] += step
                simplex.append(vertex)
            
            # Evaluate initial simplex
            f_values = []
            for i in range(dim + 1):
                if calls_made >= budget or time_left() < 1:
                    return
                f = make_call(simplex[i])
                if f is None:
                    return
                f_values.append(f)
            
            # Nelder-Mead loop
            while calls_made < budget and time_left() > 1:
                # Sort simplex by function value
                indices = np.argsort(f_values)
                simplex = [simplex[i] for i in indices]
                f_values = [f_values[i] for i in indices]
                
                # Compute centroid of all points except the worst
                centroid = np.mean(simplex[:-1], axis=0)
                
                # Reflection
                alpha = 1.0
                x_r = centroid + alpha * (centroid - simplex[-1])
                x_r = np.clip(x_r, -5.0, 5.0)
                f_r = make_call(x_r)
                if f_r is None:
                    break
                
                if f_r < f_values[0]:
                    # Expansion
                    gamma = 2.0
                    x_e = centroid + gamma * (centroid - simplex[-1])
                    x_e = np.clip(x_e, -5.0, 5.0)
                    f_e = make_call(x_e)
                    if f_e is None:
                        break
                    if f_e < f_r:
                        simplex[-1] = x_e
                        f_values[-1] = f_e
                    else:
                        simplex[-1] = x_r
                        f_values[-1] = f_r
                elif f_r < f_values[-2]:
                    # Reflection accepted
                    simplex[-1] = x_r
                    f_values[-1] = f_r
                else:
                    # Contraction
                    beta = 0.5
                    # Outside contraction
                    if f_r < f_values[-1]:
                        x_c = centroid + beta * (x_r - centroid)
                    else:
                        # Inside contraction
                        x_c = centroid - beta * (x_r - centroid)
                    
                    x_c = np.clip(x_c, -5.0, 5.0)
                    f_c = make_call(x_c)
                    if f_c is None:
                        break
                    
                    if f_c < f_values[-1]:
                        simplex[-1] = x_c
                        f_values[-1] = f_c
                    else:
                        # Shrink
                        for i in range(1, dim + 1):
                            simplex[i] = simplex[0] + 0.5 * (simplex[i] - simplex[0])
                            if calls_made >= budget or time_left() < 1:
                                return
                            f_shrink = make_call(simplex[i])
                            if f_shrink is None:
                                return
                            f_values[i] = f_shrink
                        
                        # Check convergence
                        if np.max(f_values) - np.min(f_values) < 1e-10:
                            break

        # Run Nelder-Mead from the best point found in Phase 1
        # We allocate the remaining budget to this
        if calls_made < budget and time_left() > 5:
            nelder_mead(best_x.copy(), budget - calls_made)

        return best_x, best_f
# EVOLVE-BLOCK-END
