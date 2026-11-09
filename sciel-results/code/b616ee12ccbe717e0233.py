"""Sobol Quasi-Random Survey + Adaptive Nelder-Mead for Schaffers F7."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import qmc

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290  # Safety margin for 300s limit
        
        d = self.dim
        budget = self.budget
        
        # State tracking
        best_x_global = None
        best_f_global = float('inf')
        calls = 0
        
        def check_time():
            return time.time() < deadline

        # Phase 1: Sobol Quasi-Random Survey
        # We use a deterministic Sobol sequence to cover the space [-5, 5]^d uniformly.
        # This is superior to random sampling for early exploration as it avoids clustering.
        # We allocate roughly 20% of the budget to this phase to establish a good starting point.
        # For d=5, a Sobol sequence of size 100 gives decent coverage.
        
        # Determine number of survey points. 
        # Let's use 40 points for survey (8% of budget) to leave plenty for exploitation.
        # Actually, for AOCC, early progress is key. A good start helps immediately.
        # Let's try 50 survey points.
        n_survey = 50
        
        # Generate Sobol points in [0, 1]^d
        sampler = qmc.Sobol(d=d, scramble=True)
        # We need at least 2^m points. 64 is the next power of 2 >= 50.
        # But we can just generate 64 and use the first 50, or generate exactly if we adjust.
        # Sobol requires power of 2 for standard usage, but we can generate more and subsample.
        # Let's generate 64 points to be safe and use the first 50.
        m = 6
        n_points = 2**m
        points = sampler.random(n_points)
        
        # Map from [0, 1]^d to [-5, 5]^d
        points_scaled = points * 10.0 - 5.0
        
        # Evaluate the first n_survey points
        for i in range(n_survey):
            if not check_time() or calls >= budget:
                break
            
            x = points_scaled[i]
            fx = func(x)
            calls += 1
            
            if fx < best_f_global:
                best_f_global = fx
                best_x_global = x.copy()
        
        # If we ran out of budget or time during survey, return best found
        if not check_time() or calls >= budget:
            return best_x_global, best_f_global
        
        # Phase 2: Adaptive Nelder-Mead Local Exploitation
        # We start the local search from the best point found in the survey.
        # We implement a simple Nelder-Mead with adaptive step size (shrink/grow).
        
        x_current = best_x_global.copy()
        f_current = best_f_global
        
        # Initialize simplex
        # Standard NM simplex: n+1 points
        # Initial step size
        step = 0.5
        
        # Create initial simplex
        simplex = [x_current.copy()]
        f_simplex = [f_current]
        
        for i in range(d):
            x_new = x_current.copy()
            x_new[i] += step
            # Clip to bounds
            x_new = np.clip(x_new, -5, 5)
            f_new = func(x_new)
            calls += 1
            
            simplex.append(x_new)
            f_simplex.append(f_new)
            
            if f_new < best_f_global:
                best_f_global = f_new
                best_x_global = x_new.copy()
        
        # Main NM Loop
        # We will run until budget or time is exhausted.
        # Standard NM operations: Reflection, Expansion, Contraction, Shrink
        
        while check_time() and calls < budget:
            # Sort simplex by fitness
            idx = np.argsort(f_simplex)
            simplex = [simplex[i] for i in idx]
            f_simplex = [f_simplex[i] for i in idx]
            
            # Centroid of all points except the worst
            centroid = np.mean(simplex[:-1], axis=0)
            
            # Worst point
            x_worst = simplex[-1]
            f_worst = f_simplex[-1]
            
            # Reflection
            alpha = 1.0
            x_ref = centroid + alpha * (centroid - x_worst)
            x_ref = np.clip(x_ref, -5, 5)
            f_ref = func(x_ref)
            calls += 1
            
            if f_ref < best_f_global:
                best_f_global = f_ref
                best_x_global = x_ref.copy()
            
            # Check if reflection is good
            f_best = f_simplex[0]
            f_second = f_simplex[1]
            
            if f_ref < f_second:
                # Reflection is good, try expansion
                gamma = 2.0
                x_exp = centroid + gamma * (centroid - x_worst)
                x_exp = np.clip(x_exp, -5, 5)
                f_exp = func(x_exp)
                calls += 1
                
                if f_exp < best_f_global:
                    best_f_global = f_exp
                    best_x_global = x_exp.copy()
                
                if f_exp < f_ref:
                    # Accept expansion
                    simplex[-1] = x_exp
                    f_simplex[-1] = f_exp
                else:
                    # Accept reflection
                    simplex[-1] = x_ref
                    f_simplex[-1] = f_ref
            else:
                # Reflection is not better than second best
                if f_ref < f_worst:
                    # Accept reflection (intermediate contraction)
                    simplex[-1] = x_ref
                    f_simplex[-1] = f_ref
                else:
                    # Contraction
                    rho = 0.5
                    x_con = centroid + rho * (x_worst - centroid)
                    x_con = np.clip(x_con, -5, 5)
                    f_con = func(x_con)
                    calls += 1
                    
                    if f_con < best_f_global:
                        best_f_global = f_con
                        best_x_global = x_con.copy()
                    
                    if f_con < f_worst:
                        # Accept contraction
                        simplex[-1] = x_con
                        f_simplex[-1] = f_con
                    else:
                        # Shrink
                        # Move all points except best towards best
                        for i in range(1, len(simplex)):
                            simplex[i] = simplex[0] + 0.5 * (simplex[i] - simplex[0])
                            simplex[i] = np.clip(simplex[i], -5, 5)
                            f_shrink = func(simplex[i])
                            calls += 1
                            f_simplex[i] = f_shrink
                            
                            if f_shrink < best_f_global:
                                best_f_global = f_shrink
                                best_x_global = simplex[i].copy()
                        
                        # If we ran out of budget during shrink, break
                        if not check_time() or calls >= budget:
                            break
                        
                        # Reduce step size if we are shrinking repeatedly? 
                        # For now, standard NM shrink handles this.
                        # However, if the simplex becomes very small, we might stagnate.
                        # We can monitor the range of fitness values.
                        if max(f_simplex) - min(f_simplex) < 1e-10:
                            # Stagnation detected in local search.
                            # We could restart NM from a random point near best, 
                            # or just break and use remaining budget for random perturbations.
                            # Let's try a few random perturbations from the best point.
                            for _ in range(5):
                                if not check_time() or calls >= budget:
                                    break
                                x_pert = best_x_global + 0.1 * np.random.standard_normal(d)
                                x_pert = np.clip(x_pert, -5, 5)
                                f_pert = func(x_pert)
                                calls += 1
                                if f_pert < best_f_global:
                                    best_f_global = f_pert
                                    best_x_global = x_pert.copy()
                            break

        return best_x_global, best_f_global
# EVOLVE-BLOCK-END
