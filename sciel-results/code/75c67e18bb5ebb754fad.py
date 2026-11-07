"""Rewritten program: Nelder-Mead Simplex with Random Restarts."""
# EVOLVE-BLOCK-START
import numpy as np


class Optimizer:
    """Nelder-Mead optimizer with random restarts."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        rng = np.random.default_rng(42)
        d = self.dim
        budget = self.budget
        calls = 0
        
        global_best_x = None
        global_best_f = np.inf
        
        # Parameters for Nelder-Mead
        alpha = 1.0  # Reflection
        gamma = 2.0  # Expansion
        rho = 0.5    # Contraction
        sigma = 0.5  # Shrink
        
        # Max calls per restart
        max_calls_per_restart = 100
        
        while calls < budget:
            # Initialize simplex
            # Start with a random center
            center = rng.uniform(-5, 5, d)
            simplex = []
            fitnesses = []
            
            # First point
            x0 = np.clip(center, -5, 5)
            f0 = func(x0)
            calls += 1
            simplex.append(x0)
            fitnesses.append(f0)
            
            # Other points: center + delta * e_i
            delta = 0.5
            for i in range(d):
                xi = center.copy()
                xi[i] += delta
                xi = np.clip(xi, -5, 5)
                fi = func(xi)
                calls += 1
                simplex.append(xi)
                fitnesses.append(fi)
                
                if calls >= budget:
                    break
            
            if calls >= budget:
                break
            
            # Sort simplex by fitness
            indices = np.argsort(fitnesses)
            simplex = [simplex[i] for i in indices]
            fitnesses = [fitnesses[i] for i in indices]
            
            # Update global best
            if fitnesses[0] < global_best_f:
                global_best_f = fitnesses[0]
                global_best_x = simplex[0].copy()
            
            # Nelder-Mead Loop
            while calls < budget:
                # Check convergence
                if len(fitnesses) > 0:
                    f_diff = fitnesses[-1] - fitnesses[0]
                    if f_diff < 1e-8:
                        break
                
                # Identify worst, second worst, best
                x_worst = simplex[-1]
                f_worst = fitnesses[-1]
                x_second_worst = simplex[-2]
                f_second_worst = fitnesses[-2]
                x_best = simplex[0]
                f_best = fitnesses[0]
                
                # Centroid of all points except worst
                centroid = np.sum(simplex[:-1], axis=0) / d
                
                # Reflection
                x_r = centroid + alpha * (centroid - x_worst)
                x_r = np.clip(x_r, -5, 5)
                f_r = func(x_r)
                calls += 1
                
                if f_r < f_best:
                    # Expansion
                    x_e = centroid + gamma * (centroid - x_worst)
                    x_e = np.clip(x_e, -5, 5)
                    f_e = func(x_e)
                    calls += 1
                    
                    if f_e < f_r:
                        simplex[-1] = x_e
                        fitnesses[-1] = f_e
                    else:
                        simplex[-1] = x_r
                        fitnesses[-1] = f_r
                elif f_r < f_second_worst:
                    # Accept reflection
                    simplex[-1] = x_r
                    fitnesses[-1] = f_r
                else:
                    # Contraction
                    if f_r < f_worst:
                        # Outside contraction
                        x_c = centroid + rho * (x_r - centroid)
                    else:
                        # Inside contraction
                        x_c = centroid - rho * (centroid - x_worst)
                        
                    x_c = np.clip(x_c, -5, 5)
                    f_c = func(x_c)
                    calls += 1
                    
                    if f_c < min(f_r, f_worst):
                        simplex[-1] = x_c
                        fitnesses[-1] = f_c
                    else:
                        # Shrink
                        for i in range(1, d + 1):
                            x_s = x_best + sigma * (simplex[i] - x_best)
                            x_s = np.clip(x_s, -5, 5)
                            f_s = func(x_s)
                            calls += 1
                            simplex[i] = x_s
                            fitnesses[i] = f_s
                        
                        if calls >= budget:
                            break
                        
                        # Re-sort after shrink
                        indices = np.argsort(fitnesses)
                        simplex = [simplex[i] for i in indices]
                        fitnesses = [fitnesses[i] for i in indices]
                        
                        if fitnesses[0] < global_best_f:
                            global_best_f = fitnesses[0]
                            global_best_x = simplex[0].copy()
                        
                        # Check if simplex is too small
                        if np.max(fitnesses) - np.min(fitnesses) < 1e-10:
                            break
                
                # Re-sort after update
                indices = np.argsort(fitnesses)
                simplex = [simplex[i] for i in indices]
                fitnesses = [fitnesses[i] for i in indices]
                
                if fitnesses[0] < global_best_f:
                    global_best_f = fitnesses[0]
                    global_best_x = simplex[0].copy()
                    
            # Restart logic: if we exited the inner loop without using all budget,
            # we might restart. But to be safe, we just continue the outer loop
            # which creates a new simplex.
            # To avoid infinite loops of tiny improvements, we can check progress.
            # For now, just restart if budget remains.
            
        return global_best_x, global_best_f
# EVOLVE-BLOCK-END
