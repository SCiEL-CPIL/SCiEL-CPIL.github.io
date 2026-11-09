import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0
        rng = np.random.default_rng(42)
        n = self.dim
        budget = self.budget
        
        best_x = np.zeros(n)
        best_f = func(best_x)
        calls = 1
        
        # Parameters for Nelder-Mead
        alpha = 1.0
        gamma = 2.0
        rho = 0.5
        sigma = 0.5
        
        # Initial simplex radius
        # Start with a radius that covers the box somewhat
        # But not too large to avoid flat regions
        init_radius = 2.0 
        
        while calls < budget and time.time() < deadline:
            # Create simplex
            simplex = [best_x.copy()]
            fvals = [best_f]
            
            for i in range(n):
                x_i = best_x.copy()
                x_i[i] += init_radius
                x_i = np.clip(x_i, -5, 5)
                f_i = func(x_i)
                calls += 1
                simplex.append(x_i)
                fvals.append(f_i)
                
            if calls >= budget:
                break
                
            # Run Nelder-Mead
            improved = False
            # Limit iterations per restart to avoid wasting budget on one run
            # Max 100 iterations per run?
            for _ in range(100):
                if calls >= budget or time.time() > deadline:
                    break
                    
                # Sort
                idx = np.argsort(fvals)
                simplex = [simplex[i] for i in idx]
                fvals = [fvals[i] for i in idx]
                
                # Update best
                if fvals[0] < best_f:
                    best_f = fvals[0]
                    best_x = simplex[0].copy()
                    improved = True
                    
                # Check convergence
                # If the difference between best and worst is tiny, or simplex size is tiny
                simplex_size = np.max([np.linalg.norm(simplex[0] - s) for s in simplex[1:]])
                if simplex_size < 1e-4:
                    break
                if fvals[-1] - fvals[0] < 1e-8:
                    break
                
                centroid = np.mean(simplex[:-1], axis=0)
                x_worst = simplex[-1]
                f_worst = fvals[-1]
                
                # Reflection
                x_r = centroid + alpha * (centroid - x_worst)
                x_r = np.clip(x_r, -5, 5)
                f_r = func(x_r)
                calls += 1
                
                if f_r < fvals[0]:
                    # Expansion
                    x_e = centroid + gamma * (centroid - x_worst)
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
                    simplex[-1] = x_r
                    fvals[-1] = f_r
                else:
                    # Contraction
                    if f_r < f_worst:
                        # Outside
                        x_c = centroid + rho * (x_r - centroid)
                        x_c = np.clip(x_c, -5, 5)
                        f_c = func(x_c)
                        calls += 1
                        
                        if f_c <= f_r:
                            simplex[-1] = x_c
                            fvals[-1] = f_c
                        else:
                            # Shrink
                            for i in range(1, n + 1):
                                if calls >= budget: break
                                simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                                simplex[i] = np.clip(simplex[i], -5, 5)
                                fvals[i] = func(simplex[i])
                                calls += 1
                    else:
                        # Inside
                        x_c = centroid - rho * (x_worst - centroid)
                        x_c = np.clip(x_c, -5, 5)
                        f_c = func(x_c)
                        calls += 1
                        
                        if f_c < f_worst:
                            simplex[-1] = x_c
                            fvals[-1] = f_c
                        else:
                            # Shrink
                            for i in range(1, n + 1):
                                if calls >= budget: break
                                simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                                simplex[i] = np.clip(simplex[i], -5, 5)
                                fvals[i] = func(simplex[i])
                                calls += 1
                            
            # If we exited the loop without using all budget, and we didn't improve significantly
            # or if we converged, we can restart.
            # If we converged (simplex small) and didn't improve in this run, restart from random.
            # If we improved, restart from best_x with smaller radius?
            # Let's restart from best_x with a smaller radius if we improved, 
            # or from a random point if we didn't.
            
            if calls < budget and time.time() < deadline:
                if improved:
                    # Refine around best
                    init_radius *= 0.5
                    if init_radius < 0.01:
                        break
                    # Restart simplex from best_x
                    # The loop above already created a simplex from best_x.
                    # We just loop back.
                    pass
                else:
                    # No improvement, try random restart
                    # Pick a random point in [-5, 5]
                    best_x = rng.uniform(-5, 5, n)
                    best_f = func(best_x)
                    calls += 1
                    init_radius = 2.0 # Reset radius for random restart
                    
        return best_x, best_f
