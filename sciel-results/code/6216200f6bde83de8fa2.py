import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        n = self.dim
        budget = self.budget
        deadline = start_time + 280
        
        best_x = np.zeros(n)
        best_f = np.inf
        calls = 0
        
        # Phase 1: Global Random Search
        # Evaluate a few random points to find a good starting basin
        n_global = min(20, budget // 5)
        global_points = np.random.uniform(-5, 5, size=(n_global, n))
        
        for i in range(n_global):
            if calls >= budget or time.time() > deadline:
                break
            x = global_points[i]
            f = func(x)
            calls += 1
            if f < best_f:
                best_f = f
                best_x = x.copy()
        
        # Phase 2: Nelder-Mead Local Optimization
        # Start from best_x
        
        # Choose initial step size
        step = 2.0
        
        # Build simplex
        simplex = np.zeros((n + 1, n))
        simplex[0] = best_x
        for i in range(n):
            simplex[i + 1] = best_x
            simplex[i + 1][i] += step
            
        # Evaluate simplex
        simplex_f = np.zeros(n + 1)
        for i in range(n + 1):
            if calls >= budget or time.time() > deadline:
                break
            simplex_f[i] = func(simplex[i])
            calls += 1
            if simplex_f[i] < best_f:
                best_f = simplex_f[i]
                best_x = simplex[i].copy()
        
        # Nelder-Mead Parameters
        alpha = 1.0
        gamma = 2.0
        rho = 0.5
        sigma = 0.5
        
        # Main NM Loop
        stagnation_counter = 0
        last_best_f = best_f
        
        while calls < budget:
            if time.time() > deadline:
                break
            
            # Sort simplex by function value
            idx = np.argsort(simplex_f)
            simplex = simplex[idx]
            simplex_f = simplex_f[idx]
            
            # Check convergence
            # If the spread of function values is very small, stop
            if np.max(simplex_f) - np.min(simplex_f) < 1e-12:
                break
            # Also check geometric convergence
            if np.linalg.norm(simplex[0] - simplex[-1]) < 1e-12:
                break
                
            # Stagnation check: if best_f hasn't improved in 50 iterations, perturb
            if best_f == last_best_f:
                stagnation_counter += 1
            else:
                stagnation_counter = 0
                last_best_f = best_f
                
            if stagnation_counter > 50:
                # Perturb the simplex
                simplex = simplex + np.random.randn(n + 1, n) * 0.1
                for i in range(n + 1):
                    if calls >= budget or time.time() > deadline:
                        break
                    simplex_f[i] = func(simplex[i])
                    calls += 1
                    if simplex_f[i] < best_f:
                        best_f = simplex_f[i]
                        best_x = simplex[i].copy()
                stagnation_counter = 0
                last_best_f = best_f
                continue

            # Centroid of all points except the worst
            x_c = np.mean(simplex[:-1], axis=0)
            
            # Reflection
            x_r = x_c + alpha * (x_c - simplex[-1])
            if calls >= budget or time.time() > deadline:
                break
            f_r = func(x_r)
            calls += 1
            if f_r < best_f:
                best_f = f_r
                best_x = x_r.copy()
            
            if f_r < simplex_f[0]:
                # Expansion
                if calls >= budget or time.time() > deadline:
                    break
                x_e = x_c + gamma * (x_r - x_c)
                f_e = func(x_e)
                calls += 1
                if f_e < best_f:
                    best_f = f_e
                    best_x = x_e.copy()
                
                if f_e < f_r:
                    simplex[-1] = x_e
                    simplex_f[-1] = f_e
                else:
                    simplex[-1] = x_r
                    simplex_f[-1] = f_r
            elif f_r < simplex_f[-2]:
                # Simple Reflection
                simplex[-1] = x_r
                simplex_f[-1] = f_r
            else:
                # Contraction
                if f_r < simplex_f[-1]:
                    # Outer Contraction
                    if calls >= budget or time.time() > deadline:
                        break
                    x_c2 = x_c + rho * (x_r - x_c)
                    f_c = func(x_c2)
                    calls += 1
                    if f_c < best_f:
                        best_f = f_c
                        best_x = x_c2.copy()
                    
                    if f_c < f_r:
                        simplex[-1] = x_c2
                        simplex_f[-1] = f_c
                    else:
                        # Shrink
                        for i in range(1, n + 1):
                            simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                            if calls >= budget or time.time() > deadline:
                                break
                            simplex_f[i] = func(simplex[i])
                            calls += 1
                            if simplex_f[i] < best_f:
                                best_f = simplex_f[i]
                                best_x = simplex[i].copy()
                else:
                    # Inner Contraction
                    if calls >= budget or time.time() > deadline:
                        break
                    x_c2 = x_c - rho * (x_c - simplex[-1])
                    f_c = func(x_c2)
                    calls += 1
                    if f_c < best_f:
                        best_f = f_c
                        best_x = x_c2.copy()
                    
                    if f_c < simplex_f[-1]:
                        simplex[-1] = x_c2
                        simplex_f[-1] = f_c
                    else:
                        # Shrink
                        for i in range(1, n + 1):
                            simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                            if calls >= budget or time.time() > deadline:
                                break
                            simplex_f[i] = func(simplex[i])
                            calls += 1
                            if simplex_f[i] < best_f:
                                best_f = simplex_f[i]
                                best_x = simplex[i].copy()
            
        return best_x, best_f
