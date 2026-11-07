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
        best_f = float('inf')
        calls = 0

        def call_func(x):
            nonlocal calls, best_x, best_f
            if calls >= budget or time.time() > deadline:
                return best_f
            val = func(x)
            calls += 1
            if val < best_f:
                best_f = val
                best_x = x.copy()
            return val

        # Phase 1: Quick exploration to find a good starting region
        # Try center and a few random points to get a sense of the landscape
        rng = np.random.RandomState(42)
        
        # Evaluate center
        x0 = np.zeros(n)
        f0 = call_func(x0)
        
        if time.time() > deadline or calls >= budget:
            return best_x, best_f

        # Try a few points spread across the domain
        for _ in range(min(5, budget - calls - 10)):
            if time.time() > deadline or calls >= budget:
                break
            x_try = rng.uniform(-4, 4, size=n)
            f_try = call_func(x_try)
        
        if time.time() > deadline or calls >= budget:
            return best_x, best_f

        # Phase 2: Use Nelder-Mead simplex method (derivative-free, good for quadratics)
        # Start from best point found so far
        x = best_x.copy()
        f = best_f
        
        # Create initial simplex
        step_size = 0.5
        simplex = [x.copy()]
        for i in range(n):
            new_point = x.copy()
            new_point[i] += step_size
            simplex.append(new_point)
        
        # Evaluate simplex points
        simplex_f = []
        for i, sp in enumerate(simplex):
            if i == 0:
                simplex_f.append(f)
            else:
                if time.time() > deadline or calls >= budget:
                    break
                sp_clipped = np.clip(sp, -5.0, 5.0)
                simplex_f.append(call_func(sp_clipped))
                simplex[i] = sp_clipped
        
        if len(simplex_f) < n + 1:
            # Not enough points, just return best
            return best_x, best_f

        # Nelder-Mead iterations
        alpha = 1.0   # reflection
        gamma = 2.0   # expansion
        rho = 0.5     # contraction
        sigma = 0.5   # shrink
        
        max_nelder_iters = 200
        for n_iter in range(max_nelder_iters):
            if time.time() > deadline or calls >= budget:
                break
            
            # Sort simplex by function value
            order = np.argsort(simplex_f)
            simplex = [simplex[i] for i in order]
            simplex_f = [simplex_f[i] for i in order]
            
            # Check convergence
            if abs(simplex_f[-1] - simplex_f[0]) < 1e-10:
                break
            
            # Centroid of all but worst point
            centroid = np.mean(simplex[:-1], axis=0)
            
            # Reflection
            x_r = centroid + alpha * (centroid - simplex[-1])
            x_r = np.clip(x_r, -5.0, 5.0)
            f_r = call_func(x_r)
            if time.time() > deadline or calls >= budget:
                break
            
            if f_r < simplex_f[0]:
                # Try expansion
                x_e = centroid + gamma * (x_r - centroid)
                x_e = np.clip(x_e, -5.0, 5.0)
                f_e = call_func(x_e)
                if time.time() > deadline or calls >= budget:
                    break
                
                if f_e < f_r:
                    simplex[-1] = x_e
                    simplex_f[-1] = f_e
                else:
                    simplex[-1] = x_r
                    simplex_f[-1] = f_r
            elif f_r < simplex_f[-2]:
                # Accept reflection
                simplex[-1] = x_r
                simplex_f[-1] = f_r
            else:
                # Contraction
                x_c = centroid + rho * (simplex[-1] - centroid)
                x_c = np.clip(x_c, -5.0, 5.0)
                f_c = call_func(x_c)
                if time.time() > deadline or calls >= budget:
                    break
                
                if f_c < simplex_f[-1]:
                    simplex[-1] = x_c
                    simplex_f[-1] = f_c
                else:
                    # Shrink
                    for i in range(1, len(simplex)):
                        simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                        simplex[i] = np.clip(simplex[i], -5.0, 5.0)
                        if time.time() > deadline or calls >= budget:
                            break
                        simplex_f[i] = call_func(simplex[i])
                    if time.time() > deadline or calls >= budget:
                        break
        
        # Phase 3: Fine-tuning with small perturbations if budget remains
        if calls < budget - 5 and time.time() < deadline:
            # Try small random perturbations around best_x
            remaining = min(50, budget - calls)
            for _ in range(remaining):
                if time.time() > deadline or calls >= budget:
                    break
                perturbation = rng.randn(n) * 0.01
                x_try = np.clip(best_x + perturbation, -5.0, 5.0)
                f_try = call_func(x_try)
        
        return best_x, best_f
