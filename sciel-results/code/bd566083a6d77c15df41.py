import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0
        dim = self.dim
        budget = self.budget
        
        best_x = None
        best_fx = np.inf
        n_calls = 0

        def evaluate(x):
            nonlocal best_x, best_fx, n_calls
            x = np.clip(x, -5, 5)
            fx = func(x)
            n_calls += 1
            if fx < best_fx:
                best_fx = fx
                best_x = x.copy()
            return fx

        # Phase 1: Global search with random restarts of Nelder-Mead
        # This is a structural change from CMA-ES to a direct method with restarts.
        # Nelder-Mead is often effective on low-dimensional, non-convex problems like BBOB f7.
        
        # Initial broad sampling
        n_init = 20
        for _ in range(n_init):
            if n_calls >= budget: break
            x = np.random.uniform(-5, 5, dim)
            evaluate(x)
        
        if best_x is None:
            return np.zeros(dim), 0.0

        # Main loop: Random restarts of Nelder-Mead
        # We use a simple manual Nelder-Mead implementation to control the budget precisely.
        
        def nelder_mead(func, x0, max_iter, max_feval):
            # Initialize simplex
            simplex = [x0.copy()]
            f0 = func(x0)
            simplex_f = [f0]
            
            # Create other vertices
            for i in range(dim):
                xi = x0.copy()
                # Step size heuristic
                if xi[i] != 0:
                    xi[i] += 0.05 * abs(xi[i])
                else:
                    xi[i] += 0.05
                simplex.append(xi)
                simplex_f.append(func(xi))
                
            fevals = dim + 1
            alpha = 1.0
            gamma = 2.0
            rho = -0.5
            sigma = 0.5
            
            for _ in range(max_iter):
                if fevals >= max_feval:
                    break
                
                # Sort simplex
                order = np.argsort(simplex_f)
                simplex = [simplex[i] for i in order]
                simplex_f = [simplex_f[i] for i in order]
                
                # Check convergence
                f_range = simplex_f[-1] - simplex_f[0]
                if f_range < 1e-10:
                    break
                
                x_worst = simplex[-1]
                f_worst = simplex_f[-1]
                
                # Centroid of all but worst
                x_c = np.mean(simplex[:-1], axis=0)
                
                # Reflection
                x_r = x_c + alpha * (x_c - x_worst)
                f_r = func(x_r)
                fevals += 1
                
                if f_r < simplex_f[0]:
                    # Expansion
                    x_e = x_c + gamma * (x_r - x_c)
                    f_e = func(x_e)
                    fevals += 1
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
                    x_cw = x_c + rho * (x_worst - x_c)
                    f_cw = func(x_cw)
                    fevals += 1
                    if f_cw < f_worst:
                        simplex[-1] = x_cw
                        simplex_f[-1] = f_cw
                    else:
                        # Shrink
                        for i in range(1, len(simplex)):
                            simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                            simplex_f[i] = func(simplex[i])
                            fevals += 1
                            if fevals >= max_feval:
                                return simplex[0], simplex_f[0], fevals
                        
            # Return best point
            best_idx = np.argmin(simplex_f)
            return simplex[best_idx], simplex_f[best_idx], fevals

        # Allocate budget for restarts
        # Let's try 5 restarts
        num_restarts = 5
        fevals_per_restart = (budget - n_calls) // num_restarts
        
        for i in range(num_restarts):
            if n_calls >= budget:
                break
            
            # Start from best point found so far, or a random point for diversity in later restarts
            if i == 0:
                x_start = best_x.copy()
            else:
                # Mix of best point and random perturbation
                if np.random.rand() < 0.5:
                    x_start = best_x + np.random.normal(0, 1.0, dim)
                else:
                    x_start = np.random.uniform(-5, 5, dim)
            
            x_start = np.clip(x_start, -5, 5)
            
            # Run Nelder-Mead
            try:
                x_opt, f_opt, used_fev = nelder_mead(func, x_start, max_iter=100, max_feval=fevals_per_restart)
                if f_opt < best_fx:
                    best_fx = f_opt
                    best_x = x_opt.copy()
            except Exception:
                pass
        
        # Final polish with a few more Nelder-Mead iterations from best
        if n_calls < budget:
            remaining = budget - n_calls
            try:
                x_opt, f_opt, _ = nelder_mead(func, best_x, max_iter=50, max_feval=remaining)
                if f_opt < best_fx:
                    best_fx = f_opt
                    best_x = x_opt.copy()
            except Exception:
                pass

        return best_x, best_fx
