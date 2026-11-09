import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 295.0
        rng = np.random.default_rng(42)
        n = self.dim
        budget = self.budget

        best_x = np.zeros(n)
        best_f = func(best_x)
        calls_used = 1

        def check_deadline():
            return time.time() >= deadline

        def do_call(x):
            nonlocal calls_used, best_f, best_x
            if calls_used >= budget or check_deadline():
                return None
            x_clipped = np.clip(x, -5, 5)
            fx = func(x_clipped)
            calls_used += 1
            if fx < best_f:
                best_f = fx
                best_x = x_clipped.copy()
            return fx

        # Phase 1: Coarse random sampling to find a good basin
        exploration_budget = max(10, int(budget * 0.1))
        
        for _ in range(exploration_budget - 1):
            if calls_used >= budget or check_deadline():
                break
            x_new = 5.0 * (rng.random(n) - 0.5)
            f_new = do_call(x_new)
            if f_new is None:
                break

        # Nelder-Mead implementation with adaptive step size
        def nelder_mead(x0, f0, max_iter=500):
            nonlocal best_f, best_x, calls_used
            
            simplex = [x0.copy()]
            fvals = [f0]
            
            initial_step = 0.5
            
            for i in range(n):
                x_new = x0.copy()
                x_new[i] += initial_step
                x_new = np.clip(x_new, -5, 5)
                if calls_used >= budget or check_deadline():
                    return
                f_new = do_call(x_new)
                if f_new is None:
                    return
                simplex.append(x_new)
                fvals.append(f_new)
            
            if len(simplex) < n + 1:
                return

            alpha = 1.0
            gamma = 2.0
            rho = 0.5
            sigma = 0.5
            ftol = 1e-10
            
            for _ in range(max_iter):
                if calls_used >= budget or check_deadline():
                    break
                
                order = np.argsort(fvals)
                simplex = [simplex[i] for i in order]
                fvals = [fvals[i] for i in order]
                
                if fvals[-1] - fvals[0] < ftol:
                    break
                
                centroid = np.mean(simplex[:-1], axis=0)
                
                x_r = centroid + alpha * (centroid - simplex[-1])
                x_r = np.clip(x_r, -5, 5)
                
                f_r = do_call(x_r)
                if f_r is None:
                    break
                
                if fvals[0] <= f_r < fvals[-2]:
                    simplex[-1] = x_r
                    fvals[-1] = f_r
                elif f_r < fvals[0]:
                    x_e = centroid + gamma * (centroid - simplex[-1])
                    x_e = np.clip(x_e, -5, 5)
                    
                    f_e = do_call(x_e)
                    if f_e is None:
                        simplex[-1] = x_r
                        fvals[-1] = f_r
                        break
                    
                    if f_e < f_r:
                        simplex[-1] = x_e
                        fvals[-1] = f_e
                    else:
                        simplex[-1] = x_r
                        fvals[-1] = f_r
                else:
                    x_c = centroid + rho * (simplex[-1] - centroid)
                    x_c = np.clip(x_c, -5, 5)
                    
                    f_c = do_call(x_c)
                    if f_c is None:
                        break
                    
                    if f_c < fvals[-1]:
                        simplex[-1] = x_c
                        fvals[-1] = f_c
                    else:
                        for i in range(1, len(simplex)):
                            simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                            simplex[i] = np.clip(simplex[i], -5, 5)
                            
                            f_new = do_call(simplex[i])
                            if f_new is None:
                                return
                            fvals[i] = f_new
                        if check_deadline() or calls_used >= budget:
                            break

        # Phase 2: Multiple NM restarts with adaptive strategy
        nelder_mead(best_x.copy(), best_f)
        
        while calls_used < budget - 10 and not check_deadline():
            remaining = budget - calls_used
            if remaining < 2 * (n + 1):
                break
            
            # Adaptive restart: if we're close to optimum, perturb less
            if best_f < 10.0:
                scale = 0.1
            elif best_f < 100.0:
                scale = 0.3
            else:
                scale = 1.0
            
            if rng.random() < 0.6:
                x_start = best_x + scale * rng.standard_normal(n)
            else:
                x_start = 5.0 * (rng.random(n) - 0.5)
            x_start = np.clip(x_start, -5, 5)
            
            f_start = do_call(x_start)
            if f_start is None:
                break
            
            nelder_mead(x_start, f_start, max_iter=200)

        return best_x, best_f
