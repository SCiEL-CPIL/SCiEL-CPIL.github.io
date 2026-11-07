import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        rng = np.random.default_rng(0)
        bounds = (-5, 5)
        
        # Global state
        calls_made = 0
        best_x = None
        best_f = float('inf')
        
        start_time = time.time()
        time_limit = 280.0 # Safety margin
        
        # Define a wrapper to count calls and track best
        def count_func(x):
            nonlocal calls_made, best_x, best_f
            if calls_made >= self.budget:
                return best_f # Should not happen
            x = np.clip(x, bounds[0], bounds[1])
            f = func(x)
            calls_made += 1
            if f < best_f:
                best_f = f
                best_x = x.copy()
            return f

        # Nelder-Mead implementation
        def nelder_mead(x0, max_iter):
            nonlocal calls_made
            dim = self.dim
            # Simplex initialization
            simplex = [x0.copy()]
            for i in range(dim):
                xi = x0.copy()
                xi[i] += 0.5
                simplex.append(np.clip(xi, bounds[0], bounds[1]))
            
            fvals = [count_func(x) for x in simplex]
            
            for _ in range(max_iter):
                if calls_made >= self.budget:
                    break
                if time.time() - start_time > time_limit:
                    break
                
                # Sort
                order = np.argsort(fvals)
                simplex = [simplex[i] for i in order]
                fvals = [fvals[i] for i in order]
                
                # Convergence check
                if max(fvals) - min(fvals) < 1e-8:
                    break
                # Also check simplex size
                if np.max(np.std(simplex, axis=0)) < 1e-6:
                    break

                x_worst = simplex[-1]
                x_best = simplex[0]
                x_centroid = np.mean(simplex[:-1], axis=0)
                
                # Reflection
                alpha = 1.0
                x_ref = x_centroid + alpha * (x_centroid - x_worst)
                x_ref = np.clip(x_ref, bounds[0], bounds[1])
                f_ref = count_func(x_ref)
                
                if f_ref < fvals[0]:
                    # Expansion
                    gamma = 2.0
                    x_exp = x_centroid + gamma * (x_centroid - x_worst)
                    x_exp = np.clip(x_exp, bounds[0], bounds[1])
                    f_exp = count_func(x_exp)
                    if f_exp < f_ref:
                        simplex[-1] = x_exp
                        fvals[-1] = f_exp
                    else:
                        simplex[-1] = x_ref
                        fvals[-1] = f_ref
                elif f_ref < fvals[-2]:
                    # Replace
                    simplex[-1] = x_ref
                    fvals[-1] = f_ref
                else:
                    # Contraction
                    rho = 0.5
                    x_con = x_centroid + rho * (x_worst - x_centroid)
                    x_con = np.clip(x_con, bounds[0], bounds[1])
                    f_con = count_func(x_con)
                    if f_con < fvals[-1]:
                        simplex[-1] = x_con
                        fvals[-1] = f_con
                    else:
                        # Shrink
                        for i in range(1, dim + 1):
                            simplex[i] = (simplex[0] + simplex[i]) / 2
                            simplex[i] = np.clip(simplex[i], bounds[0], bounds[1])
                            fvals[i] = count_func(simplex[i])
                            if calls_made >= self.budget:
                                break
            
            return simplex[0], fvals[0]

        # Multi-start strategy
        num_starts = 5
        evals_per_start = self.budget // num_starts
        
        for i in range(num_starts):
            if calls_made >= self.budget:
                break
            if time.time() - start_time > time_limit:
                break
            
            # Random start
            x0 = rng.uniform(bounds[0], bounds[1], self.dim)
            nelder_mead(x0, evals_per_start)
            
        return best_x, best_f
