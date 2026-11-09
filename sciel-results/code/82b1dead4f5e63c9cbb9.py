import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget

        # Time guard: evaluator kills at 330s, leave 40s safety margin
        start_time = time.time()
        deadline = start_time + 290

        rng = np.random.default_rng(42)

        best_x = np.zeros(dim)
        best_f = np.inf
        evals_used = 0

        def try_func(x):
            nonlocal evals_used, best_f, best_x
            if evals_used >= budget:
                return np.inf
            if time.time() > deadline:
                return np.inf
            x = np.clip(np.asarray(x, dtype=float), -5.0, 5.0)
            f_val = func(x)
            if not np.isfinite(f_val):
                return np.inf
            evals_used += 1
            if f_val < best_f:
                best_f = f_val
                best_x = x.copy()
            return f_val

        def nelder_mead(x_start, max_iter=100):
            nonlocal evals_used, best_f, best_x, deadline
            
            if time.time() > deadline:
                return x_start, best_f

            # Initialize simplex: x_start and dim neighbors with small offset
            alpha = 0.5
            simplex = [x_start.copy()]
            for i in range(dim):
                xi = x_start.copy()
                xi[i] += alpha
                simplex.append(xi)
            
            # Evaluate initial simplex
            fs = []
            for i in range(dim + 1):
                if evals_used >= budget or time.time() > deadline:
                    return best_x, best_f
                f_val = try_func(simplex[i])
                fs.append(f_val)
            
            # Nelder-Mead parameters
            alpha_nm = 1.0  # reflection
            gamma = 2.0     # expansion
            rho = 0.5       # contraction
            sigma = 0.5     # shrink
            max_iter_nm = max(1, (budget - evals_used) // (dim + 1))
            
            for it in range(max_iter_nm):
                if evals_used >= budget or time.time() > deadline:
                    break
                
                # Sort simplex by function value
                order = np.argsort(fs)
                simplex = [simplex[i] for i in order]
                fs = [fs[i] for i in order]
                
                # Check for convergence
                if len(set([round(f, 12) for f in fs])) == 1:
                    break
                
                # Centroid of all but worst point
                centroid = np.mean(simplex[:dim], axis=0)
                
                # Reflection
                x_ref = centroid + alpha_nm * (centroid - simplex[-1])
                f_ref = try_func(x_ref)
                
                if f_ref < fs[0]:
                    # Try expansion
                    x_exp = centroid + gamma * (x_ref - centroid)
                    f_exp = try_func(x_exp)
                    if f_exp < f_ref:
                        simplex[-1] = x_exp
                        fs[-1] = f_exp
                    else:
                        simplex[-1] = x_ref
                        fs[-1] = f_ref
                elif f_ref < fs[-2]:
                    simplex[-1] = x_ref
                    fs[-1] = f_ref
                else:
                    # Contraction
                    x_con = centroid + rho * (simplex[-1] - centroid)
                    f_con = try_func(x_con)
                    if f_con < fs[-1]:
                        simplex[-1] = x_con
                        fs[-1] = f_con
                    else:
                        # Shrink
                        for i in range(1, dim + 1):
                            simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                            fs[i] = try_func(simplex[i])
                        if evals_used >= budget or time.time() > deadline:
                            break
                
                # Update global best
                for i in range(dim + 1):
                    if fs[i] < best_f:
                        best_f = fs[i]
                        best_x = simplex[i].copy()
            
            return best_x, best_f

        # Phase 1: Multi-start exploration with 10 diverse points
        # Use Halton sequence for quasi-random coverage
        def halton(index, base):
            result = 0.0
            f = 1.0
            i = index
            while i > 0:
                f /= base
                result += f * (i % base)
                i = i // base
            return result

        # Generate 10 diverse starting points using Halton sequence
        start_points = []
        for i in range(1, 11):
            x = np.array([halton(i, base) for base in [2, 3, 5, 7, 11]])
            x = x * 8.0 - 4.0  # Map to [-4, 4]
            start_points.append(x)
        
        # Evaluate all starting points
        for x_start in start_points:
            if evals_used >= budget or time.time() > deadline:
                break
            try_func(x_start)

        if evals_used >= budget or time.time() > deadline:
            return best_x, best_f

        # Phase 2: Run Nelder-Mead from each starting point
        # Allocate remaining budget among the starts
        remaining = budget - evals_used
        if remaining > 0:
            # Sort starts by function value
            start_fvals = [try_func(sp) if evals_used < budget and time.time() < deadline else np.inf for sp in start_points]
            
            # Actually, we already evaluated them, so let's track their values
            # Let's re-evaluate to get proper values for sorting (but this wastes budget)
            # Better: track during initial evaluation
            
            # For simplicity, run NM from the best few starts
            # Run NM from the best 3-5 starts
            n_nm_runs = min(5, len(start_points))
            
            # Sort start points by their initial function values
            # We need to re-evaluate to sort properly, or track during phase 1
            # Let's just run NM from the best_x and a few others
            
            # Run NM from current best
            if evals_used < budget and time.time() < deadline:
                nelder_mead(best_x.copy(), max_iter=50)
            
            # Run NM from a few other diverse points
            for i in range(min(4, len(start_points))):
                if evals_used >= budget or time.time() > deadline:
                    break
                nelder_mead(start_points[i].copy(), max_iter=30)
            
            # If we have budget left, do another round of NM from the best
            if evals_used < budget and time.time() < deadline:
                nelder_mead(best_x.copy(), max_iter=40)

        return best_x, best_f
