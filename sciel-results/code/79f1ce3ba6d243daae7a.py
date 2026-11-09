import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget

        start_time = time.time()
        deadline = start_time + 280

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
            try:
                f_val = float(func(x))
            except Exception:
                return np.inf
            if not np.isfinite(f_val):
                return np.inf
            evals_used += 1
            if f_val < best_f:
                best_f = f_val
                best_x = x.copy()
            return f_val

        def halton(index, base):
            result = 0.0
            f = 1.0
            i = index
            while i > 0:
                f /= base
                result += f * (i % base)
                i //= base
            return result

        # Phase 1: Broad exploration with stratified sampling
        # For step ellipsoidal, the function is flat far from the optimum
        # and drops sharply near it. We need to locate the "cliff".
        
        # Start with center
        try_func(np.zeros(dim))
        
        # Use Halton sequence for diverse coverage in [-4, 4]^dim
        for i in range(30):
            if evals_used >= budget or time.time() > deadline:
                break
            x = np.array([halton(i + 2, base) * 8.0 - 4.0 for base in range(2, 2 + dim)])
            try_func(x)

        if evals_used >= budget or time.time() > deadline:
            return best_x, best_f

        # Phase 2: Multi-start Nelder-Mead simplex method
        # Create 10 diverse initial centroids from the best points found so far
        # and run aggressive local refinement on each
        
        # Collect candidate starting points: center, halton points, and best found
        candidates = [np.zeros(dim)]
        for i in range(30):
            x = np.array([halton(i + 2, base) * 8.0 - 4.0 for base in range(2, 2 + dim)])
            candidates.append(x)
        candidates.append(best_x.copy())
        
        # Sort candidates by their function values (we need to evaluate some)
        # But we already have evaluations for the first 31 points
        # Let's just use the best few points as starting centroids
        
        # Get top 10 diverse points from our evaluations
        # Since we don't store all evaluated points, let's create 10 diverse starts
        # based on the Halton sequence and the best point
        
        starts = []
        # Start 1: the best point found
        starts.append(best_x.copy())
        
        # Starts 2-10: diverse points from Halton sequence, offset slightly
        for i in range(9):
            if evals_used >= budget or time.time() > deadline:
                break
            base_idx = i * 4 + 3  # spread out indices
            x = np.array([halton(base_idx, base) * 8.0 - 4.0 for base in range(2, 2 + dim)])
            # Add small perturbation to make it distinct
            x += 0.1 * np.array([halton(base_idx + 100, base) * 2 - 1 for base in range(2, 2 + dim)])
            x = np.clip(x, -5.0, 5.0)
            starts.append(x)
        
        # For each start, run Nelder-Mead with remaining budget
        remaining = budget - evals_used
        if remaining > 0:
            # Allocate budget: give more to the best start
            # First start gets 40% of remaining, others share the rest
            nm_budgets = []
            first_budget = max(20, int(remaining * 0.4))
            nm_budgets.append(first_budget)
            rest_budget = remaining - first_budget
            per_start = max(5, rest_budget // max(1, len(starts) - 1))
            for _ in range(len(starts) - 1):
                nm_budgets.append(per_start)
            
            # Run Nelder-Mead for each start
            for start_idx, start_x in enumerate(starts):
                if evals_used >= budget or time.time() > deadline:
                    break
                
                nm_budget = nm_budgets[start_idx]
                if nm_budget < 10:
                    continue
                
                # Initialize simplex: start_x plus dim perturbed points
                simplex = [start_x.copy()]
                simplex_f = [try_func(start_x)]
                
                # Create perturbed points
                step = 0.5
                for d in range(dim):
                    x_pert = start_x.copy()
                    x_pert[d] += step
                    x_pert = np.clip(x_pert, -5.0, 5.0)
                    simplex.append(x_pert)
                    simplex_f.append(try_func(x_pert))
                
                if evals_used >= budget or time.time() > deadline:
                    break
                
                # Nelder-Mead iterations
                alpha = 1.0   # reflection
                gamma = 2.0   # expansion
                rho = 0.5     # contraction
                sigma = 0.5   # shrink
                
                max_iters = nm_budget
                for it in range(max_iters):
                    if evals_used >= budget or time.time() > deadline:
                        break
                    
                    # Sort simplex by function value
                    order = np.argsort(simplex_f)
                    simplex = [simplex[i] for i in order]
                    simplex_f = [simplex_f[i] for i in order]
                    
                    # Centroid of all but worst
                    centroid = np.mean(simplex[:-1], axis=0)
                    
                    # Reflect worst point
                    worst = simplex[-1]
                    x_ref = centroid + alpha * (centroid - worst)
                    x_ref = np.clip(x_ref, -5.0, 5.0)
                    f_ref = try_func(x_ref)
                    
                    if evals_used >= budget or time.time() > deadline:
                        break
                    
                    if f_ref < simplex_f[0]:
                        # Try expansion
                        x_exp = centroid + gamma * (x_ref - centroid)
                        x_exp = np.clip(x_exp, -5.0, 5.0)
                        f_exp = try_func(x_exp)
                        
                        if evals_used >= budget or time.time() > deadline:
                            break
                        
                        if f_exp < f_ref:
                            simplex[-1] = x_exp
                            simplex_f[-1] = f_exp
                        else:
                            simplex[-1] = x_ref
                            simplex_f[-1] = f_ref
                    elif f_ref < simplex_f[-2]:
                        # Accept reflection
                        simplex[-1] = x_ref
                        simplex_f[-1] = f_ref
                    else:
                        # Contraction
                        if f_ref < simplex_f[-1]:
                            # Outside contraction
                            x_con = centroid + rho * (x_ref - centroid)
                        else:
                            # Inside contraction
                            x_con = centroid - rho * (x_ref - centroid)
                        x_con = np.clip(x_con, -5.0, 5.0)
                        f_con = try_func(x_con)
                        
                        if evals_used >= budget or time.time() > deadline:
                            break
                        
                        if f_con <= f_ref:
                            simplex[-1] = x_con
                            simplex_f[-1] = f_con
                        else:
                            # Shrink
                            for i in range(1, dim + 1):
                                simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                                simplex[i] = np.clip(simplex[i], -5.0, 5.0)
                                simplex_f[i] = try_func(simplex[i])
                                if evals_used >= budget or time.time() > deadline:
                                    break
                            if evals_used >= budget or time.time() > deadline:
                                break
        
        return best_x, best_f
