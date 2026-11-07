import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 285.0
        dim = self.dim
        budget = self.budget
        
        best_x = None
        best_f = np.inf
        calls = 0
        
        def time_ok():
            return time.time() < deadline
            
        def record(x, f):
            nonlocal best_x, best_f
            if f < best_f:
                best_f = f
                best_x = x.copy()
        
        # Multi-start Nelder-Mead ensemble with Latin Hypercube sampling
        n_starts = 5
        simplex_size = dim + 1  # dim+1 vertices
        
        # Latin Hypercube sampling for diverse starting points
        # Each dimension gets one point from each of n_starts strata
        lhs_points = np.zeros((n_starts, dim))
        for d in range(dim):
            # Random permutation of strata for each dimension
            perm = np.random.permutation(n_starts)
            for s in range(n_starts):
                # Point in stratum perm[s] for dimension d
                # Uniform in [perm[s]/n_starts, (perm[s]+1)/n_starts] mapped to [-5, 5]
                frac = (perm[s] + np.random.random()) / n_starts
                lhs_points[s, d] = -5 + 10 * frac
        
        # Initialize simplexes for each start
        # Each simplex has dim+1 vertices
        alpha = 0.5  # Initial simplex spread
        
        simplexes = []
        f_simplexes = []
        
        # Evaluate initial simplexes
        for s in range(n_starts):
            if calls >= budget or not time_ok():
                break
                
            simplex = []
            f_vals = []
            
            # First vertex: the LHS point
            x0 = lhs_points[s].copy()
            x0 = np.clip(x0, -5, 5)
            f0 = func(x0)
            calls += 1
            record(x0, f0)
            simplex.append(x0)
            f_vals.append(f0)
            
            # Additional vertices: perturb each dimension
            for d in range(dim):
                if calls >= budget or not time_ok():
                    break
                x_i = x0.copy()
                x_i[d] += alpha
                x_i = np.clip(x_i, -5, 5)
                f_i = func(x_i)
                calls += 1
                record(x_i, f_i)
                simplex.append(x_i)
                f_vals.append(f_i)
            
            if len(simplex) >= 2:  # Need at least 2 vertices to continue
                simplexes.append(simplex)
                f_simplexes.append(f_vals)
        
        if len(simplexes) == 0:
            return best_x, best_f
        
        # Nelder-Mead parameters
        gamma = 1.0
        rho = 2.0
        sigma_nm = 0.5
        shrink = 0.5
        
        # Track convergence for each simplex
        converged = [False] * len(simplexes)
        
        # Interleave evaluations across all active simplexes
        active_indices = list(range(len(simplexes)))
        
        while calls < budget and time_ok() and active_indices:
            # Pick the next simplex to update (round-robin)
            idx = active_indices.pop(0)
            active_indices.append(idx)
            
            simplex = simplexes[idx]
            f_vals = f_simplexes[idx]
            
            # Sort by fitness
            order = np.argsort(f_vals)
            simplex = [simplex[i] for i in order]
            f_vals = [f_vals[i] for i in order]
            
            # Check convergence
            if len(f_vals) >= 2 and (f_vals[-1] - f_vals[0]) < 1e-10:
                converged[idx] = True
                continue
            
            x_best = simplex[0]
            f_best_s = f_vals[0]
            x_worst = simplex[-1]
            f_worst = f_vals[-1]
            
            if len(simplex) < 3:
                # Need at least 3 points for centroid calculation
                # Try to expand the simplex
                if calls < budget and time_ok():
                    d_rand = np.random.randint(dim)
                    x_new = x_best.copy()
                    x_new[d_rand] += alpha * 0.1
                    x_new = np.clip(x_new, -5, 5)
                    f_new = func(x_new)
                    calls += 1
                    record(x_new, f_new)
                    simplex.append(x_new)
                    f_vals.append(f_new)
                    simplexes[idx] = simplex
                    f_simplexes[idx] = f_vals
                continue
            
            centroid = np.mean(simplex[:-1], axis=0)
            
            # Reflection
            x_r = centroid + gamma * (centroid - x_worst)
            x_r = np.clip(x_r, -5, 5)
            if not time_ok() or calls >= budget:
                break
            f_r = func(x_r)
            calls += 1
            record(x_r, f_r)
            
            if f_r < f_best_s:
                # Expansion
                x_e = centroid + rho * (x_r - centroid)
                x_e = np.clip(x_e, -5, 5)
                if not time_ok() or calls >= budget:
                    break
                f_e = func(x_e)
                calls += 1
                record(x_e, f_e)
                if f_e < f_r:
                    simplex[-1] = x_e
                    f_vals[-1] = f_e
                else:
                    simplex[-1] = x_r
                    f_vals[-1] = f_r
            elif f_r < f_vals[-2]:
                # Accept reflection
                simplex[-1] = x_r
                f_vals[-1] = f_r
            else:
                # Contraction
                if f_r < f_worst:
                    x_c = centroid + sigma_nm * (x_r - centroid)
                else:
                    x_c = centroid - sigma_nm * (x_worst - centroid)
                x_c = np.clip(x_c, -5, 5)
                if not time_ok() or calls >= budget:
                    break
                f_c = func(x_c)
                calls += 1
                record(x_c, f_c)
                if f_c < min(f_r, f_worst):
                    simplex[-1] = x_c
                    f_vals[-1] = f_c
                else:
                    # Shrink
                    for i in range(1, len(simplex)):
                        if not time_ok() or calls >= budget:
                            break
                        simplex[i] = x_best + shrink * (simplex[i] - x_best)
                        simplex[i] = np.clip(simplex[i], -5, 5)
                        f_simplex_i = func(simplex[i])
                        calls += 1
                        record(simplex[i], f_simplex_i)
                        f_vals[i] = f_simplex_i
                    if not time_ok():
                        break
            
            simplexes[idx] = simplex
            f_simplexes[idx] = f_vals
            
            # Remove converged simplexes
            if converged[idx]:
                active_indices.remove(idx)
        
        return best_x, best_f
