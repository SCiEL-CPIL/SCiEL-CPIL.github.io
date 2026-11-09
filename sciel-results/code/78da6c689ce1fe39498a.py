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

        # Phase 1: Broad exploration to find the "cliff" of the step function
        # The step ellipsoidal function is flat far from the optimum and drops
        # sharply near it. We need to locate where the drop happens.
        
        # Start with center
        try_func(np.zeros(dim))
        
        # Use a Halton-like sequence for structured exploration
        def halton(index, base):
            result = 0.0
            f = 1.0
            i = index
            while i > 0:
                f /= base
                result += f * (i % base)
                i //= base
            return result

        # Generate diverse points using Halton sequence mapped to [-4, 4]^dim
        for i in range(30):
            if evals_used >= budget or time.time() > deadline:
                break
            x = np.array([halton(i + 2, base) * 8.0 - 4.0 for base in range(2, 2 + dim)])
            try_func(x)

        if evals_used >= budget or time.time() > deadline:
            return best_x, best_f

        # Phase 2: Multi-start Nelder-Mead simplex method
        # This exploits the step-ellipsoidal structure by rapidly contracting
        # simplices around the discontinuity.
        
        # Create 10 diverse initial centroids based on best found and random perturbations
        n_starts = 10
        centroids = []
        
        # First centroid is the best point found
        centroids.append(best_x.copy())
        
        # Add diverse centroids
        for i in range(n_starts - 1):
            if evals_used >= budget or time.time() > deadline:
                break
            # Mix of scaled best_x and random points
            if i % 3 == 0:
                # Perturbed best
                centroid = best_x + rng.standard_normal(dim) * 0.5
            elif i % 3 == 1:
                # Random point in a smaller box around best
                centroid = best_x + rng.uniform(-1.5, 1.5, dim)
            else:
                # Random point in full box
                centroid = rng.uniform(-4.0, 4.0, dim)
            centroid = np.clip(centroid, -5.0, 5.0)
            centroids.append(centroid)
        
        # For each centroid, run Nelder-Mead
        for centroid_idx in range(len(centroids)):
            if evals_used >= budget or time.time() > deadline:
                break
            
            # Build initial simplex around centroid
            simplex = [centroids[centroid_idx].copy()]
            for d in range(dim):
                if evals_used >= budget or time.time() > deadline:
                    break
                v = centroids[centroid_idx].copy()
                # Use adaptive step size based on distance from center
                step = 0.5 if abs(centroids[centroid_idx][d]) < 2.0 else 1.0
                v[d] += step
                v = np.clip(v, -5.0, 5.0)
                simplex.append(v)
            
            if len(simplex) < dim + 1:
                continue
            
            # Evaluate initial simplex
            simplex_f = []
            for v in simplex:
                if evals_used >= budget or time.time() > deadline:
                    break
                simplex_f.append(try_func(v))
            
            if len(simplex_f) < dim + 1:
                continue
            
            # Nelder-Mead iterations
            alpha = 1.0   # reflection
            gamma = 2.0   # expansion
            rho = 0.5     # contraction
            sigma = 0.5   # shrink
            
            max_iter = 50
            for iteration in range(max_iter):
                if evals_used >= budget or time.time() > deadline:
                    break
                
                # Sort simplex by function value
                order = np.argsort(simplex_f)
                simplex = [simplex[i] for i in order]
                simplex_f = [simplex_f[i] for i in order]
                
                # Check convergence
                if simplex_f[0] - simplex_f[-1] < 1e-10:
                    break
                
                # Centroid of all but worst point
                centroid = np.mean(simplex[:-1], axis=0)
                worst = simplex[-1]
                worst_f = simplex_f[-1]
                
                # Reflection
                x_r = centroid + alpha * (centroid - worst)
                x_r = np.clip(x_r, -5.0, 5.0)
                f_r = try_func(x_r)
                
                if f_r < simplex_f[0]:
                    # Expansion
                    x_e = centroid + gamma * (centroid - worst)
                    x_e = np.clip(x_e, -5.0, 5.0)
                    f_e = try_func(x_e)
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
                    if f_r < worst_f:
                        # Outside contraction
                        x_c = centroid + rho * (x_r - centroid)
                    else:
                        # Inside contraction
                        x_c = centroid - rho * (centroid - worst)
                    x_c = np.clip(x_c, -5.0, 5.0)
                    f_c = try_func(x_c)
                    
                    if f_c < min(f_r, worst_f):
                        simplex[-1] = x_c
                        simplex_f[-1] = f_c
                    else:
                        # Shrink
                        for i in range(1, len(simplex)):
                            simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                            simplex_f[i] = try_func(simplex[i])
                        if evals_used >= budget or time.time() > deadline:
                            break
            
            # Update global best
            if len(simplex_f) > 0:
                local_best_idx = np.argmin(simplex_f)
                if simplex_f[local_best_idx] < best_f:
                    best_f = simplex_f[local_best_idx]
                    best_x = simplex[local_best_idx].copy()

        if evals_used >= budget or time.time() > deadline:
            return best_x, best_f

        # Phase 3: Final polish with small-step coordinate descent
        x_curr = best_x.copy()
        f_curr = best_f
        
        for coord in range(dim):
            if evals_used >= budget or time.time() > deadline:
                break
            
            for step in [0.1, 0.05, 0.02, 0.01, 0.005, 0.002, 0.001]:
                if evals_used >= budget or time.time() > deadline:
                    break
                
                improved = False
                for sign in [1.0, -1.0]:
                    if evals_used >= budget or time.time() > deadline:
                        break
                    
                    x_new = x_curr.copy()
                    x_new[coord] += sign * step
                    x_new = np.clip(x_new, -5.0, 5.0)
                    f_new = try_func(x_new)
                    
                    if f_new < f_curr - 1e-10:
                        x_curr = x_new.copy()
                        f_curr = f_new
                        improved = True
                        break
                
                if not improved:
                    break

        return best_x, best_f
