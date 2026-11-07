"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-start Nelder-Mead ensemble with time guard."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0  # Safety margin for 300s limit
        dim = self.dim
        budget = self.budget
        calls = 0
        best_x = None
        best_f = np.inf

        def time_ok():
            return time.time() < deadline

        def record(x, f):
            nonlocal best_x, best_f
            if f < best_f:
                best_f = f
                best_x = x.copy()

        # Generate diverse starting points using Latin Hypercube Sampling
        n_starts = 5
        rng = np.random.default_rng(42)
        
        # Latin Hypercube Sampling for dim dimensions
        starts = []
        for _ in range(n_starts):
            x = np.zeros(dim)
            for d in range(dim):
                x[d] = rng.uniform(-4, 4)
            starts.append(x)
        
        # Also add center and a few random points for diversity
        starts.append(np.zeros(dim))
        for _ in range(2):
            starts.append(rng.uniform(-4, 4, dim))
        
        # Evaluate all starting points
        start_points = []
        for x in starts:
            if not time_ok() or calls >= budget:
                break
            x = np.clip(x, -5, 5)
            fx = func(x)
            calls += 1
            record(x, fx)
            start_points.append((x, fx))
            if not time_ok():
                break
        
        if not start_points:
            return best_x, best_f

        # Sort starting points by function value
        start_points.sort(key=lambda t: t[1])
        
        # Run Nelder-Mead from the best few starting points
        # We'll run Nelder-Mead on up to 3 of the best starting points
        nm_starts = start_points[:min(3, len(start_points))]
        
        # Nelder-Mead parameters
        gamma = 1.0
        rho = 2.0
        sigma_nm = 0.5
        shrink = 0.5
        alpha = 0.5  # Initial simplex size
        
        for x0, f0 in nm_starts:
            if not time_ok() or calls >= budget:
                break
            
            # Initialize simplex around this starting point
            simplex = [x0.copy()]
            f_simplex = [f0]
            
            for i in range(dim):
                if not time_ok() or calls >= budget:
                    break
                x_i = x0.copy()
                x_i[i] += alpha
                x_i = np.clip(x_i, -5, 5)
                f_i = func(x_i)
                calls += 1
                simplex.append(x_i)
                f_simplex.append(f_i)
                record(x_i, f_i)
                if not time_ok():
                    break
            
            if len(simplex) < 2:
                continue
            
            # Run Nelder-Mead iterations
            max_nm_iters = 100
            for _ in range(max_nm_iters):
                if not time_ok() or calls >= budget:
                    break
                
                # Sort simplex by function value
                indices = np.argsort(f_simplex)
                simplex = [simplex[i] for i in indices]
                f_simplex = [f_simplex[i] for i in indices]
                
                x_best = simplex[0]
                f_best = f_simplex[0]
                x_worst = simplex[-1]
                f_worst = f_simplex[-1]
                x_second_worst = simplex[-2]
                f_second_worst = f_simplex[-2]
                
                # Centroid of all but worst
                centroid = np.mean(simplex[:-1], axis=0)
                
                # Reflection
                x_r = centroid + gamma * (centroid - x_worst)
                x_r = np.clip(x_r, -5, 5)
                if not time_ok() or calls >= budget:
                    break
                f_r = func(x_r)
                calls += 1
                record(x_r, f_r)
                
                if f_r < f_best:
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
                        f_simplex[-1] = f_e
                    else:
                        simplex[-1] = x_r
                        f_simplex[-1] = f_r
                elif f_r < f_second_worst:
                    simplex[-1] = x_r
                    f_simplex[-1] = f_r
                else:
                    # Contraction
                    if f_r < f_worst:
                        # Outside contraction
                        x_c = centroid + sigma_nm * (x_r - centroid)
                    else:
                        # Inside contraction
                        x_c = centroid - sigma_nm * (x_worst - centroid)
                    x_c = np.clip(x_c, -5, 5)
                    
                    if not time_ok() or calls >= budget:
                        break
                    f_c = func(x_c)
                    calls += 1
                    record(x_c, f_c)
                    
                    if f_c < min(f_r, f_worst):
                        simplex[-1] = x_c
                        f_simplex[-1] = f_c
                    else:
                        # Shrink
                        for i in range(1, len(simplex)):
                            if not time_ok() or calls >= budget:
                                break
                            simplex[i] = x_best + shrink * (simplex[i] - x_best)
                            simplex[i] = np.clip(simplex[i], -5, 5)
                            f_simplex[i] = func(simplex[i])
                            calls += 1
                            record(simplex[i], f_simplex[i])
                        if not time_ok():
                            break
                
                # Check for convergence
                if np.max(f_simplex) - np.min(f_simplex) < 1e-12:
                    break
        
        return best_x, best_f
# EVOLVE-BLOCK-END
