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

        def time_ok():
            return time.time() < deadline

        calls = 0
        best_f = np.inf
        best_x = np.zeros(dim)

        def eval_func(x):
            nonlocal calls, best_f, best_x
            if calls >= budget or not time_ok():
                return np.inf
            calls += 1
            f = float(func(x))
            if f < best_f:
                best_f = f
                best_x = x.copy()
            return f

        # Phase 1: Random sampling to find a good starting point
        # Use 30 random points in [-4, 4]^5 (minimizer is in [-4,4]^5)
        rng = np.random.default_rng(42)
        n_random = min(30, budget - 1)
        for i in range(n_random):
            x = rng.uniform(-4.0, 4.0, size=dim)
            eval_func(x)

        # Phase 2: Multi-start Nelder-Mead with 10 independent runs
        # Each run gets 45 evaluations (total 450, leaving 20 for initial random)
        # Actually: 30 random + 10*45 = 480, leaving 20 buffer
        n_starts = 10
        evals_per_start = min(45, (budget - calls - 5) // n_starts)
        
        # Create deterministic starting points spread across the box
        # Use a grid-like pattern with some randomization
        base_points = []
        # First, use the best point from random sampling
        base_points.append(best_x.copy())
        
        # Add 9 more points: corners of a sub-grid plus random perturbations
        # Use deterministic points based on fractions of the box
        fractions = [0.1, 0.3, 0.5, 0.7, 0.9]
        # Create 9 points from combinations of these fractions
        import itertools
        frac_combos = list(itertools.product(fractions[:3], repeat=dim))[:9]
        for combo in frac_combos:
            x = np.array([c * 8.0 - 4.0 for c in combo])  # map [0,1] to [-4,4]
            base_points.append(x)
        
        # If we still need more points, add random ones
        while len(base_points) < n_starts:
            base_points.append(rng.uniform(-4.0, 4.0, size=dim))
        
        base_points = base_points[:n_starts]
        
        # Run Nelder-Mead from each starting point
        for start_idx in range(n_starts):
            if calls >= budget - 5 or not time_ok():
                break
            
            x0 = base_points[start_idx].copy()
            
            # Initialize simplex
            simplex = []
            simplex.append(x0)
            for i in range(dim):
                xi = x0.copy()
                # Use a small perturbation based on the coordinate
                step = 0.3 * (1.0 / (start_idx + 1))  # decreasing step size
                if xi[i] == 0:
                    xi[i] = step
                else:
                    xi[i] += step * np.sign(xi[i])
                simplex.append(xi)
            
            fvals = [eval_func(simplex[i]) for i in range(dim + 1)]
            
            alpha = 1.0   # reflection
            gamma = 2.0   # expansion
            rho = 0.5     # contraction
            sigma = 0.5   # shrink
            
            max_nelder = evals_per_start
            n_iter = 0
            
            while n_iter < max_nelder and time_ok() and calls < budget:
                n_iter += 1
                # Sort simplex
                idx = np.argsort(fvals)
                simplex = [simplex[i] for i in idx]
                fvals = [fvals[i] for i in idx]
                
                # Centroid of all but worst
                centroid = np.mean(simplex[:dim], axis=0)
                
                # Reflection
                x_r = centroid + alpha * (centroid - simplex[-1])
                f_r = eval_func(x_r)
                
                if f_r < fvals[0]:
                    # Expansion
                    x_e = centroid + gamma * (x_r - centroid)
                    f_e = eval_func(x_e)
                    if f_e < f_r:
                        simplex[-1] = x_e
                        fvals[-1] = f_e
                    else:
                        simplex[-1] = x_r
                        fvals[-1] = f_r
                elif f_r < fvals[-2]:
                    # Accept reflection
                    simplex[-1] = x_r
                    fvals[-1] = f_r
                else:
                    # Contraction
                    if f_r < fvals[-1]:
                        # Outside contraction
                        x_c = centroid + rho * (x_r - centroid)
                    else:
                        # Inside contraction
                        x_c = centroid - rho * (centroid - simplex[-1])
                    f_c = eval_func(x_c)
                    if f_c < min(f_r, fvals[-1]):
                        simplex[-1] = x_c
                        fvals[-1] = f_c
                    else:
                        # Shrink
                        for i in range(1, dim + 1):
                            simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                            fvals[i] = eval_func(simplex[i])
                
                # Check if we've converged
                if max(fvals) - min(fvals) < 1e-8:
                    break

        return best_x, best_f
