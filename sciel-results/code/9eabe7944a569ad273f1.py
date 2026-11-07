"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-start Nelder-Mead ensemble for BBOB f14."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280  # Safety margin for 300s limit
        rng = np.random.default_rng(42)
        
        # Track global best
        best_x = None
        best_f = np.inf
        calls_made = 0
        
        def eval_func(x):
            nonlocal best_x, best_f, calls_made
            x = np.clip(x, -5, 5)
            f = func(x)
            calls_made += 1
            if f < best_f:
                best_f = f
                best_x = x.copy()
            return f

        # Initialize 5 diverse simplexes using Latin Hypercube-like sampling
        n_starts = 5
        simplex_list = []
        
        for s in range(n_starts):
            # Center for this start
            center = rng.uniform(-4, 4, self.dim)
            simplex = [center.copy()]
            for i in range(self.dim):
                dx = center.copy()
                dx[i] += 1.0
                simplex.append(np.clip(dx, -5, 5))
            simplex_list.append(simplex)
        
        # Evaluate initial simplexes
        for simplex in simplex_list:
            for i, x in enumerate(simplex):
                f = eval_func(x)
                simplex[i] = (x, f)
        
        # Sort simplexes by their best vertex
        def simplex_best(simplex):
            return min(f for _, f in simplex)
        
        # Nelder-Mead parameters
        alpha = 1.0
        gamma = 2.0
        rho = 0.5
        sigma = 0.5
        max_iter_per_simplex = 100
        
        # Main loop: run Nelder-Mead on the most promising simplex
        # We interleave or pick the best one to improve
        # Given the budget, we'll run them sequentially but prioritize the best
        
        active_simplexes = simplex_list[:]
        
        while calls_made < self.budget and time.time() < deadline:
            # Pick the simplex with the best current value
            active_simplexes.sort(key=simplex_best)
            simplex = active_simplexes[0]
            
            # Perform one Nelder-Mead iteration on this simplex
            # Sort vertices
            simplex.sort(key=lambda p: p[1])
            
            x_worst = simplex[-1][0]
            f_worst = simplex[-1][1]
            x_second = simplex[-2][0]
            f_second = simplex[-2][1]
            
            # Centroid of all but worst
            x_centroid = np.mean([p[0] for p in simplex[:-1]], axis=0)
            
            # Reflection
            x_reflect = x_centroid + alpha * (x_centroid - x_worst)
            f_reflect = eval_func(x_reflect)
            
            if f_reflect < simplex[0][1]:
                # Expansion
                x_expand = x_centroid + gamma * (x_reflect - x_worst)
                f_expand = eval_func(x_expand)
                if f_expand < f_reflect:
                    simplex[-1] = (x_expand, f_expand)
                else:
                    simplex[-1] = (x_reflect, f_reflect)
            elif f_reflect < f_second:
                # Accept reflection
                simplex[-1] = (x_reflect, f_reflect)
            else:
                # Contraction
                if f_reflect < f_worst:
                    # Outside contraction
                    x_contract = x_centroid + rho * (x_reflect - x_worst)
                else:
                    # Inside contraction
                    x_contract = x_centroid - rho * (x_worst - x_centroid)
                f_contract = eval_func(x_contract)
                
                if f_contract <= min(f_reflect, f_worst):
                    simplex[-1] = (x_contract, f_contract)
                else:
                    # Shrink
                    for i in range(1, len(simplex)):
                        x_shrink = simplex[0][0] + sigma * (simplex[i][0] - simplex[0][0])
                        f_shrink = eval_func(x_shrink)
                        simplex[i] = (x_shrink, f_shrink)
            
            # If the simplex has converged, restart it
            f_range = simplex[-1][1] - simplex[0][1]
            if f_range < 1e-6:
                # Restart this simplex with a new random center
                new_center = rng.uniform(-4, 4, self.dim)
                new_simplex = [new_center.copy()]
                for i in range(self.dim):
                    dx = new_center.copy()
                    dx[i] += 1.0
                    new_simplex.append(np.clip(dx, -5, 5))
                for i, x in enumerate(new_simplex):
                    f = eval_func(x)
                    new_simplex[i] = (x, f)
                active_simplexes[0] = new_simplex

        return best_x, best_f
# EVOLVE-BLOCK-END
