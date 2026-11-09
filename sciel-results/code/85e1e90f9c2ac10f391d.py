"""Rewritten optimizer for BBOB f6 (Attractive sector) using Nelder-Mead Simplex with Restarts."""
# EVOLVE-BLOCK-START
import numpy as np


class Optimizer:
    """Continuous optimizer using Nelder-Mead Simplex method with adaptive restarts."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        rng = np.random.default_rng(42)

        calls_made = 0
        x_best = None
        f_best = np.inf

        def eval_func(x):
            nonlocal calls_made, x_best, f_best
            if calls_made >= budget:
                return np.inf
            x = np.clip(x, -5, 5)
            f = func(x)
            calls_made += 1
            if f < f_best:
                f_best = f
                x_best = x.copy()
            return f

        def nelder_mead(x_center, step_size, max_calls):
            nonlocal calls_made, x_best, f_best
            
            # Create initial simplex
            # x0 is center, xi = x0 + step_size * e_i
            simplex = [x_center.copy()]
            for i in range(dim):
                p = x_center.copy()
                p[i] += step_size
                simplex.append(np.clip(p, -5, 5))
            
            # Evaluate initial simplex
            f_values = [eval_func(p) for p in simplex]
            
            # If we ran out of calls, return
            if calls_made >= budget:
                return

            # Nelder-Mead parameters
            alpha = 1.0
            gamma = 2.0
            rho = 0.5
            sigma = 0.5
            
            max_iter = max(10, (max_calls - (dim + 1)) // 2) # Rough estimate
            
            for _ in range(max_iter):
                if calls_made >= budget:
                    break
                
                # Sort simplex by fitness
                sorted_idx = np.argsort(f_values)
                simplex = [simplex[i] for i in sorted_idx]
                f_values = [f_values[i] for i in sorted_idx]
                
                # Check convergence
                if abs(f_values[0] - f_values[-1]) < 1e-10:
                    break
                
                # Centroid of best N points (excluding worst)
                centroid = np.mean(simplex[:-1], axis=0)
                
                # Reflection
                x_ref = centroid + alpha * (centroid - simplex[-1])
                x_ref = np.clip(x_ref, -5, 5)
                f_ref = eval_func(x_ref)
                
                if f_ref < f_values[0]:
                    # Expansion
                    x_exp = centroid + gamma * (x_ref - centroid)
                    x_exp = np.clip(x_exp, -5, 5)
                    f_exp = eval_func(x_exp)
                    if f_exp < f_ref:
                        simplex[-1] = x_exp
                        f_values[-1] = f_exp
                    else:
                        simplex[-1] = x_ref
                        f_values[-1] = f_ref
                elif f_ref < f_values[-2]:
                    # Accept reflection
                    simplex[-1] = x_ref
                    f_values[-1] = f_ref
                else:
                    # Contraction
                    # Check if f_ref is better than worst
                    if f_ref < f_values[-1]:
                        # Outside contraction
                        x_con = centroid + rho * (x_ref - centroid)
                    else:
                        # Inside contraction
                        x_con = centroid - rho * (centroid - simplex[-1])
                    
                    x_con = np.clip(x_con, -5, 5)
                    f_con = eval_func(x_con)
                    
                    if f_con < min(f_ref, f_values[-1]):
                        simplex[-1] = x_con
                        f_values[-1] = f_con
                    else:
                        # Shrink
                        for i in range(1, dim + 1):
                            simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                            simplex[i] = np.clip(simplex[i], -5, 5)
                            f_values[i] = eval_func(simplex[i])
                            if calls_made >= budget:
                                return

        # Phase 1: Exploration
        # Start with a random point in [-3, 3]
        x_start = rng.uniform(-3, 3, dim)
        step_size_1 = 2.0
        budget_phase1 = int(budget * 0.6)
        nelder_mead(x_start, step_size_1, budget_phase1)
        
        # Phase 2: Exploitation
        if calls_made < budget and x_best is not None:
            x_start_2 = x_best.copy()
            step_size_2 = 0.5
            budget_phase2 = budget - calls_made
            nelder_mead(x_start_2, step_size_2, budget_phase2)

        return x_best, f_best
# EVOLVE-BLOCK-END
