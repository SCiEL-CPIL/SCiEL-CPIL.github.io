"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Nelder-Mead optimizer for smooth black-box optimization."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        t0 = time.time()
        deadline = t0 + 280.0  # Safety margin
        budget = self.budget
        dim = self.dim

        calls = 0
        best_fx = np.inf
        best_x = np.zeros(dim)

        def safe_func(x):
            nonlocal calls, best_fx, best_x
            if calls >= budget:
                return np.inf
            calls += 1
            # Clip to domain
            x_clipped = np.clip(x, -5.0, 5.0)
            try:
                fx = float(func(x_clipped))
            except Exception:
                fx = np.inf
            if np.isfinite(fx) and fx < best_fx:
                best_fx = fx
                best_x = x_clipped.copy()
            return fx

        # Initialize Nelder-Mead simplex
        # Start at origin, extend vertices along axes
        simplex = []
        f_values = []
        
        # Initial point at 0
        x0 = np.zeros(dim)
        f0 = safe_func(x0)
        simplex.append(x0)
        f_values.append(f0)

        # Create other vertices
        # Use a step size that covers a good portion of the domain initially
        # 1.0 is a standard choice for [-5, 5]
        step = 1.0
        for i in range(dim):
            xi = np.zeros(dim)
            xi[i] = step
            fi = safe_func(xi)
            simplex.append(xi)
            f_values.append(fi)

        # Nelder-Mead parameters
        alpha = 1.0  # Reflection
        gamma = 2.0  # Expansion
        rho = 0.5    # Contraction
        sigma = 0.5  # Shrink

        n = dim + 1
        iterations = 0
        max_iterations = budget // 2  # Rough estimate

        while calls < budget - n and iterations < max_iterations:
            if time.time() > deadline:
                break
            
            iterations += 1

            # Sort simplex by function value
            order = np.argsort(f_values)
            simplex = [simplex[i] for i in order]
            f_values = [f_values[i] for i in order]

            # Check convergence
            if f_values[-1] - f_values[0] < 1e-10:
                break

            x_worst = simplex[-1]
            x_best = simplex[0]
            x_second_best = simplex[1]
            
            # Centroid of all points except the worst
            x_centroid = np.mean(simplex[:-1], axis=0)

            # 1. Reflection
            x_ref = x_centroid + alpha * (x_centroid - x_worst)
            f_ref = safe_func(x_ref)

            if f_values[0] <= f_ref < f_values[-2]:
                # Reflection accepted
                simplex[-1] = x_ref
                f_values[-1] = f_ref
            elif f_ref < f_values[0]:
                # Expansion
                x_exp = x_centroid + gamma * (x_ref - x_centroid)
                f_exp = safe_func(x_exp)
                if f_exp < f_ref:
                    simplex[-1] = x_exp
                    f_values[-1] = f_exp
                else:
                    simplex[-1] = x_ref
                    f_values[-1] = f_ref
            else:
                # Contraction
                if f_ref < f_values[-1]:
                    # Outside contraction
                    x_con = x_centroid + rho * (x_ref - x_centroid)
                else:
                    # Inside contraction
                    x_con = x_centroid - rho * (x_worst - x_centroid)
                
                f_con = safe_func(x_con)
                if f_con < min(f_ref, f_values[-1]):
                    simplex[-1] = x_con
                    f_values[-1] = f_con
                else:
                    # Shrink
                    for i in range(1, n):
                        simplex[i] = x_best + sigma * (simplex[i] - x_best)
                        f_values[i] = safe_func(simplex[i])
        
        return best_x, best_fx
# EVOLVE-BLOCK-END
