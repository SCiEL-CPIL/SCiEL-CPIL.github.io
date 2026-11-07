"""Memetic Nelder-Mead optimizer for BBOB f24."""
# EVOLVE-BLOCK-START
import numpy as np


class Optimizer:
    """Nelder-Mead based optimizer with periodic resets."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        n = self.dim
        budget = self.budget
        rng = np.random.default_rng(42)

        best_x = np.zeros(n)
        best_f = np.inf
        calls = 0

        # Nelder-Mead parameters
        alpha = 1.0    # reflection
        gamma = 2.0    # expansion
        rho = 0.5      # contraction
        sigma = 0.5    # shrinkage

        # Initial simplex
        # Start with a large simplex to cover the box [-5, 5]^5
        init_step = 4.0
        
        def create_simplex(center):
            simplex = []
            f_vals = []
            # Center point
            x0 = center.copy()
            x0 = np.clip(x0, -5, 5)
            f0 = func(x0)
            calls += 1
            simplex.append(x0)
            f_vals.append(f0)
            
            # Axis points
            for i in range(n):
                xi = center.copy()
                xi[i] += init_step
                xi = np.clip(xi, -5, 5)
                fi = func(xi)
                calls += 1
                simplex.append(xi)
                f_vals.append(fi)
            return np.array(simplex), np.array(f_vals)

        # Initial simplex at origin
        simplex, f_vals = create_simplex(np.zeros(n))
        
        # Track best
        idx_min = np.argmin(f_vals)
        best_x = simplex[idx_min].copy()
        best_f = f_vals[idx_min]

        # Nelder-Mead loop
        max_iter = budget - calls
        for i in range(max_iter):
            if calls >= budget:
                break

            # Sort simplex by function value
            order = np.argsort(f_vals)
            simplex = simplex[order]
            f_vals = f_vals[order]

            # Update best
            if f_vals[0] < best_f:
                best_f = f_vals[0]
                best_x = simplex[0].copy()

            # Centroid of all points except the worst
            centroid = np.mean(simplex[:-1], axis=0)

            # Reflection
            x_ref = centroid + alpha * (centroid - simplex[-1])
            x_ref = np.clip(x_ref, -5, 5)
            f_ref = func(x_ref)
            calls += 1

            if f_ref < f_vals[0]:
                # Expansion
                x_exp = centroid + gamma * (x_ref - centroid)
                x_exp = np.clip(x_exp, -5, 5)
                f_exp = func(x_exp)
                calls += 1
                if f_exp < f_ref:
                    simplex[-1] = x_exp
                    f_vals[-1] = f_exp
                else:
                    simplex[-1] = x_ref
                    f_vals[-1] = f_ref
            elif f_ref < f_vals[-2]:
                # Reflection is good
                simplex[-1] = x_ref
                f_vals[-1] = f_ref
            else:
                # Contraction
                if f_ref < f_vals[-1]:
                    # Outside contraction
                    x_con = centroid + rho * (x_ref - centroid)
                else:
                    # Inside contraction
                    x_con = centroid - rho * (simplex[-1] - centroid)
                
                x_con = np.clip(x_con, -5, 5)
                f_con = func(x_con)
                calls += 1

                if f_con < min(f_ref, f_vals[-1]):
                    simplex[-1] = x_con
                    f_vals[-1] = f_con
                else:
                    # Shrinkage
                    for j in range(1, n + 1):
                        simplex[j] = simplex[0] + sigma * (simplex[j] - simplex[0])
                        simplex[j] = np.clip(simplex[j], -5, 5)
                        f_vals[j] = func(simplex[j])
                        calls += 1
                        if calls >= budget:
                            break
                    if calls >= budget:
                        break

            # Check for convergence or stagnation
            # If simplex is too small, reset
            if np.max(f_vals) - np.min(f_vals) < 1e-6:
                # Reset simplex around best point with smaller step
                init_step *= 0.5
                if init_step < 1e-3:
                    break
                simplex, f_vals = create_simplex(best_x)
                # Update best from new simplex
                idx_min = np.argmin(f_vals)
                if f_vals[idx_min] < best_f:
                    best_f = f_vals[idx_min]
                    best_x = simplex[idx_min].copy()

        return best_x, best_f
# EVOLVE-BLOCK-END
