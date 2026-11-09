"""Hybrid Nelder-Mead with adaptive restarts and final gradient polish for BBOB f12."""
# EVOLVE-BLOCK-START
import time
import numpy as np


class Optimizer:
    """Nelder-Mead optimizer with adaptive restarts for smooth black-box optimization."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = int(self.budget)
        t0 = time.time()
        deadline = t0 + 280.0  # safety margin vs. 330 s kill
        max_calls = budget

        calls = [0]
        best_fx = float('inf')
        best_x = np.zeros(dim)
        rng = np.random.default_rng(42)

        def f(x):
            nonlocal best_fx, best_x
            if calls[0] >= max_calls:
                return float('inf')
            calls[0] += 1
            try:
                x_clipped = np.clip(x, -5.0, 5.0)
                val = float(func(x_clipped))
            except Exception:
                return float('inf')
            if not np.isfinite(val):
                return float('inf')
            if val < best_fx:
                best_fx = val
                best_x = x_clipped.copy()
            return val

        def remaining():
            return deadline - time.time()

        def finite_diff_gradient(x, h=1e-5):
            """Estimate gradient using central differences."""
            g = np.zeros(dim)
            for i in range(dim):
                xp = x.copy()
                xm = x.copy()
                xp[i] += h
                xm[i] -= h
                g[i] = (f(xp) - f(xm)) / (2 * h)
            return g

        def nelder_mead(x0, fx0, max_iter=200):
            """
            Run Nelder-Mead optimization starting from x0.
            Returns the best point and function value found.
            """
            n = dim + 1
            # Initialize simplex
            simplex = [x0.copy()]
            f_values = [fx0]
            
            # Create initial vertices with a reasonable step size
            # Bent cigar is anisotropic, so we start with a moderate step
            step = 1.0
            for i in range(dim):
                xi = x0.copy()
                xi[i] += step
                fi = f(xi)
                simplex.append(xi)
                f_values.append(fi)
                
            if calls[0] >= max_calls:
                return x0, fx0

            # NM Parameters
            alpha = 1.0  # Reflection
            gamma = 2.0  # Expansion
            rho = 0.5    # Contraction
            sigma = 0.5  # Shrink

            local_best_fx = min(f_values)
            local_best_x = simplex[int(np.argmin(f_values))].copy()
            stagnation = 0
            max_stagnation = 30

            for _ in range(max_iter):
                if calls[0] >= max_calls or remaining() < 10.0:
                    break

                # Sort simplex by function value
                order = np.argsort(f_values)
                simplex = [simplex[i] for i in order]
                f_values = [f_values[i] for i in order]

                # Check convergence
                if f_values[-1] - f_values[0] < 1e-10:
                    break

                # Check stagnation
                if f_values[0] >= local_best_fx - 1e-10:
                    stagnation += 1
                else:
                    local_best_fx = f_values[0]
                    local_best_x = simplex[0].copy()
                    stagnation = 0
                    
                if stagnation > max_stagnation:
                    break

                x_worst = simplex[-1]
                x_best = simplex[0]
                x_centroid = np.mean(simplex[:-1], axis=0)

                # 1. Reflection
                x_ref = x_centroid + alpha * (x_centroid - x_worst)
                f_ref = f(x_ref)

                if f_values[0] <= f_ref < f_values[-2]:
                    # Reflection accepted
                    simplex[-1] = x_ref
                    f_values[-1] = f_ref
                elif f_ref < f_values[0]:
                    # Expansion
                    x_exp = x_centroid + gamma * (x_ref - x_centroid)
                    f_exp = f(x_exp)
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
                    
                    f_con = f(x_con)
                    if f_con < min(f_ref, f_values[-1]):
                        simplex[-1] = x_con
                        f_values[-1] = f_con
                    else:
                        # Shrink
                        for i in range(1, n):
                            simplex[i] = x_best + sigma * (simplex[i] - x_best)
                            f_values[i] = f(simplex[i])

            # Return the best point found in this run
            final_order = np.argsort(f_values)
            return simplex[final_order[0]], f_values[final_order[0]]

        # Phase 1: Nelder-Mead with restarts
        # Allocate 80% of budget to NM, 20% for potential polishing or extra restarts
        nm_budget = int(budget * 0.8)
        
        # Start at origin
        x0 = np.zeros(dim)
        fx0 = f(x0)
        
        x, fx = nelder_mead(x0, fx0, max_iter=nm_budget)
        
        # If we still have budget, try a restart from a random point
        if calls[0] < max_calls and remaining() > 20.0:
            x1 = rng.uniform(-4, 4, dim)
            fx1 = f(x1)
            x_new, fx_new = nelder_mead(x1, fx1, max_iter=100)
            if fx_new < fx:
                x, fx = x_new, fx_new

        # Phase 2: Gradient-based polishing (optional, if budget allows)
        # Bent cigar is smooth, so gradient info can help in the final phase
        if calls[0] < max_calls - 20 and remaining() > 15.0:
            x = best_x.copy()
            fx = f(x)
            for _ in range(10):
                if calls[0] >= max_calls or remaining() < 5.0:
                    break
                g = finite_diff_gradient(x)
                if np.linalg.norm(g) < 1e-8:
                    break
                # Simple gradient step with line search
                d = -g
                step_size = 0.1 / (np.linalg.norm(d) + 1e-8)
                x_new = x + step_size * d
                x_new = np.clip(x_new, -5, 5)
                fx_new = f(x_new)
                if fx_new < fx:
                    x = x_new
                    fx = fx_new
                else:
                    break

        if best_x is None or not np.all(np.isfinite(best_x)):
            best_x = np.zeros(dim)
            if calls[0] < max_calls:
                best_fx = f(best_x)

        return best_x, best_fx
# EVOLVE-BLOCK-END
