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

        def run_nelder_mead(x_start, step_size, max_calls_remaining):
            nonlocal calls, best_fx, best_x
            
            # Initialize simplex
            simplex = []
            f_values = []
            
            x0 = x_start.copy()
            f0 = safe_func(x0)
            simplex.append(x0)
            f_values.append(f0)

            for i in range(dim):
                xi = x_start.copy()
                xi[i] += step_size
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
            max_iterations = max_calls_remaining // n
            
            # Convergence criteria
            min_simplex_size = 1e-6
            max_no_improve = 50
            
            no_improve_count = 0
            last_best_fx = best_fx

            while calls < budget and iterations < max_iterations:
                if time.time() > deadline:
                    return

                iterations += 1

                # Sort simplex by function value
                order = np.argsort(f_values)
                simplex = [simplex[i] for i in order]
                f_values = [f_values[i] for i in order]

                # Check convergence based on simplex size
                simplex_size = np.linalg.norm(simplex[-1] - simplex[0])
                if simplex_size < min_simplex_size:
                    break

                # Check for stall
                if best_fx < last_best_fx - 1e-10:
                    no_improve_count = 0
                    last_best_fx = best_fx
                else:
                    no_improve_count += 1
                    if no_improve_count > max_no_improve:
                        break

                x_worst = simplex[-1]
                x_best = simplex[0]
                
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
                            if calls >= budget:
                                return

        # Initial run with a larger step to cover the domain
        # Domain is [-5, 5], so a step of 2.0 is reasonable
        run_nelder_mead(np.zeros(dim), 2.0, budget)

        # If we still have budget, do a polish run with a smaller step from the best point
        if calls < budget:
            run_nelder_mead(best_x, 0.5, budget - calls)

        return best_x, best_fx
# EVOLVE-BLOCK-END
