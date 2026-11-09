"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


class Optimizer:
    """
    Multi-start Nelder-Mead optimizer.
    Uses derivative-free local search (Nelder-Mead) from multiple random starting points.
    This is robust for Rosenbrock-like functions where gradients can be misleading.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        rng = np.random.default_rng(42)
        calls_made = [0]
        best_x = None
        best_f = np.inf

        def counting_func(x):
            if calls_made[0] >= self.budget:
                return 1e10
            calls_made[0] += 1
            return func(x)

        # 1. Random Search Phase: Find good starting points
        n_random = 20
        candidates = []
        for _ in range(n_random):
            if calls_made[0] >= self.budget:
                break
            x = rng.uniform(-5, 5, self.dim)
            fx = counting_func(x)
            candidates.append((fx, x))
            if fx < best_f:
                best_f = fx
                best_x = x.copy()

        # Sort candidates by function value
        candidates.sort(key=lambda t: t[0])

        # 2. Nelder-Mead Phase: Refine top candidates
        # Define Nelder-Mead routine
        def nelder_mead(x0, bounds, max_iter):
            dim = self.dim
            alpha = 1.0
            gamma = 2.0
            rho = 0.5
            sigma = 0.5

            # Initialize simplex
            simplex = []
            simplex.append(x0.copy())
            for i in range(dim):
                x_new = x0.copy()
                step = 0.1
                x_new[i] += step
                x_new = np.clip(x_new, bounds[0], bounds[1])
                simplex.append(x_new)

            f_simplex = []
            for i in range(dim + 1):
                if calls_made[0] >= self.budget:
                    return None, np.inf
                f_simplex.append(counting_func(simplex[i]))

            for _ in range(max_iter):
                if calls_made[0] >= self.budget:
                    break

                # Sort simplex
                idx = np.argsort(f_simplex)
                simplex = [simplex[i] for i in idx]
                f_simplex = [f_simplex[i] for i in idx]

                # Centroid of best dim points
                centroid = np.mean(simplex[:-1], axis=0)

                # Reflection
                x_ref = centroid + alpha * (centroid - simplex[-1])
                x_ref = np.clip(x_ref, bounds[0], bounds[1])
                if calls_made[0] >= self.budget:
                    break
                f_ref = counting_func(x_ref)

                if f_ref < f_simplex[0]:
                    # Expansion
                    x_exp = centroid + gamma * (x_ref - centroid)
                    x_exp = np.clip(x_exp, bounds[0], bounds[1])
                    if calls_made[0] >= self.budget:
                        break
                    f_exp = counting_func(x_exp)

                    if f_exp < f_ref:
                        simplex[-1] = x_exp
                        f_simplex[-1] = f_exp
                    else:
                        simplex[-1] = x_ref
                        f_simplex[-1] = f_ref
                elif f_ref < f_simplex[-2]:
                    simplex[-1] = x_ref
                    f_simplex[-1] = f_ref
                else:
                    # Contraction
                    if f_ref < f_simplex[-1]:
                        x_con = centroid + rho * (x_ref - centroid)
                    else:
                        x_con = centroid - rho * (x_ref - centroid)

                    x_con = np.clip(x_con, bounds[0], bounds[1])
                    if calls_made[0] >= self.budget:
                        break
                    f_con = counting_func(x_con)

                    if f_con <= min(f_ref, f_simplex[-1]):
                        simplex[-1] = x_con
                        f_simplex[-1] = f_con
                    else:
                        # Shrink
                        for i in range(1, dim + 1):
                            simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                            simplex[i] = np.clip(simplex[i], bounds[0], bounds[1])
                            if calls_made[0] >= self.budget:
                                break
                            f_simplex[i] = counting_func(simplex[i])

            min_idx = np.argmin(f_simplex)
            return simplex[min_idx], f_simplex[min_idx]

        bounds = (-5, 5)
        
        # Run Nelder-Mead on top 3 candidates
        # We have ~480 calls left. NM uses ~3-6 calls per iteration.
        # 100 iterations * 3 calls = 300 calls. Safe.
        for fx, x in candidates[:3]:
            if calls_made[0] >= self.budget:
                break
            
            x_res, f_res = nelder_mead(x, bounds, max_iter=100)
            
            if f_res < best_f:
                best_f = f_res
                best_x = x_res

        return best_x, best_f
# EVOLVE-BLOCK-END
