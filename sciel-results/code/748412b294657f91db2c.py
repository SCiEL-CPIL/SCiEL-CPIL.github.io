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
        deadline = start_time + 285.0

        rng = np.random.default_rng(42)

        best_x = np.zeros(dim)
        best_f = np.inf
        calls_made = 0

        def call_func(x):
            nonlocal calls_made, best_f, best_x
            fx = func(np.clip(x, -5, 5))
            calls_made += 1
            if fx < best_f:
                best_f = fx
                best_x = np.clip(x, -5, 5).copy()
            return fx

        # Phase 1: Random initialization to find a good starting point
        n_init = 30
        init_points = []
        for _ in range(n_init):
            if calls_made >= budget or time.time() > deadline:
                break
            x = rng.uniform(-5, 5, dim)
            fx = call_func(x)
            init_points.append((fx, x.copy()))

        if len(init_points) == 0:
            return best_x, best_f

        init_points.sort(key=lambda t: t[0])
        start_x = init_points[0][1].copy()

        # Phase 2: Nelder-Mead simplex method
        # Initialize simplex
        simplex = [start_x.copy()]
        simplex_f = [call_func(start_x)]

        # Add vertices around the starting point
        step_size = 0.5
        for i in range(dim):
            v = start_x.copy()
            v[i] += step_size
            fv = call_func(v)
            simplex.append(v.copy())
            simplex_f.append(fv)

        # Nelder-Mead parameters
        alpha = 1.0   # reflection
        gamma = 2.0   # expansion
        rho = 0.5     # contraction
        sigma_nm = 0.5  # shrink
        max_iter = 500
        tol = 1e-10

        for iteration in range(max_iter):
            if calls_made >= budget or time.time() > deadline:
                break

            # Sort simplex by fitness
            indices = np.argsort(simplex_f)
            simplex = [simplex[i] for i in indices]
            simplex_f = [simplex_f[i] for i in indices]

            # Check convergence
            if abs(simplex_f[-1] - simplex_f[0]) < tol:
                break

            # Centroid of all but worst point
            centroid = np.mean(simplex[:-1], axis=0)

            # Reflection
            x_ref = centroid + alpha * (centroid - simplex[-1])
            f_ref = call_func(x_ref)

            if f_ref < simplex_f[0]:
                # Expansion
                x_exp = centroid + gamma * (centroid - simplex[-1])
                f_exp = call_func(x_exp)
                if f_exp < f_ref:
                    simplex[-1] = x_exp
                    simplex_f[-1] = f_exp
                else:
                    simplex[-1] = x_ref
                    simplex_f[-1] = f_ref
            elif f_ref < simplex_f[-2]:
                # Simple reflection
                simplex[-1] = x_ref
                simplex_f[-1] = f_ref
            else:
                # Contraction
                if f_ref < simplex_f[-1]:
                    # Outside contraction
                    x_con = centroid + 0.5 * (x_ref - centroid)
                else:
                    # Inside contraction
                    x_con = centroid - 0.5 * (centroid - simplex[-1])
                f_con = call_func(x_con)
                if f_con < min(f_ref, simplex_f[-1]):
                    simplex[-1] = x_con
                    simplex_f[-1] = f_con
                else:
                    # Shrink
                    for i in range(1, dim + 1):
                        simplex[i] = simplex[0] + sigma_nm * (simplex[i] - simplex[0])
                        simplex_f[i] = call_func(simplex[i])

            # Adaptive step size: shrink if no improvement for a while
            if iteration % 50 == 0 and iteration > 0:
                step_size *= 0.7
                # Re-center simplex if it's getting too small
                spread = np.max(np.array(simplex), axis=0) - np.min(np.array(simplex), axis=0)
                if np.max(spread) < 1e-4:
                    # Restart with new random point
                    x_new = rng.uniform(-5, 5, dim)
                    f_new = call_func(x_new)
                    if f_new < simplex_f[0]:
                        simplex = [x_new.copy()]
                        simplex_f = [f_new]
                        for i in range(dim):
                            v = x_new.copy()
                            v[i] += step_size
                            fv = call_func(v)
                            simplex.append(v.copy())
                            simplex_f.append(fv)

        return best_x, best_f
