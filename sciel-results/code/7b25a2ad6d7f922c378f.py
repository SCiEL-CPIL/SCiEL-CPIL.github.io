"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Nelder-Mead optimizer with adaptive scaling and restarts for BBOB f13."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget

        # Deadline guard
        start_time = time.time()
        deadline = start_time + 320.0

        calls_used = 0
        best_x = np.zeros(dim)
        best_f = np.inf

        def eval_point(x):
            nonlocal best_x, best_f, calls_used
            fx = func(x)
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            calls_used += 1
            return fx

        # Nelder-Mead parameters
        alpha = 1.0
        gamma = 2.0
        rho = 0.5
        sigma = 0.5
        min_delta = 1e-10

        # Initial simplex
        # Start at origin with a reasonable spread
        center = np.zeros(dim)
        f_center = eval_point(center)
        calls_used += 1

        # Create initial simplex
        simplex = np.zeros((dim + 1, dim))
        simplex_f = np.zeros(dim + 1)
        simplex[0] = center
        simplex_f[0] = f_center

        for i in range(dim):
            simplex[i + 1] = center.copy()
            simplex[i + 1][i] = 1.0  # Initial step size
            simplex_f[i + 1] = eval_point(simplex[i + 1])

        # Main loop
        while calls_used < budget:
            if time.time() > deadline:
                break

            # Sort simplex by function value
            idx = np.argsort(simplex_f)
            simplex = simplex[idx]
            simplex_f = simplex_f[idx]

            # Check convergence
            if np.std(simplex_f) < 1e-12:
                break

            # Compute centroid of best dim points (excluding worst)
            centroid = np.mean(simplex[:-1], axis=0)

            # Reflection
            x_ref = centroid + alpha * (centroid - simplex[-1])
            f_ref = eval_point(x_ref)

            if f_ref < simplex_f[0]:
                # Expansion
                x_exp = centroid + gamma * (centroid - simplex[-1])
                f_exp = eval_point(x_exp)
                if f_exp < f_ref:
                    simplex[-1] = x_exp
                    simplex_f[-1] = f_exp
                else:
                    simplex[-1] = x_ref
                    simplex_f[-1] = f_ref
            elif f_ref < simplex_f[-2]:
                # Accept reflection
                simplex[-1] = x_ref
                simplex_f[-1] = f_ref
            else:
                # Contraction
                x_con = centroid + rho * (simplex[-1] - centroid)
                f_con = eval_point(x_con)
                if f_con < simplex_f[-1]:
                    simplex[-1] = x_con
                    simplex_f[-1] = f_con
                else:
                    # Shrink
                    for i in range(1, dim + 1):
                        simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                        simplex_f[i] = eval_point(simplex[i])

            # Adaptive scaling: if simplex is too large, shrink it
            # If simplex is too small, restart
            simplex_range = np.max(simplex, axis=0) - np.min(simplex, axis=0)
            if np.all(simplex_range < min_delta):
                # Restart with a new random point
                new_center = np.random.uniform(-3, 3, dim)
                f_new = eval_point(new_center)
                if f_new < simplex_f[0]:
                    # Replace the worst point
                    simplex[-1] = new_center
                    simplex_f[-1] = f_new
                    # Re-expand simplex
                    for i in range(dim):
                        simplex[i + 1] = simplex[i].copy()
                        simplex[i + 1][i] = simplex[i + 1][i] + 0.5
                        simplex_f[i + 1] = eval_point(simplex[i + 1])
                else:
                    # Shrink simplex towards best point
                    for i in range(1, dim + 1):
                        simplex[i] = simplex[0] + 0.5 * (simplex[i] - simplex[0])
                        simplex_f[i] = eval_point(simplex[i])

        # Return best found
        if best_x is None:
            best_x = np.zeros(dim)
            best_f = func(best_x)

        return best_x, best_f
# EVOLVE-BLOCK-END
