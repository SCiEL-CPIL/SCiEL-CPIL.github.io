"""Rewritten program using Nelder-Mead simplex optimization with adaptive restarts."""
# EVOLVE-BLOCK-START
import numpy as np


class Optimizer:
    """Nelder-Mead simplex optimizer for the BBOB-style f6 (Attractive sector)."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget

        # Track global best
        best_x = None
        best_f = np.inf
        calls_used = 0

        def update_best(x, f):
            nonlocal best_x, best_f
            if f < best_f:
                best_f = f
                best_x = x.copy()

        # Phase 1: Broad exploration with multiple starting points
        # Try a few different initial points to find a good basin
        start_points = [
            np.zeros(dim),
            np.array([1.0] * dim),
            np.array([-1.0] * dim),
            np.array([2.0, -2.0, 1.0, -1.0, 0.5]),
            np.array([-0.5, 0.5, -0.5, 0.5, -0.5]),
        ]

        # Evaluate a subset of start points to find the best region
        explored = []
        for sp in start_points:
            sp = np.clip(sp, -5, 5)
            f_sp = func(sp)
            calls_used += 1
            explored.append((f_sp, sp))
            update_best(sp, f_sp)
            if calls_used >= budget:
                break

        # Sort by function value and pick the best as our starting point
        explored.sort(key=lambda x: x[0])
        x0 = explored[0][1].copy()
        f0 = explored[0][0]

        # Phase 2: Nelder-Mead from the best starting point
        # Build initial simplex around x0
        step = 0.5
        simplex = [x0.copy()]
        simplex_f = [f0]

        for i in range(dim):
            xi = x0.copy()
            xi[i] += step
            xi = np.clip(xi, -5, 5)
            simplex.append(xi)
            f_i = func(xi)
            calls_used += 1
            simplex_f.append(f_i)
            update_best(xi, f_i)

        if calls_used >= budget:
            return best_x, best_f

        # Nelder-Mead parameters (adapted for better performance)
        alpha = 1.0  # reflection
        gamma = 2.0  # expansion
        rho = 0.5    # contraction
        sigma = 0.5  # shrink

        # Sort simplex by function value
        def sort_simplex():
            nonlocal simplex, simplex_f
            idx = np.argsort(simplex_f)
            simplex = [simplex[i] for i in idx]
            simplex_f = [simplex_f[i] for i in idx]

        sort_simplex()

        # Adaptive restart: if no improvement for N iterations, restart from best point
        no_improve_count = 0
        max_no_improve = 20
        prev_best_f = best_f

        while calls_used < budget:
            # Check convergence
            f_spread = simplex_f[-1] - simplex_f[0]
            x_spread = np.max(np.abs(np.array(simplex) - np.array(simplex[0])))
            if f_spread < 1e-8 and x_spread < 1e-6:
                # Stagnation detected - adaptive restart
                no_improve_count += 1
                if no_improve_count > max_no_improve:
                    # Restart from best point with new random perturbation
                    perturbation_scale = 0.3 * (1.0 / (1.0 + 0.1 * no_improve_count))
                    x_new = best_x + perturbation_scale * np.random.randn(dim)
                    x_new = np.clip(x_new, -5, 5)
                    f_new = func(x_new)
                    calls_used += 1
                    update_best(x_new, f_new)
                    if calls_used >= budget:
                        break

                    # Rebuild simplex around new point
                    simplex = [x_new.copy()]
                    simplex_f = [f_new]
                    step = perturbation_scale
                    for i in range(dim):
                        xi = x_new.copy()
                        xi[i] += step
                        xi = np.clip(xi, -5, 5)
                        f_i = func(xi)
                        calls_used += 1
                        simplex_f.append(f_i)
                        update_best(xi, f_i)
                        if calls_used >= budget:
                            break
                    if calls_used >= budget:
                        break
                    sort_simplex()
                    no_improve_count = 0
                    prev_best_f = best_f
                    continue
                else:
                    # Small perturbation to escape local minimum
                    simplex[0] = simplex[0] + 0.05 * np.random.randn(dim)
                    simplex[0] = np.clip(simplex[0], -5, 5)
                    simplex_f[0] = func(simplex[0])
                    calls_used += 1
                    update_best(simplex[0], simplex_f[0])
                    sort_simplex()
                    continue

            # Compute centroid of all points except the worst
            centroid = np.mean(simplex[:-1], axis=0)

            # Reflection
            x_r = centroid + alpha * (centroid - simplex[-1])
            x_r = np.clip(x_r, -5, 5)
            f_r = func(x_r)
            calls_used += 1
            update_best(x_r, f_r)

            if f_r < simplex_f[0]:
                # Try expansion
                if calls_used >= budget:
                    break
                x_e = centroid + gamma * (x_r - centroid)
                x_e = np.clip(x_e, -5, 5)
                f_e = func(x_e)
                calls_used += 1
                update_best(x_e, f_e)
                if f_e < f_r:
                    simplex[-1] = x_e
                    simplex_f[-1] = f_e
                else:
                    simplex[-1] = x_r
                    simplex_f[-1] = f_r
            elif f_r < simplex_f[-2]:
                # Accept reflection
                simplex[-1] = x_r
                simplex_f[-1] = f_r
            else:
                # Contraction
                if calls_used >= budget:
                    break
                # Outside contraction if f_r < f_w, else inside contraction
                if f_r < simplex_f[-1]:
                    x_c = centroid + rho * (x_r - centroid)
                else:
                    x_c = centroid - rho * (simplex[-1] - centroid)
                x_c = np.clip(x_c, -5, 5)
                f_c = func(x_c)
                calls_used += 1
                update_best(x_c, f_c)

                if f_c < min(f_r, simplex_f[-1]):
                    simplex[-1] = x_c
                    simplex_f[-1] = f_c
                else:
                    # Shrink
                    for i in range(1, dim + 1):
                        if calls_used >= budget:
                            break
                        simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                        simplex[i] = np.clip(simplex[i], -5, 5)
                        simplex_f[i] = func(simplex[i])
                        calls_used += 1
                        update_best(simplex[i], simplex_f[i])
                    if calls_used >= budget:
                        break

            # Update no-improvement counter
            if best_f < prev_best_f - 1e-10:
                no_improve_count = 0
                prev_best_f = best_f
            else:
                no_improve_count += 1

            sort_simplex()

        return best_x, best_f
# EVOLVE-BLOCK-END
