"""Rewritten optimizer: multi-restart adaptive (1+1)-ES + Nelder-Mead polish."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-restart adaptive (1+1)-ES with per-dim sigma, plus NM polish."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 275.0  # safety margin
        dim = self.dim
        budget = self.budget

        best_x = None
        best_f = np.inf

        def update_best(x, f):
            nonlocal best_x, best_f
            if f < best_f:
                best_x = x.copy()
                best_f = f

        rng = np.random.default_rng(12345)

        # --- Phase 1: Multi-restart (1+1)-ES with per-dimension sigma ---
        n_restarts = 8
        budget_per_restart = max(20, (budget - 50) // n_restarts)  # reserve 50 for NM

        for restart in range(n_restarts):
            if time.time() > deadline:
                break

            # Choose initial point and sigma
            if restart == 0:
                x = np.zeros(dim)
                sigma = np.full(dim, 3.0)
            elif restart == 1:
                x = rng.uniform(-4, 4, dim)
                sigma = np.full(dim, 2.0)
            elif restart == 2:
                x = rng.uniform(-5, 5, dim)
                sigma = np.full(dim, 1.5)
            elif restart == 3:
                x = rng.uniform(-3, 3, dim)
                sigma = np.full(dim, 1.0)
            elif restart == 4:
                x = rng.uniform(-5, 5, dim)
                sigma = np.full(dim, 0.5)
            elif restart == 5:
                x = rng.uniform(-2, 2, dim)
                sigma = np.full(dim, 0.3)
            elif restart == 6:
                x = rng.uniform(-5, 5, dim)
                sigma = np.random.uniform(0.1, 2.0, dim)
            else:
                x = rng.uniform(-5, 5, dim)
                sigma = np.random.uniform(0.05, 1.0, dim)

            fx = func(x)
            update_best(x, fx)
            calls_used = 1

            # Track success rate for sigma adaptation
            successes = 0
            window = 10

            for step in range(budget_per_restart):
                if time.time() > deadline:
                    break
                if calls_used >= budget:
                    break

                # Generate candidate
                z = rng.standard_normal(dim)
                y = np.clip(x + sigma * z, -5, 5)
                fy = func(y)
                calls_used += 1

                if fy < fx:
                    # Success: move to y, increase sigma slightly
                    x, fx = y, fy
                    update_best(x, fx)
                    sigma *= 1.1
                    successes += 1
                else:
                    # Failure: decrease sigma
                    sigma *= 0.9

                # Adapt sigma based on success rate over window
                if (step + 1) % window == 0:
                    rate = successes / window
                    if rate > 0.5:
                        sigma *= 1.2
                    elif rate < 0.1:
                        sigma *= 0.7
                    successes = 0

                # Clamp sigma
                sigma = np.clip(sigma, 1e-6, 5.0)

        # --- Phase 2: Nelder-Mead polish from best point ---
        if time.time() < deadline and calls_used < budget and best_x is not None:
            x_nm = best_x.copy()
            fx_nm = best_f

            # Initial simplex
            simplex = [x_nm.copy()]
            for i in range(dim):
                xi = x_nm.copy()
                step = 0.1 if abs(xi[i]) > 1e-4 else 0.1
                xi[i] += step
                xi = np.clip(xi, -5, 5)
                simplex.append(xi)

            f_simplex = [func(s) for s in simplex]
            calls_used += dim

            # Nelder-Mead iterations
            max_nm_iter = 200
            for it in range(max_nm_iter):
                if time.time() > deadline:
                    break
                if calls_used >= budget:
                    break

                # Sort simplex
                order = np.argsort(f_simplex)
                simplex = [simplex[i] for i in order]
                f_simplex = [f_simplex[i] for i in order]

                # Update best
                update_best(simplex[0], f_simplex[0])

                # Centroid of all but worst
                centroid = np.mean(simplex[:-1], axis=0)

                # Reflection
                alpha = 1.0
                x_r = np.clip(centroid + alpha * (centroid - simplex[-1]), -5, 5)
                f_r = func(x_r)
                calls_used += 1

                if f_r < f_simplex[0]:
                    # Expansion
                    gamma = 2.0
                    x_e = np.clip(centroid + gamma * (centroid - simplex[-1]), -5, 5)
                    f_e = func(x_e)
                    calls_used += 1
                    if f_e < f_r:
                        simplex[-1] = x_e
                        f_simplex[-1] = f_e
                    else:
                        simplex[-1] = x_r
                        f_simplex[-1] = f_r
                elif f_r < f_simplex[-2]:
                    simplex[-1] = x_r
                    f_simplex[-1] = f_r
                else:
                    # Contraction
                    rho = 0.5
                    x_c = np.clip(centroid + rho * (simplex[-1] - centroid), -5, 5)
                    f_c = func(x_c)
                    calls_used += 1
                    if f_c < f_simplex[-1]:
                        simplex[-1] = x_c
                        f_simplex[-1] = f_c
                    else:
                        # Shrink
                        for i in range(1, dim + 1):
                            simplex[i] = np.clip(simplex[0] + 0.5 * (simplex[i] - simplex[0]), -5, 5)
                            f_simplex[i] = func(simplex[i])
                            calls_used += 1

                # Check convergence
                if max(f_simplex) - min(f_simplex) < 1e-10:
                    break

        return best_x, best_f
# EVOLVE-BLOCK-END
