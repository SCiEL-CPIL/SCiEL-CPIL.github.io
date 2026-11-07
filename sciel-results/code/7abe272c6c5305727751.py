"""Rewritten optimizer for BBOB f9 (rotated Rosenbrock)."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """
    Two-phase optimizer for rotated Rosenbrock (f9):
    Phase 1: CMA-ES to find a good basin (fast global exploration).
    Phase 2: Nelder-Mead with adaptive restarts for fine-tuning in the valley.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        deadline = time.time() + 280  # safety margin before 300s limit

        # Track best solution found so far
        best_x = None
        best_f = np.inf
        calls_made = 0

        def count_call(x):
            nonlocal calls_made, best_x, best_f
            fx = func(x)
            calls_made += 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            return fx

        # --- Phase 1: CMA-ES ---
        # Use CMA-ES from scipy for robust global search
        from scipy.optimize import minimize

        # Initialize with a point in the center
        x0 = np.zeros(dim)
        f0 = count_call(x0)

        # CMA-ES parameters
        maxiter = min(200, max(50, (budget - 10) // 3))
        # Use a reasonable sigma for the initial population
        # The box is [-5,5]^5, so sigma ~ 1.5 is reasonable
        options = {
            'maxiter': maxiter,
            'maxfev': min(150, budget - 50),
            'sigma0': 1.5,
            'bounds': [(-5.0, 5.0)] * dim,
            'popsize': 12,
        }

        try:
            res = minimize(
                count_call,
                x0,
                method='CMA',
                options=options
            )
            if res.fun < best_f:
                best_f = res.fun
                best_x = res.x.copy()
        except Exception:
            pass

        # Check if we're running out of time or budget
        if time.time() > deadline or calls_made >= budget - 10:
            return best_x, best_f

        # --- Phase 2: Nelder-Mead with restarts ---
        # Use the best point from CMA-ES as starting point
        x_nm = best_x.copy()
        f_nm = best_f

        # Nelder-Mead with adaptive restarts
        nm_budget = min(100, budget - calls_made - 5)
        restarts = 3
        nm_calls = 0

        for restart in range(restarts):
            if nm_calls >= nm_budget or time.time() > deadline:
                break

            # Create initial simplex
            simplex = [x_nm.copy()]
            for i in range(dim):
                pt = x_nm.copy()
                pt[i] += 0.5 if pt[i] < 5.0 else -0.5
                pt[i] = np.clip(pt[i], -5.0, 5.0)
                simplex.append(pt)

            # Evaluate initial simplex
            simplex_f = []
            for pt in simplex:
                if nm_calls >= nm_budget or time.time() > deadline:
                    break
                simplex_f.append(count_call(pt))
                nm_calls += 1

            if len(simplex_f) < dim + 1:
                break

            # Nelder-Mead iterations
            max_nm_iter = 80
            for it in range(max_nm_iter):
                if nm_calls >= nm_budget or time.time() > deadline:
                    break

                # Sort simplex by function value
                order = np.argsort(simplex_f)
                simplex = [simplex[i] for i in order]
                simplex_f = [simplex_f[i] for i in order]

                # Check convergence
                if abs(simplex_f[0] - simplex_f[-1]) < 1e-10:
                    break

                centroid = np.mean(simplex[:-1], axis=0)

                # Reflection
                alpha = 1.0
                x_ref = centroid + alpha * (centroid - simplex[-1])
                x_ref = np.clip(x_ref, -5.0, 5.0)
                f_ref = count_call(x_ref)
                nm_calls += 1

                if f_ref < simplex_f[0]:
                    # Expansion
                    gamma = 2.0
                    x_exp = centroid + gamma * (centroid - simplex[-1])
                    x_exp = np.clip(x_exp, -5.0, 5.0)
                    f_exp = count_call(x_exp)
                    nm_calls += 1

                    if f_exp < f_ref:
                        simplex[-1] = x_exp
                        simplex_f[-1] = f_exp
                    else:
                        simplex[-1] = x_ref
                        simplex_f[-1] = f_ref
                elif f_ref < simplex_f[-2]:
                    simplex[-1] = x_ref
                    simplex_f[-1] = f_ref
                else:
                    # Contraction
                    rho = 0.5
                    x_con = centroid + rho * (simplex[-1] - centroid)
                    x_con = np.clip(x_con, -5.0, 5.0)
                    f_con = count_call(x_con)
                    nm_calls += 1

                    if f_con < simplex_f[-1]:
                        simplex[-1] = x_con
                        simplex_f[-1] = f_con
                    else:
                        # Shrink
                        for i in range(1, len(simplex)):
                            simplex[i] = simplex[0] + 0.5 * (simplex[i] - simplex[0])
                            simplex[i] = np.clip(simplex[i], -5.0, 5.0)
                            if nm_calls >= nm_budget or time.time() > deadline:
                                break
                            simplex_f[i] = count_call(simplex[i])
                            nm_calls += 1
                        if nm_calls >= nm_budget or time.time() > deadline:
                            break

            # Update best
            if simplex_f[0] < best_f:
                best_f = simplex_f[0]
                best_x = simplex[0].copy()

            # Prepare for next restart: perturb slightly
            if restart < restarts - 1:
                rng = np.random.default_rng(42 + restart)
                x_nm = best_x + 0.1 * rng.standard_normal(dim)
                x_nm = np.clip(x_nm, -5.0, 5.0)
                # Don't count this as a new call, just set as starting point
                f_nm = best_f

        # Final check
        if best_x is None:
            best_x = x0
            best_f = f0

        return best_x, best_f
# EVOLVE-BLOCK-END
