"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """
    Two-phase optimizer for BBOB f13 (Sharp Ridge):
    Phase 1: Random exploration to find a good starting point.
    Phase 2: Nelder-Mead on the full 5D space, which handles the rotated
             sharp ridge structure well due to its adaptive simplex.
    """

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0  # safety margin before 300s limit
        rng = np.random.default_rng(42)
        dim = self.dim
        budget = self.budget

        best_x = None
        best_f = float('inf')
        calls = 0

        # Phase 1: Initial random sampling (~15% of budget, capped at 60)
        init_budget = min(60, max(20, budget // 8))
        for _ in range(init_budget):
            if calls >= budget or time.time() > deadline:
                break
            x = rng.uniform(-5, 5, dim)
            fx = func(x)
            calls += 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()

        if best_x is None:
            x = np.zeros(dim)
            fx = func(x)
            calls = 1
            best_f = fx
            best_x = x.copy()

        # Phase 2: Nelder-Mead with adaptive restarts
        # Nelder-Mead works well on rotated sharp ridges because its simplex
        # adapts to the local geometry.
        remaining = budget - calls
        if remaining <= 0 or time.time() > deadline:
            return best_x, best_f

        # Run Nelder-Mead for the remaining budget
        # We'll implement a custom Nelder-Mead to control the call count
        # and integrate with our deadline guard.

        def nelder_mead(func, x0, budget_left, deadline, dim, rng):
            """
            Custom Nelder-Mead that tracks calls and respects budget/deadline.
            Returns (best_x, best_f, calls_made).
            """
            # Initialize simplex: x0 and x0 + alpha * e_i for each dimension
            alpha = 0.5
            simplex = [x0.copy()]
            for i in range(dim):
                xi = x0.copy()
                xi[i] += alpha
                simplex.append(np.clip(xi, -5, 5))

            # Evaluate all simplex vertices
            fx_list = []
            calls_made = 0
            for v in simplex:
                if calls_made >= budget_left or time.time() > deadline:
                    break
                fv = func(v)
                calls_made += 1
                fx_list.append(fv)

            if calls_made == 0:
                return x0, float('inf'), 0

            # Pad if we couldn't evaluate all vertices
            while len(fx_list) < len(simplex):
                fx_list.append(fx_list[-1])

            # Nelder-Mead parameters
            alpha_nm = 1.0  # reflection
            gamma = 2.0     # expansion
            rho = 0.5       # contraction
            sigma = 0.5     # shrink

            max_iters = budget_left
            for iteration in range(max_iters):
                if calls_made >= budget_left or time.time() > deadline:
                    break

                # Sort simplex by function value
                order = np.argsort(fx_list)
                simplex = [simplex[i].copy() for i in order]
                fx_list = [fx_list[i] for i in order]

                # Check convergence
                if fx_list[-1] - fx_list[0] < 1e-12:
                    break

                # Centroid of all but worst
                centroid = np.mean(simplex[:-1], axis=0)

                # Reflection
                x_ref = centroid + alpha_nm * (centroid - simplex[-1])
                x_ref = np.clip(x_ref, -5, 5)
                if calls_made >= budget_left or time.time() > deadline:
                    break
                f_ref = func(x_ref)
                calls_made += 1

                if f_ref < fx_list[0]:
                    # Expansion
                    x_exp = centroid + gamma * (x_ref - centroid)
                    x_exp = np.clip(x_exp, -5, 5)
                    if calls_made >= budget_left or time.time() > deadline:
                        simplex[-1] = x_ref
                        fx_list[-1] = f_ref
                        break
                    f_exp = func(x_exp)
                    calls_made += 1
                    if f_exp < f_ref:
                        simplex[-1] = x_exp
                        fx_list[-1] = f_exp
                    else:
                        simplex[-1] = x_ref
                        fx_list[-1] = f_ref
                elif f_ref < fx_list[-2]:
                    # Simple reflection
                    simplex[-1] = x_ref
                    fx_list[-1] = f_ref
                else:
                    # Contraction
                    if f_ref < fx_list[-1]:
                        # Outside contraction
                        x_con = centroid + rho * (x_ref - centroid)
                    else:
                        # Inside contraction
                        x_con = centroid - rho * (x_ref - centroid)
                    x_con = np.clip(x_con, -5, 5)
                    if calls_made >= budget_left or time.time() > deadline:
                        break
                    f_con = func(x_con)
                    calls_made += 1
                    if f_con <= min(f_ref, fx_list[-1]):
                        simplex[-1] = x_con
                        fx_list[-1] = f_con
                    else:
                        # Shrink
                        for i in range(1, len(simplex)):
                            simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                            simplex[i] = np.clip(simplex[i], -5, 5)
                            if calls_made >= budget_left or time.time() > deadline:
                                break
                            fx_list[i] = func(simplex[i])
                            calls_made += 1
                        if calls_made >= budget_left or time.time() > deadline:
                            break

            # Find best in final simplex
            best_idx = np.argmin(fx_list)
            return simplex[best_idx], fx_list[best_idx], calls_made

        # Run Nelder-Mead with the remaining budget
        nm_x, nm_f, nm_calls = nelder_mead(
            func, best_x, remaining, deadline, dim, rng
        )

        calls += nm_calls
        if nm_f < best_f:
            best_f = nm_f
            best_x = nm_x.copy()

        return best_x, best_f
# EVOLVE-BLOCK-END
