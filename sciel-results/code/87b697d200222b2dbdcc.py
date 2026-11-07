"""Direct optimizer for BBOB-style f6 (Attractive sector)."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-restart Nelder-Mead simplex optimizer for BBOB f6."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        t0 = time.time()
        deadline = t0 + 290.0  # 10s safety margin before 300s limit

        rng = np.random.default_rng(42)
        dim = self.dim
        budget = self.budget

        best_x = None
        best_f = np.inf
        calls_made = 0

        def eval_func(x):
            nonlocal calls_made, best_f, best_x
            x = np.clip(np.asarray(x, dtype=np.float64), -5.0, 5.0)
            fx = float(func(x))
            calls_made += 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            return fx

        def nelder_mead(start_x, max_calls):
            """Run Nelder-Mead from start_x with at most max_calls evaluations."""
            nonlocal calls_made, best_f, best_x, deadline

            # Initialize simplex: start point + dim perturbed points
            simplex = [start_x.copy()]
            for i in range(dim):
                v = start_x.copy()
                # Perturb in coordinate i, staying within bounds
                if start_x[i] < 4.5:
                    v[i] = min(start_x[i] + 0.5, 5.0)
                else:
                    v[i] = max(start_x[i] - 0.5, -5.0)
                simplex.append(v)

            # Evaluate initial simplex
            f_vals = [eval_func(x) for x in simplex]

            # Nelder-Mead parameters
            alpha = 1.0    # reflection
            gamma = 2.0    # expansion
            rho = 0.5      # contraction
            sigma = 0.5    # shrink

            while calls_made < max_calls:
                if time.time() > deadline:
                    break

                # Sort simplex by function value
                order = np.argsort(f_vals)
                simplex = [simplex[i] for i in order]
                f_vals = [f_vals[i] for i in order]

                # Centroid of all but worst point
                centroid = np.mean(simplex[:-1], axis=0)

                # Reflection
                x_r = centroid + alpha * (centroid - simplex[-1])
                x_r = np.clip(x_r, -5.0, 5.0)
                f_r = eval_func(x_r)

                if f_r < f_vals[0]:
                    # Try expansion
                    x_e = centroid + gamma * (x_r - centroid)
                    x_e = np.clip(x_e, -5.0, 5.0)
                    f_e = eval_func(x_e)
                    if f_e < f_r:
                        simplex[-1] = x_e
                        f_vals[-1] = f_e
                    else:
                        simplex[-1] = x_r
                        f_vals[-1] = f_r
                elif f_r < f_vals[-2]:
                    # Accept reflection
                    simplex[-1] = x_r
                    f_vals[-1] = f_r
                else:
                    # Contraction
                    if f_r < f_vals[-1]:
                        # Outside contraction
                        x_c = centroid + rho * (x_r - centroid)
                    else:
                        # Inside contraction
                        x_c = centroid - rho * (simplex[-1] - centroid)
                    x_c = np.clip(x_c, -5.0, 5.0)
                    f_c = eval_func(x_c)

                    if f_c < min(f_r, f_vals[-1]):
                        simplex[-1] = x_c
                        f_vals[-1] = f_c
                    else:
                        # Shrink: move all points except best toward best
                        for i in range(1, dim + 1):
                            simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                            simplex[i] = np.clip(simplex[i], -5.0, 5.0)
                            f_vals[i] = eval_func(simplex[i])

            # Update global best
            for x, f in zip(simplex, f_vals):
                if f < best_f:
                    best_f = f
                    best_x = x.copy()

        # Multi-restart strategy: 5 diverse starting points
        # Allocate budget: give more to early promising starts
        # Start points: center, corners-ish, random
        start_points = [
            np.zeros(dim),                          # center
            np.array([4.0] * dim),                  # corner +
            np.array([-4.0] * dim),                 # corner -
            rng.uniform(-4, 4, dim),                # random
            rng.uniform(-4, 4, dim),                # random
        ]

        # Budget allocation: decreasing weights
        weights = [0.30, 0.25, 0.20, 0.15, 0.10]
        budgets = [int(budget * w) for w in weights]
        # Ensure sum doesn't exceed budget
        total_alloc = sum(budgets)
        if total_alloc > budget:
            budgets[-1] = budget - sum(budgets[:-1])

        for i, (sp, bc) in enumerate(zip(start_points, budgets)):
            if calls_made >= budget:
                break
            if time.time() > deadline:
                break
            # Cap budget for this restart to remaining
            bc = min(bc, budget - calls_made)
            if bc < 3:  # Need at least dim+1 = 6 calls for simplex
                break
            nelder_mead(sp, calls_made + bc)

        # If we have remaining calls, do a final Nelder-Mead polish on best found
        if calls_made < budget and best_x is not None and time.time() < deadline:
            remaining = budget - calls_made
            if remaining >= 6:
                nelder_mead(best_x.copy(), budget)

        # Ensure we have a valid result
        if best_x is None:
            x = np.zeros(dim)
            best_f = float(func(x))
            calls_made += 1
            best_x = x

        return best_x, best_f
# EVOLVE-BLOCK-END
