"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


class Optimizer:
    """Nelder-Mead simplex optimization with adaptive restarts and local polishing."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        rng = np.random.default_rng(42)

        # Track best solution found overall
        best_x = None
        best_f = np.inf
        calls_made = 0

        def evaluate(x):
            nonlocal calls_made, best_x, best_f
            x = np.clip(x, -5.0, 5.0)
            fx = func(x)
            calls_made += 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            return fx

        # Nelder-Mead parameters
        alpha = 1.0    # reflection
        gamma = 2.0    # expansion
        rho = 0.5      # contraction
        sigma_nm = 0.5 # shrink

        # Initial simplex size
        initial_delta = 2.0

        # Adaptive step size for restarts
        current_delta = initial_delta

        # Strategy: Run Nelder-Mead until stagnation or budget exhaustion.
        # If stagnated, restart with a perturbed simplex around the best point.
        # Reserve some budget for final local coordinate search.

        local_search_budget = 50
        nm_budget = budget - local_search_budget

        max_restarts = 10
        stagnation_limit = 20 # Number of iterations without improvement to trigger restart

        for restart in range(max_restarts):
            if calls_made >= nm_budget:
                break

            # Initialize simplex around the current best point (or random for first restart)
            if best_x is None:
                center = rng.uniform(-4.0, 4.0, dim)
            else:
                center = best_x.copy()
                # Add a small perturbation to avoid exact same starting point
                center += rng.normal(0, current_delta * 0.1, dim)
                center = np.clip(center, -5.0, 5.0)

            # Create simplex
            simplex = np.zeros((dim + 1, dim))
            simplex[0] = center
            for i in range(dim):
                simplex[i + 1] = center.copy()
                simplex[i + 1][i] += current_delta

            # Evaluate initial simplex
            fx_values = np.zeros(dim + 1)
            for i in range(dim + 1):
                if calls_made >= nm_budget:
                    break
                fx_values[i] = evaluate(simplex[i])

            if calls_made >= nm_budget:
                break

            # Nelder-Mead iterations
            no_improve_count = 0
            last_best_f = best_f

            while calls_made < nm_budget:
                # Sort simplex by function value
                order = np.argsort(fx_values)
                simplex = simplex[order]
                fx_values = fx_values[order]

                # Check for convergence
                f_range = fx_values[-1] - fx_values[0]
                if f_range < 1e-10:
                    break

                # Compute centroid of all but worst point
                centroid = np.mean(simplex[:-1], axis=0)

                # Reflection
                x_r = centroid + alpha * (centroid - simplex[-1])
                x_r = np.clip(x_r, -5.0, 5.0)
                f_r = evaluate(x_r)

                if calls_made >= nm_budget:
                    break

                improved = False
                if f_r < fx_values[0]:
                    # Expansion
                    x_e = centroid + gamma * (x_r - centroid)
                    x_e = np.clip(x_e, -5.0, 5.0)
                    f_e = evaluate(x_e)

                    if calls_made >= nm_budget:
                        break

                    if f_e < f_r:
                        simplex[-1] = x_e
                        fx_values[-1] = f_e
                    else:
                        simplex[-1] = x_r
                        fx_values[-1] = f_r
                elif f_r < fx_values[-2]:
                    # Simple reflection
                    simplex[-1] = x_r
                    fx_values[-1] = f_r
                else:
                    # Contraction
                    if f_r < fx_values[-1]:
                        # Outer contraction
                        x_c = centroid + rho * (x_r - centroid)
                    else:
                        # Inner contraction
                        x_c = centroid - rho * (x_r - centroid)
                    x_c = np.clip(x_c, -5.0, 5.0)
                    f_c = evaluate(x_c)

                    if calls_made >= nm_budget:
                        break

                    if f_c < min(f_r, fx_values[-1]):
                        simplex[-1] = x_c
                        fx_values[-1] = f_c
                    else:
                        # Shrink
                        for i in range(1, dim + 1):
                            simplex[i] = simplex[0] + sigma_nm * (simplex[i] - simplex[0])
                            if calls_made < nm_budget:
                                fx_values[i] = evaluate(simplex[i])

                        if calls_made >= nm_budget:
                            break

                # Check for improvement
                if best_f < last_best_f - 1e-8:
                    no_improve_count = 0
                    last_best_f = best_f
                else:
                    no_improve_count += 1

                if no_improve_count >= stagnation_limit:
                    # Restart with larger or smaller step?
                    # If we stuck, maybe try a larger jump or smaller fine-tuning.
                    # Let's try a larger jump to escape local minima.
                    current_delta *= 1.5
                    break

            # If we finished the loop naturally without breaking due to stagnation,
            # we might have converged.
            if no_improve_count < stagnation_limit and calls_made < nm_budget:
                 # Reduce step size for next restart to fine-tune
                 current_delta *= 0.5

        # Final local coordinate search (Hill Climbing)
        if best_x is not None and calls_made < budget:
            step = 0.5
            while calls_made < budget:
                improved = False
                for i in range(dim):
                    if calls_made >= budget:
                        break
                    # Try positive step
                    x_try = best_x.copy()
                    x_try[i] += step
                    x_try = np.clip(x_try, -5.0, 5.0)
                    f_try = evaluate(x_try)
                    if f_try < best_f:
                        best_x = x_try
                        best_f = f_try
                        improved = True

                if calls_made >= budget:
                    break

                # Try negative step
                for i in range(dim):
                    if calls_made >= budget:
                        break
                    x_try = best_x.copy()
                    x_try[i] -= step
                    x_try = np.clip(x_try, -5.0, 5.0)
                    f_try = evaluate(x_try)
                    if f_try < best_f:
                        best_x = x_try
                        best_f = f_try
                        improved = True

                if not improved:
                    step *= 0.5
                    if step < 1e-6:
                        break

        return best_x, best_f
# EVOLVE-BLOCK-END
