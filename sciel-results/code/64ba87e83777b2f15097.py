"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


class Optimizer:
    """Nelder-Mead simplex optimization with restarts and adaptive step sizes."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        rng = np.random.default_rng(0)

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
        initial_delta = 1.0

        # We'll do multiple restarts if budget allows
        # Each restart uses a portion of the budget
        # Let's use a strategy: try Nelder-Mead for some budget, then restart if stuck

        # Determine how many restarts to attempt
        # For 5D and 500 budget, let's do several short runs
        # Each Nelder-Mead iteration costs (dim+1) function evaluations for the initial simplex
        # Then each step costs 1-3 evaluations

        # Strategy: run Nelder-Mead with a budget of ~100-150 calls per attempt,
        # then restart from the best point with a fresh simplex

        restart_budget = 120
        max_restarts = min(5, budget // restart_budget)

        for restart in range(max_restarts):
            if calls_made >= budget:
                break

            # Initialize simplex around the current best point (or random for first restart)
            if best_x is None:
                center = rng.uniform(-4.0, 4.0, dim)
            else:
                center = best_x.copy()

            # Create simplex
            simplex = np.zeros((dim + 1, dim))
            simplex[0] = center
            for i in range(dim):
                simplex[i + 1] = center.copy()
                simplex[i + 1][i] += initial_delta

            # Evaluate initial simplex
            fx_values = np.zeros(dim + 1)
            for i in range(dim + 1):
                if calls_made >= budget:
                    break
                fx_values[i] = evaluate(simplex[i])

            if calls_made >= budget:
                break

            # Nelder-Mead iterations
            while calls_made < budget:
                # Sort simplex by function value
                order = np.argsort(fx_values)
                simplex = simplex[order]
                fx_values = fx_values[order]

                # Check for convergence
                f_range = fx_values[-1] - fx_values[0]
                if f_range < 1e-12:
                    break

                # Compute centroid of all but worst point
                centroid = np.mean(simplex[:-1], axis=0)

                # Reflection
                x_r = centroid + alpha * (centroid - simplex[-1])
                x_r = np.clip(x_r, -5.0, 5.0)
                f_r = evaluate(x_r)

                if calls_made >= budget:
                    break

                if f_r < fx_values[0]:
                    # Expansion
                    x_e = centroid + gamma * (x_r - centroid)
                    x_e = np.clip(x_e, -5.0, 5.0)
                    f_e = evaluate(x_e)

                    if calls_made >= budget:
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

                    if calls_made >= budget:
                        break

                    if f_c < min(f_r, fx_values[-1]):
                        simplex[-1] = x_c
                        fx_values[-1] = f_c
                    else:
                        # Shrink
                        for i in range(1, dim + 1):
                            simplex[i] = simplex[0] + sigma_nm * (simplex[i] - simplex[0])
                            if calls_made < budget:
                                fx_values[i] = evaluate(simplex[i])

                        if calls_made >= budget:
                            break

        return best_x, best_f
# EVOLVE-BLOCK-END
