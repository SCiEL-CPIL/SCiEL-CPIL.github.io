"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize


class Optimizer:
    """Deterministic gradient-free trust-region method (Nelder-Mead) with
    adaptive restarts and multi-start to exploit local convexity quickly."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget

        # We'll do a small number of restarts with different initial simplices.
        # For dim=5, Nelder-Mead needs dim+1 = 6 points per simplex.
        # Each iteration typically uses 1-2 function evaluations (depending on reflection/expansion/contraction).
        # Let's allocate budget across several restarts.

        rng = np.random.default_rng(42)

        best_x = None
        best_f = np.inf
        calls_made = 0

        def wrapped_func(x):
            nonlocal calls_made
            if calls_made >= budget:
                return np.inf
            calls_made += 1
            return func(x)

        # Multi-start: try a few different starting points
        # For dim=5, let's do 3-5 restarts
        n_restarts = 3
        budget_per_restart = budget // n_restarts

        for restart in range(n_restarts):
            if calls_made >= budget:
                break

            # Choose initial center
            if restart == 0:
                center = np.zeros(dim)
            elif restart == 1:
                center = rng.uniform(-3, 3, dim)
            else:
                center = rng.uniform(-4, 4, dim)

            # Create initial simplex
            # Standard Nelder-Mead simplex initialization
            simplex = np.zeros((dim + 1, dim))
            simplex[0] = center
            for i in range(dim):
                simplex[i + 1] = center.copy()
                simplex[i + 1][i] += 0.5  # small perturbation

            # Evaluate initial simplex
            f_values = np.zeros(dim + 1)
            for i in range(dim + 1):
                f_values[i] = wrapped_func(simplex[i])
                if calls_made >= budget:
                    break

            if calls_made >= budget:
                # Find best so far
                valid_idx = np.where(np.isfinite(f_values))[0]
                if len(valid_idx) > 0:
                    best_idx = valid_idx[np.argmin(f_values[valid_idx])]
                    if f_values[best_idx] < best_f:
                        best_f = f_values[best_idx]
                        best_x = simplex[best_idx].copy()
                break

            # Run Nelder-Mead for remaining budget in this restart
            # We'll implement a manual loop to control budget precisely
            max_iter = budget - calls_made
            alpha, gamma, rho, sigma_nm = 1.0, 2.0, 0.5, 0.5

            for _ in range(max_iter):
                if calls_made >= budget:
                    break

                # Sort simplex by function value
                order = np.argsort(f_values)
                simplex = simplex[order]
                f_values = f_values[order]

                # Check convergence
                f_range = f_values[-1] - f_values[0]
                if f_range < 1e-10:
                    break

                # Centroid of all but worst point
                centroid = np.mean(simplex[:-1], axis=0)

                # Reflection
                x_r = centroid + alpha * (centroid - simplex[-1])
                f_r = wrapped_func(x_r)
                if calls_made >= budget:
                    break

                if f_values[0] <= f_r < f_values[-2]:
                    # Replace worst with reflection
                    simplex[-1] = x_r
                    f_values[-1] = f_r
                elif f_r < f_values[0]:
                    # Expansion
                    x_e = centroid + gamma * (x_r - centroid)
                    f_e = wrapped_func(x_e)
                    if calls_made >= budget:
                        break
                    if f_e < f_r:
                        simplex[-1] = x_e
                        f_values[-1] = f_e
                    else:
                        simplex[-1] = x_r
                        f_values[-1] = f_r
                else:
                    # Contraction
                    if f_r < f_values[-1]:
                        # Outside contraction
                        x_c = centroid + rho * (x_r - centroid)
                    else:
                        # Inside contraction
                        x_c = centroid - rho * (simplex[-1] - centroid)
                    f_c = wrapped_func(x_c)
                    if calls_made >= budget:
                        break

                    if f_c < min(f_r, f_values[-1]):
                        simplex[-1] = x_c
                        f_values[-1] = f_c
                    else:
                        # Shrink
                        for i in range(1, dim + 1):
                            simplex[i] = simplex[0] + sigma_nm * (simplex[i] - simplex[0])
                            f_values[i] = wrapped_func(simplex[i])
                            if calls_made >= budget:
                                break

                # Track best
                best_idx = np.argmin(f_values)
                if f_values[best_idx] < best_f:
                    best_f = f_values[best_idx]
                    best_x = simplex[best_idx].copy()

                if calls_made >= budget:
                    break

        if best_x is None:
            # Fallback: return a reasonable point
            best_x = np.zeros(dim)
            best_f = func(best_x)

        return best_x, best_f
# EVOLVE-BLOCK-END
