"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


class Optimizer:
    """Nelder-Mead simplex optimizer with restarts and adaptive step size."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        rng = np.random.default_rng(42)
        best_x = None
        best_f = np.inf
        calls_used = 0
        max_calls = self.budget

        def count_call(x):
            nonlocal calls_used
            calls_used += 1
            return func(x)

        def nelder_mead(start_x, remaining_budget):
            """Run Nelder-Mead simplex optimization from start_x."""
            nonlocal best_x, best_f, calls_used

            if remaining_budget < self.dim + 2:
                return

            # Initialize simplex
            simplex = np.zeros((self.dim + 1, self.dim))
            simplex[0] = start_x
            for i in range(1, self.dim + 1):
                simplex[i] = start_x.copy()
                simplex[i, i - 1] += 0.5 if start_x[i - 1] < 0 else -0.5

            # Clip to bounds
            simplex = np.clip(simplex, -5, 5)

            # Evaluate initial simplex
            fvals = np.zeros(self.dim + 1)
            for i in range(self.dim + 1):
                if calls_used >= max_calls:
                    break
                fvals[i] = count_call(simplex[i])

            # Update global best
            min_idx = np.argmin(fvals)
            if fvals[min_idx] < best_f:
                best_f = fvals[min_idx]
                best_x = simplex[min_idx].copy()

            # Nelder-Mead parameters
            alpha = 1.0  # reflection
            gamma = 2.0  # expansion
            rho = 0.5    # contraction
            sigma_nm = 0.5  # shrink

            max_iter = remaining_budget - (self.dim + 1)
            for _ in range(max(0, max_iter)):
                if calls_used >= max_calls:
                    break

                # Sort simplex by function values
                order = np.argsort(fvals)
                simplex = simplex[order]
                fvals = fvals[order]

                # Centroid of all but worst
                centroid = np.mean(simplex[:-1], axis=0)

                # Reflection
                x_ref = centroid + alpha * (centroid - simplex[-1])
                x_ref = np.clip(x_ref, -5, 5)
                f_ref = count_call(x_ref)

                if f_ref < fvals[0]:
                    # Expansion
                    x_exp = centroid + gamma * (x_ref - centroid)
                    x_exp = np.clip(x_exp, -5, 5)
                    f_exp = count_call(x_exp)
                    if f_exp < f_ref:
                        simplex[-1] = x_exp
                        fvals[-1] = f_exp
                    else:
                        simplex[-1] = x_ref
                        fvals[-1] = f_ref
                elif f_ref < fvals[-2]:
                    # Accept reflection
                    simplex[-1] = x_ref
                    fvals[-1] = f_ref
                else:
                    # Contraction
                    if f_ref < fvals[-1]:
                        # Outside contraction
                        x_con = centroid + rho * (x_ref - centroid)
                    else:
                        # Inside contraction
                        x_con = centroid - rho * (simplex[-1] - centroid)
                    x_con = np.clip(x_con, -5, 5)
                    f_con = count_call(x_con)

                    if f_con <= min(f_ref, fvals[-1]):
                        simplex[-1] = x_con
                        fvals[-1] = f_con
                    else:
                        # Shrink
                        for i in range(1, self.dim + 1):
                            simplex[i] = simplex[0] + sigma_nm * (simplex[i] - simplex[0])
                            simplex[i] = np.clip(simplex[i], -5, 5)
                            fvals[i] = count_call(simplex[i])

                # Update global best
                min_idx = np.argmin(fvals)
                if fvals[min_idx] < best_f:
                    best_f = fvals[min_idx]
                    best_x = simplex[min_idx].copy()

                # Check for convergence
                if np.max(fvals) - np.min(fvals) < 1e-10:
                    break

            return simplex, fvals

        # Multi-start Nelder-Mead
        # First try a few random starts with different simplex sizes
        n_starts = min(5, max(1, (max_calls - 10) // (self.dim + 10)))
        starts_per_start = (max_calls - n_starts * 2) // n_starts

        for s in range(n_starts):
            if calls_used >= max_calls:
                break

            # Generate start point
            if s == 0:
                start_x = np.zeros(self.dim)
            else:
                start_x = rng.uniform(-4, 4, self.dim)

            remaining = max_calls - calls_used
            if remaining < self.dim + 2:
                break

            # Run Nelder-Mead from this start
            simplex, fvals = nelder_mead(start_x, remaining)

            # If we have budget left, do a local refinement around the best point
            if calls_used < max_calls and best_x is not None:
                # Small-scale Nelder-Mead around best
                refined_start = best_x.copy()
                remaining2 = max_calls - calls_used
                if remaining2 >= self.dim + 2:
                    # Use smaller initial simplex
                    simplex2 = np.zeros((self.dim + 1, self.dim))
                    simplex2[0] = refined_start
                    for i in range(1, self.dim + 1):
                        simplex2[i] = refined_start.copy()
                        simplex2[i, i - 1] += 0.1
                    simplex2 = np.clip(simplex2, -5, 5)

                    fvals2 = np.zeros(self.dim + 1)
                    for i in range(self.dim + 1):
                        if calls_used >= max_calls:
                            break
                        fvals2[i] = count_call(simplex2[i])

                    min_idx2 = np.argmin(fvals2)
                    if fvals2[min_idx2] < best_f:
                        best_f = fvals2[min_idx2]
                        best_x = simplex2[min_idx2].copy()

        return best_x, best_f
# EVOLVE-BLOCK-END
