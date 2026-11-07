"""Two-phase optimizer: coarse grid + Nelder-Mead with adaptive restarts."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Coarse grid search followed by Nelder-Mead with adaptive restarts."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0  # safety margin before 300s limit
        budget = self.budget
        dim = self.dim

        calls = [0]

        def evaluate(x):
            if calls[0] >= budget:
                return None
            calls[0] += 1
            val = func(x)
            return val

        # Phase 1: Coarse grid search over central region
        # Use a 3^dim grid over [-3, 3]^dim to find promising basin
        grid_points = []
        for coord in range(3 ** dim):
            x = np.zeros(dim)
            c = coord
            for d in range(dim):
                # Map to -3, 0, 3
                val = (c % 3) - 1
                x[d] = val * 3.0
                c //= 3
            grid_points.append(x)

        # Also add some random points for diversity
        rng = np.random.default_rng(42)
        for _ in range(20):
            x = rng.uniform(-4, 4, dim)
            grid_points.append(x)

        # Evaluate grid points
        best_x = None
        best_f = np.inf
        for x in grid_points:
            if calls[0] >= budget:
                break
            f = evaluate(x)
            if f is None:
                break
            if f < best_f:
                best_f = f
                best_x = x.copy()

        # Phase 2: Nelder-Mead with adaptive restarts
        # Use remaining budget for local optimization
        remaining = budget - calls[0]

        # Nelder-Mead parameters
        alpha = 1.0   # reflection
        gamma = 2.0   # expansion
        rho = 0.5     # contraction
        sigma = 0.5   # shrink

        def nelder_mead(x0, f0, max_iter):
            """Nelder-Mead simplex method."""
            n = dim
            # Initialize simplex
            simplex = [x0.copy()]
            fvals = [f0]

            # Create other simplex points
            for i in range(n):
                xi = x0.copy()
                xi[i] += 0.1 if xi[i] < 4.9 else -0.1
                xi = np.clip(xi, -5, 5)
                fi = evaluate(xi)
                if fi is None:
                    break
                simplex.append(xi)
                fvals.append(fi)

            if len(simplex) < n + 1:
                return x0, f0

            for _ in range(max_iter):
                if calls[0] >= budget:
                    break
                if time.time() > deadline:
                    break

                # Sort by function value
                order = np.argsort(fvals)
                simplex = [simplex[i] for i in order]
                fvals = [fvals[i] for i in order]

                # Centroid of all but worst
                centroid = np.mean(simplex[:-1], axis=0)

                # Reflection
                x_r = centroid + alpha * (centroid - simplex[-1])
                x_r = np.clip(x_r, -5, 5)
                f_r = evaluate(x_r)
                if f_r is None:
                    break

                if f_r < fvals[0]:
                    # Expansion
                    x_e = centroid + gamma * (x_r - centroid)
                    x_e = np.clip(x_e, -5, 5)
                    f_e = evaluate(x_e)
                    if f_e is None:
                        break
                    if f_e < f_r:
                        simplex[-1] = x_e
                        fvals[-1] = f_e
                    else:
                        simplex[-1] = x_r
                        fvals[-1] = f_r
                elif f_r < fvals[-2]:
                    # Accept reflection
                    simplex[-1] = x_r
                    fvals[-1] = f_r
                else:
                    # Contraction
                    if f_r < fvals[-1]:
                        # Outside contraction
                        x_c = centroid + rho * (x_r - centroid)
                        x_c = np.clip(x_c, -5, 5)
                        f_c = evaluate(x_c)
                        if f_c is None:
                            break
                        if f_c <= f_r:
                            simplex[-1] = x_c
                            fvals[-1] = f_c
                        else:
                            # Shrink
                            for i in range(1, len(simplex)):
                                simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                                simplex[i] = np.clip(simplex[i], -5, 5)
                                f_i = evaluate(simplex[i])
                                if f_i is None:
                                    return simplex[0], fvals[0]
                                fvals[i] = f_i
                    else:
                        # Inside contraction
                        x_c = centroid - rho * (centroid - simplex[-1])
                        x_c = np.clip(x_c, -5, 5)
                        f_c = evaluate(x_c)
                        if f_c is None:
                            break
                        if f_c < fvals[-1]:
                            simplex[-1] = x_c
                            fvals[-1] = f_c
                        else:
                            # Shrink
                            for i in range(1, len(simplex)):
                                simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                                simplex[i] = np.clip(simplex[i], -5, 5)
                                f_i = evaluate(simplex[i])
                                if f_i is None:
                                    return simplex[0], fvals[0]
                                fvals[i] = f_i

                # Update best
                min_idx = np.argmin(fvals)
                if fvals[min_idx] < best_f:
                    best_f = fvals[min_idx]
                    best_x = simplex[min_idx].copy()

            # Return best from simplex
            min_idx = np.argmin(fvals)
            return simplex[min_idx], fvals[min_idx]

        # Run Nelder-Mead with adaptive restarts
        # Split remaining budget into chunks for restarts
        num_restarts = 5
        iter_per_restart = max(10, remaining // (num_restarts * (dim + 1)))

        for restart in range(num_restarts):
            if calls[0] >= budget:
                break
            if time.time() > deadline:
                break

            # Perturb current best for restart
            if restart == 0:
                x_start = best_x.copy()
                f_start = best_f
            else:
                # Perturb with decreasing step size
                step = 1.0 / (restart + 1)
                x_start = best_x + step * rng.standard_normal(dim)
                x_start = np.clip(x_start, -5, 5)
                f_start = evaluate(x_start)
                if f_start is None:
                    break
                if f_start < best_f:
                    best_f = f_start
                    best_x = x_start.copy()

            x_opt, f_opt = nelder_mead(x_start, f_start, iter_per_restart)

            if f_opt < best_f:
                best_f = f_opt
                best_x = x_opt.copy()

        return best_x, best_f
# EVOLVE-BLOCK-END
