"""Rewritten optimizer for BBOB f23 (Katsuura) using Nelder-Mead with restarts."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Nelder-Mead with restarts for Katsuura optimization."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        d = self.dim
        B = self.budget
        deadline = time.time() + 280  # safety margin before 300s limit
        rng = np.random.default_rng(12345)

        best_x = None
        best_f = np.inf
        calls_made = 0

        def evaluate(x):
            nonlocal calls_made
            x = np.clip(x, -5, 5)
            f = func(x)
            calls_made += 1
            return x, f

        # Initial exploration: sample a few random points
        n_init = min(10, B // 5)
        init_points = []
        for i in range(n_init):
            x = rng.uniform(-4, 4, d)
            x, f = evaluate(x)
            init_points.append((f, x))
            if f < best_f:
                best_f = f
                best_x = x.copy()

        # Sort by function value
        init_points.sort(key=lambda t: t[0])

        # Nelder-Mead parameters
        alpha = 1.0  # reflection
        gamma = 2.0  # expansion
        rho = 0.5    # contraction
        sigma = 0.5  # shrink

        # Try Nelder-Mead from the best initial points
        for start_idx in range(min(3, len(init_points))):
            if calls_made >= B or time.time() > deadline:
                break

            # Build initial simplex
            f0, x0 = init_points[start_idx]
            simplex = [x0.copy()]
            for j in range(d):
                dx = x0.copy()
                step = 0.1 if dx[j] >= 0 else -0.1
                dx[j] += step
                simplex.append(dx)

            # Evaluate simplex
            simplex_f = [f0]
            for j in range(1, d + 1):
                _, fj = evaluate(simplex[j])
                simplex_f.append(fj)

            # Track best
            for j in range(d + 1):
                if simplex_f[j] < best_f:
                    best_f = simplex_f[j]
                    best_x = simplex[j].copy()

            # Nelder-Mead iterations
            max_iter = B - calls_made
            for it in range(max_iter):
                if calls_made >= B or time.time() > deadline:
                    break

                # Sort simplex by function value
                order = np.argsort(simplex_f)
                simplex = [simplex[i].copy() for i in order]
                simplex_f = [simplex_f[i] for i in order]

                # Centroid of all but worst
                centroid = np.mean(simplex[:-1], axis=0)

                # Reflection
                xr = centroid + alpha * (centroid - simplex[-1])
                xr = np.clip(xr, -5, 5)
                _, fr = evaluate(xr)

                if fr < simplex_f[0]:
                    # Expansion
                    xe = centroid + gamma * (centroid - simplex[-1])
                    xe = np.clip(xe, -5, 5)
                    _, fe = evaluate(xe)
                    if fe < fr:
                        simplex[-1] = xe
                        simplex_f[-1] = fe
                    else:
                        simplex[-1] = xr
                        simplex_f[-1] = fr
                elif fr < simplex_f[-2]:
                    # Accept reflection
                    simplex[-1] = xr
                    simplex_f[-1] = fr
                else:
                    # Contraction
                    if fr < simplex_f[-1]:
                        # Outside contraction
                        xc = centroid + rho * (xr - centroid)
                        xc = np.clip(xc, -5, 5)
                        _, fc = evaluate(xc)
                        if fc <= fr:
                            simplex[-1] = xc
                            simplex_f[-1] = fc
                        else:
                            # Shrink
                            for j in range(1, d + 1):
                                simplex[j] = simplex[0] + sigma * (simplex[j] - simplex[0])
                                simplex[j] = np.clip(simplex[j], -5, 5)
                                simplex_f[j] = evaluate(simplex[j])[1]
                    else:
                        # Inside contraction
                        xc = centroid - rho * (centroid - simplex[-1])
                        xc = np.clip(xc, -5, 5)
                        _, fc = evaluate(xc)
                        if fc <= simplex_f[-1]:
                            simplex[-1] = xc
                            simplex_f[-1] = fc
                        else:
                            # Shrink
                            for j in range(1, d + 1):
                                simplex[j] = simplex[0] + sigma * (simplex[j] - simplex[0])
                                simplex[j] = np.clip(simplex[j], -5, 5)
                                simplex_f[j] = evaluate(simplex[j])[1]

                # Update best
                for j in range(d + 1):
                    if simplex_f[j] < best_f:
                        best_f = simplex_f[j]
                        best_x = simplex[j].copy()

                # Convergence check
                spread = np.max(simplex_f) - np.min(simplex_f)
                if spread < 1e-10:
                    break

        # Final local refinement with small steps around best_x
        if best_x is not None and calls_made < B:
            for _ in range(min(50, B - calls_made)):
                if time.time() > deadline:
                    break
                step_size = 0.01
                dx = rng.uniform(-step_size, step_size, d)
                x_new = np.clip(best_x + dx, -5, 5)
                _, f_new = evaluate(x_new)
                if f_new < best_f:
                    best_f = f_new
                    best_x = x_new.copy()

        return best_x, best_f
# EVOLVE-BLOCK-END
