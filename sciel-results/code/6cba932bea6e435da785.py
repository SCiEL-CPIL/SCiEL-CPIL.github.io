"""Rewritten optimizer: Nelder-Mead with adaptive valley tracking for Rosenbrock."""
import numpy as np
import time


class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        deadline = time.time() + 280  # safety margin before 300s kill

        # Initial point: try a few random starts, pick the best
        rng = np.random.default_rng(42)
        best_x = None
        best_f = np.inf
        for _ in range(3):
            x0 = rng.uniform(-4, 4, dim)
            f0 = func(x0)
            if f0 < best_f:
                best_f = f0
                best_x = x0.copy()

        x = best_x
        fx = best_f
        calls = 3

        # Nelder-Mead simplex
        # Initialize simplex around x
        simplex = [x.copy() for _ in range(dim + 1)]
        fvals = [fx]
        # First vertex is x, others are perturbed
        step = 0.5
        for i in range(1, dim + 1):
            simplex[i] = x.copy()
            simplex[i][i] += step
            simplex[i] = np.clip(simplex[i], -5, 5)
            fvals.append(func(simplex[i]))
            calls += 1
            if fvals[-1] < fx:
                fx = fvals[-1]
                x = simplex[i].copy()

        # Nelder-Mead parameters
        alpha = 1.0   # reflection
        gamma = 2.0   # expansion
        rho = 0.5     # contraction
        sigma = 0.5   # shrink

        # Adaptive step size for valley tracking
        valley_step = 0.5

        while calls < budget:
            if time.time() > deadline:
                break

            # Sort simplex by function value
            order = np.argsort(fvals)
            simplex = [simplex[i].copy() for i in order]
            fvals = [fvals[i] for i in order]

            # Check convergence
            if abs(fvals[-1] - fvals[0]) < 1e-10:
                break

            # Centroid of all but worst
            centroid = np.mean(simplex[:-1], axis=0)

            # Reflection
            xr = centroid + alpha * (centroid - simplex[-1])
            xr = np.clip(xr, -5, 5)
            fr = func(xr)
            calls += 1
            if fr < fvals[0]:
                # Expansion
                xe = centroid + gamma * (centroid - simplex[-1])
                xe = np.clip(xe, -5, 5)
                fe = func(xe)
                calls += 1
                if fe < fr:
                    simplex[-1] = xe
                    fvals[-1] = fe
                else:
                    simplex[-1] = xr
                    fvals[-1] = fr
            elif fr < fvals[-2]:
                simplex[-1] = xr
                fvals[-1] = fr
            else:
                # Contraction
                if fr < fvals[-1]:
                    # Outside contraction
                    xc = centroid + rho * (xr - centroid)
                    xc = np.clip(xc, -5, 5)
                    fc = func(xc)
                    calls += 1
                    if fc <= fr:
                        simplex[-1] = xc
                        fvals[-1] = fc
                    else:
                        # Shrink
                        for i in range(1, dim + 1):
                            simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                            simplex[i] = np.clip(simplex[i], -5, 5)
                            fvals[i] = func(simplex[i])
                            calls += 1
                else:
                    # Inside contraction
                    xc = centroid - rho * (simplex[-1] - centroid)
                    xc = np.clip(xc, -5, 5)
                    fc = func(xc)
                    calls += 1
                    if fc < fvals[-1]:
                        simplex[-1] = xc
                        fvals[-1] = fc
                    else:
                        # Shrink
                        for i in range(1, dim + 1):
                            simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                            simplex[i] = np.clip(simplex[i], -5, 5)
                            fvals[i] = func(simplex[i])
                            calls += 1

            # Track best
            min_idx = np.argmin(fvals)
            if fvals[min_idx] < fx:
                fx = fvals[min_idx]
                x = simplex[min_idx].copy()

            # Adaptive valley tracking: if we're in the valley (high f but decreasing),
            # do a line search along the valley direction
            # For Rosenbrock, the valley is roughly where x_i ≈ x_{i-1}^2
            # Check if we're close to the valley
            in_valley = False
            for i in range(1, dim):
                if abs(x[i] - x[i-1]**2) < 1.0:
                    in_valley = True
                    break

            if in_valley and calls < budget - 10:
                # Move along the valley direction
                # The valley direction is approximately (1, 2*x[0], 2*x[1], ..., 2*x[dim-2])
                direction = np.ones(dim)
                for i in range(1, dim):
                    direction[i] = 2 * x[i-1]
                direction_norm = np.linalg.norm(direction)
                if direction_norm > 1e-8:
                    direction /= direction_norm
                    # Try a step in the valley direction
                    for step_size in [valley_step, valley_step / 2, valley_step / 4]:
                        if calls >= budget:
                            break
                        x_trial = x + step_size * direction
                        x_trial = np.clip(x_trial, -5, 5)
                        f_trial = func(x_trial)
                        calls += 1
                        if f_trial < fx:
                            x = x_trial
                            fx = f_trial
                            break
                    # Adjust valley step based on success
                    if f_trial < fx + 1e-6:
                        valley_step = min(valley_step * 1.2, 2.0)
                    else:
                        valley_step = max(valley_step * 0.8, 0.1)

        return x, fx
