"""Optimizer for BBOB f17 (Schaffers F7) with diverse exploration and local refinement."""
import numpy as np
import time


class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280  # safety margin before 300s limit
        dim = self.dim
        budget = self.budget

        best_x = None
        best_f = np.inf
        calls = 0

        def eval_point(x):
            nonlocal calls, best_x, best_f
            x = np.clip(x, -5, 5)
            f = func(x)
            calls += 1
            if f < best_f:
                best_f = f
                best_x = x.copy()
            return f

        # Phase 1: Diverse random sampling to find a good basin
        # Use a mix of uniform and structured points
        rng = np.random.default_rng(42)

        # Start with center and some structured points
        initial_points = [
            np.zeros(dim),
            np.array([0.5] * dim),
            np.array([-0.5] * dim),
        ]
        # Add random points
        n_random = 30
        random_points = rng.uniform(-5, 5, (n_random, dim))

        all_points = initial_points + [random_points[i] for i in range(n_random)]

        for p in all_points:
            if calls >= budget or time.time() > deadline:
                break
            eval_point(p)

        # Phase 2: CMA-ES style local search from the best point
        # Use a simple (1+1)-ES with adaptive step size
        x = best_x.copy()
        fx = best_f
        sigma = 1.0

        # Adapt sigma based on success rate
        successes = 0
        window = 10

        for i in range(budget - calls):
            if time.time() > deadline:
                break

            y = x + sigma * rng.standard_normal(dim)
            y = np.clip(y, -5, 5)
            fy = func(y)
            calls += 1

            if fy < fx:
                x, fx = y, fy
                successes += 1
                if successes >= window * 0.6:
                    sigma *= 1.2
                    successes = 0
                    window = max(5, window // 2)
            else:
                if successes <= window * 0.4:
                    sigma *= 0.7
                    successes = 0
                    window = max(5, window // 2)

            # If sigma becomes too small, try a restart from a random point
            if sigma < 1e-4 and calls < budget - 50:
                # Restart from a random point near current best
                x = x + rng.uniform(-2, 2, dim)
                x = np.clip(x, -5, 5)
                fx = func(x)
                calls += 1
                if fx < best_f:
                    best_f = fx
                    best_x = x.copy()
                sigma = 0.5

        # Phase 3: Nelder-Mead refinement if we have budget left
        if calls < budget - 20 and best_x is not None:
            x0 = best_x.copy()
            f0 = func(x0)
            calls += 1
            if f0 < best_f:
                best_f = f0
                best_x = x0.copy()

            # Nelder-Mead
            simplex = [x0.copy()]
            for j in range(dim):
                perturbation = x0.copy()
                step = 0.1 if abs(x0[j]) < 1e-8 else 0.1 * abs(x0[j])
                perturbation[j] += step
                simplex.append(np.clip(perturbation, -5, 5))

            simplex_f = [f0]
            for j in range(1, dim + 1):
                fj = func(simplex[j])
                calls += 1
                simplex_f.append(fj)
                if fj < best_f:
                    best_f = fj
                    best_x = simplex[j].copy()

            max_iter = min(100, budget - calls - 5)
            for _ in range(max_iter):
                if time.time() > deadline or calls >= budget - 5:
                    break

                # Sort simplex by function value
                order = np.argsort(simplex_f)
                simplex = [simplex[i] for i in order]
                simplex_f = [simplex_f[i] for i in order]

                # Centroid of all but worst point
                centroid = np.mean(simplex[:-1], axis=0)

                # Reflection
                alpha = 1.0
                xr = centroid + alpha * (centroid - simplex[-1])
                xr = np.clip(xr, -5, 5)
                fr = func(xr)
                calls += 1
                if fr < best_f:
                    best_f = fr
                    best_x = xr.copy()

                if simplex_f[0] <= fr < simplex_f[-2]:
                    simplex[-1] = xr
                    simplex_f[-1] = fr
                elif fr < simplex_f[0]:
                    # Expansion
                    gamma = 2.0
                    xe = centroid + gamma * (centroid - simplex[-1])
                    xe = np.clip(xe, -5, 5)
                    fe = func(xe)
                    calls += 1
                    if fe < best_f:
                        best_f = fe
                        best_x = xe.copy()
                    if fe < fr:
                        simplex[-1] = xe
                        simplex_f[-1] = fe
                    else:
                        simplex[-1] = xr
                        simplex_f[-1] = fr
                else:
                    # Contraction
                    rho = 0.5
                    if fr < simplex_f[-1]:
                        xc = centroid + rho * (simplex[-1] - centroid)
                    else:
                        xc = centroid - rho * (simplex[-1] - centroid)
                    xc = np.clip(xc, -5, 5)
                    fc = func(xc)
                    calls += 1
                    if fc < best_f:
                        best_f = fc
                        best_x = xc.copy()
                    if fc < min(fr, simplex_f[-1]):
                        simplex[-1] = xc
                        simplex_f[-1] = fc
                    else:
                        # Shrink
                        for j in range(1, dim + 1):
                            simplex[j] = (simplex[0] + simplex[j]) / 2
                            simplex_f[j] = func(simplex[j])
                            calls += 1
                            if simplex_f[j] < best_f:
                                best_f = simplex_f[j]
                                best_x = simplex[j].copy()
                            if calls >= budget - 5 or time.time() > deadline:
                                break
                        if calls >= budget - 5 or time.time() > deadline:
                            break

        return best_x, best_f
