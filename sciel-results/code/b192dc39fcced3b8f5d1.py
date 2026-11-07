"""Nelder-Mead simplex with adaptive reset and multi-start diversification for f24 (Lunacek bi-Rastrigin)."""
import numpy as np
import time


class Optimizer:
    """
    Derivative-free trust-region optimizer using Nelder-Mead simplex method
    with adaptive reset and multi-start diversification. Models the landscape
    as a polytope (simplex) rather than a Gaussian, better suited for the
    non-smooth Rastrigin component and skewed Lunacek basin.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget

        start_time = time.time()
        deadline = start_time + 285.0  # Safety margin before 300s limit

        best_x = None
        best_f = np.inf
        calls_made = [0]

        def counted_func(x):
            calls_made[0] += 1
            if time.time() > deadline:
                raise TimeoutError("Deadline exceeded")
            return float(func(x))

        rng = np.random.default_rng(12345)

        # Phase 1: Multi-start exploration to identify promising basins
        # Lunacek has two basins; we need to sample broadly
        n_explore = min(30, budget // 15)
        candidates = []
        for i in range(n_explore):
            if calls_made[0] >= budget or time.time() > deadline:
                break
            x = rng.uniform(-4.8, 4.8, dim)
            fx = counted_func(x)
            candidates.append((fx, x.copy()))
            if fx < best_f:
                best_f = fx
                best_x = x.copy()

        # Sort by fitness
        candidates.sort(key=lambda t: t[0])

        # Phase 2: Nelder-Mead from multiple starting points
        # Allocate remaining budget across starts
        remaining = budget - calls_made[0]
        n_starts = min(5, len(candidates))
        budget_per_start = max(50, remaining // n_starts)

        for start_idx in range(n_starts):
            if calls_made[0] >= budget or time.time() > deadline:
                break

            x0 = candidates[start_idx][1].copy()
            f0 = candidates[start_idx][0]

            # Build initial simplex around x0
            # Use a spread that adapts to the problem scale
            spread = 1.5
            simplex = [x0.copy()]
            for d in range(dim):
                v = x0.copy()
                v[d] += spread
                simplex.append(v)

            # Evaluate simplex vertices
            simplex_f = [f0]
            for i in range(1, dim + 1):
                if calls_made[0] >= budget or time.time() > deadline:
                    break
                fx = counted_func(simplex[i])
                simplex_f.append(fx)
                if fx < best_f:
                    best_f = fx
                    best_x = simplex[i].copy()

            # Nelder-Mead iterations
            n_iter_limit = budget_per_start
            iter_count = 0
            stagnation = 0
            max_stagnation = 40

            while iter_count < n_iter_limit and calls_made[0] < budget and time.time() < deadline:
                iter_count += 1

                # Sort simplex by function value
                order = np.argsort(simplex_f)
                simplex = [simplex[i].copy() for i in order]
                simplex_f = [simplex_f[i] for i in order]

                # Centroid of all but worst vertex
                centroid = np.mean(simplex[:-1], axis=0)
                worst = simplex[-1]
                worst_f = simplex_f[-1]

                # Reflection
                alpha = 1.0
                x_reflect = centroid + alpha * (centroid - worst)
                x_reflect = np.clip(x_reflect, -5, 5)

                if calls_made[0] >= budget or time.time() > deadline:
                    break
                f_reflect = counted_func(x_reflect)
                if f_reflect < best_f:
                    best_f = f_reflect
                    best_x = x_reflect.copy()

                if f_reflect < simplex_f[0]:
                    # Expansion
                    gamma = 2.0
                    x_expand = centroid + gamma * (centroid - worst)
                    x_expand = np.clip(x_expand, -5, 5)
                    if calls_made[0] >= budget or time.time() > deadline:
                        break
                    f_expand = counted_func(x_expand)
                    if f_expand < best_f:
                        best_f = f_expand
                        best_x = x_expand.copy()
                    if f_expand < f_reflect:
                        simplex[-1] = x_expand
                        simplex_f[-1] = f_expand
                    else:
                        simplex[-1] = x_reflect
                        simplex_f[-1] = f_reflect
                    stagnation = 0
                elif f_reflect < simplex_f[-2]:
                    # Accept reflection
                    simplex[-1] = x_reflect
                    simplex_f[-1] = f_reflect
                    stagnation = 0
                else:
                    # Contraction
                    rho = 0.5
                    x_contract = centroid + rho * (worst - centroid)
                    x_contract = np.clip(x_contract, -5, 5)
                    if calls_made[0] >= budget or time.time() > deadline:
                        break
                    f_contract = counted_func(x_contract)
                    if f_contract < best_f:
                        best_f = f_contract
                        best_x = x_contract.copy()

                    if f_contract < worst_f:
                        simplex[-1] = x_contract
                        simplex_f[-1] = f_contract
                        stagnation = 0
                    else:
                        # Shrink
                        for i in range(1, dim + 1):
                            simplex[i] = simplex[0] + 0.5 * (simplex[i] - simplex[0])
                            simplex[i] = np.clip(simplex[i], -5, 5)
                            if calls_made[0] >= budget or time.time() > deadline:
                                break
                            simplex_f[i] = counted_func(simplex[i])
                            if simplex_f[i] < best_f:
                                best_f = simplex_f[i]
                                best_x = simplex[i].copy()
                        stagnation += 1

                # Check for convergence or stagnation
                f_range = max(simplex_f) - min(simplex_f)
                if f_range < 1e-10:
                    break

                if stagnation >= max_stagnation:
                    # Reset: jitter the simplex around current best
                    for i in range(1, dim + 1):
                        simplex[i] = best_x + rng.uniform(-0.5, 0.5, dim)
                        simplex[i] = np.clip(simplex[i], -5, 5)
                        if calls_made[0] >= budget or time.time() > deadline:
                            break
                        simplex_f[i] = counted_func(simplex[i])
                        if simplex_f[i] < best_f:
                            best_f = simplex_f[i]
                            best_x = simplex[i].copy()
                    simplex[0] = best_x.copy()
                    simplex_f[0] = best_f
                    stagnation = 0

        # Phase 3: Final local polish with coordinate descent
        if calls_made[0] < budget and time.time() < deadline and best_x is not None:
            x = best_x.copy()
            fx = best_f
            step = 0.3
            for _ in range(30):
                if calls_made[0] >= budget or time.time() > deadline:
                    break
                improved = False
                for d in range(dim):
                    if calls_made[0] >= budget or time.time() > deadline:
                        break
                    for sign in [1, -1]:
                        if calls_made[0] >= budget or time.time() > deadline:
                            break
                        x_trial = x.copy()
                        x_trial[d] += sign * step
                        x_trial = np.clip(x_trial, -5, 5)
                        fx_trial = counted_func(x_trial)
                        if fx_trial < fx:
                            x = x_trial
                            fx = fx_trial
                            improved = True
                            if fx < best_f:
                                best_f = fx
                                best_x = x.copy()
                            break
                if not improved:
                    step *= 0.7
                    if step < 0.01:
                        break
                else:
                    step = max(0.05, step * 0.9)

        if best_x is None:
            best_x = np.zeros(dim)
            if calls_made[0] < budget:
                best_f = counted_func(best_x)

        return best_x, best_f
