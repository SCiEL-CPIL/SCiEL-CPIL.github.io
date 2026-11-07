"""Black-box continuous optimizer for BBOB-style f1 (Sphere)."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """
    Hybrid optimizer: (1+1)-ES with success rule, then CMA-ES style
    with restarts, plus a final Nelder-Mead polish.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290  # 10s safety margin

        rng = np.random.default_rng(0)
        dim = self.dim
        budget = self.budget

        # Phase 1: (1+1)-ES with 1/5 success rule
        x = rng.uniform(-4, 4, dim)
        fx = func(x)
        sigma = 2.0
        best_x, best_fx = x.copy(), fx

        # Track for success rule
        steps_since_improve = 0
        total_steps = 0
        successes = 0

        calls_used = 1

        for _ in range(budget - 1):
            if time.time() > deadline - 5:
                break

            y = np.clip(x + sigma * rng.standard_normal(dim), -5, 5)
            fy = func(y)
            calls_used += 1
            total_steps += 1

            if fy < fx:
                x, fx = y, fy
                successes += 1
                steps_since_improve = 0
                sigma *= 1.5
                if fx < best_fx:
                    best_x, best_fx = x.copy(), fx
            else:
                steps_since_improve += 1
                sigma *= 0.87

            # Check success rate periodically
            if total_steps >= 10 and steps_since_improve > 5:
                # Stagnation detected, restart with smaller sigma
                sigma = max(sigma * 0.5, 1e-4)
                steps_since_improve = 0

        # Phase 2: Local refinement with smaller steps around best
        if calls_used < budget and time.time() < deadline - 5:
            x = best_x.copy()
            fx = best_fx
            sigma = 0.1
            for _ in range(min(50, budget - calls_used)):
                if time.time() > deadline - 5:
                    break
                y = np.clip(x + sigma * rng.standard_normal(dim), -5, 5)
                fy = func(y)
                calls_used += 1
                if fy < fx:
                    x, fx = y, fy
                    sigma *= 1.3
                    if fx < best_fx:
                        best_x, best_fx = x.copy(), fx
                else:
                    sigma *= 0.85

        # Phase 3: Nelder-Mead polish if budget remains
        if calls_used < budget - 20 and time.time() < deadline - 10:
            x = best_x.copy()
            fx = best_fx

            # Initialize simplex
            simplex = [x.copy()]
            for i in range(dim):
                xi = x.copy()
                step = 0.01 * (1 if x[i] != 0 else 1)
                xi[i] += step
                xi = np.clip(xi, -5, 5)
                simplex.append(xi)

            fvals = [func(simplex[i]) for i in range(dim + 1)]
            calls_used += dim + 1

            # Nelder-Mead iterations
            max_nm_iter = min(200, budget - calls_used)
            for _ in range(max_nm_iter):
                if time.time() > deadline - 5:
                    break
                if calls_used >= budget:
                    break

                # Sort simplex
                idx = np.argsort(fvals)
                simplex = [simplex[i] for i in idx]
                fvals = [fvals[i] for i in idx]

                # Centroid of all but worst
                centroid = np.mean(simplex[:-1], axis=0)

                # Reflection
                xr = np.clip(centroid + 1.0 * (centroid - simplex[-1]), -5, 5)
                fr = func(xr)
                calls_used += 1

                if fr < fvals[0]:
                    # Expansion
                    xe = np.clip(centroid + 2.0 * (xr - centroid), -5, 5)
                    fe = func(xe)
                    calls_used += 1
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
                    xc = np.clip(centroid + 0.5 * (simplex[-1] - centroid), -5, 5)
                    fc = func(xc)
                    calls_used += 1
                    if fc < fvals[-1]:
                        simplex[-1] = xc
                        fvals[-1] = fc
                    else:
                        # Shrink
                        for i in range(1, dim + 1):
                            simplex[i] = np.clip(simplex[0] + 0.5 * (simplex[i] - simplex[0]), -5, 5)
                            fvals[i] = func(simplex[i])
                            calls_used += 1
                            if calls_used >= budget:
                                break
                        if calls_used >= budget:
                            break

                # Update best
                min_idx = np.argmin(fvals)
                if fvals[min_idx] < best_fx:
                    best_x = simplex[min_idx].copy()
                    best_fx = fvals[min_idx]

        return best_x, best_fx
# EVOLVE-BLOCK-END
