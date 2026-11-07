"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Hybrid (1+1)-ES with Nelder-Mead refinement for sharp ridge f13."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        t0 = time.time()
        deadline = t0 + 290  # safety margin before 300s limit

        rng = np.random.default_rng(42)
        dim = self.dim
        budget = self.budget

        # Phase 1: Rapid (1+1)-ES for early AOCC gain
        x = rng.uniform(-5, 5, dim)
        fx = func(x)
        best_x, best_f = x.copy(), fx
        calls = 1
        sigma = 1.5

        # Use most of the budget for ES phase
        es_budget = int(budget * 0.6)
        for i in range(es_budget - 1):
            if time.time() > deadline:
                break
            y = np.clip(x + sigma * rng.standard_normal(dim), -5, 5)
            fy = func(y)
            calls += 1
            if fy < fx:
                x, fx = y, fy
                sigma *= 1.2
            else:
                sigma *= 0.85
            if fy < best_f:
                best_x, best_f = y.copy(), fy

        # Phase 2: Nelder-Mead refinement using remaining budget
        remaining = budget - calls
        if remaining > 10 and time.time() < deadline:
            # Initialize simplex around best point
            simplex = [best_x.copy()]
            step = 0.5
            for d in range(dim):
                v = best_x.copy()
                v[d] += step
                simplex.append(np.clip(v, -5, 5))

            simplexf = [func(simplex[i]) for i in range(dim + 1)]
            calls += dim + 1

            # Nelder-Mead iterations
            max_nm_iter = min(remaining - (dim + 1), 200)
            for _ in range(max_nm_iter):
                if time.time() > deadline:
                    break
                # Sort by function value
                order = np.argsort(simplexf)
                simplex = [simplex[i] for i in order]
                simplexf = [simplexf[i] for i in order]

                # Centroid of all but worst
                centroid = np.mean(simplex[:-1], axis=0)

                # Reflection
                xr = centroid + 1.0 * (centroid - simplex[-1])
                xr = np.clip(xr, -5, 5)
                fr = func(xr)
                calls += 1

                if fr < simplexf[0]:
                    # Expansion
                    xe = centroid + 2.0 * (centroid - simplex[-1])
                    xe = np.clip(xe, -5, 5)
                    fe = func(xe)
                    calls += 1
                    if fe < fr:
                        simplex[-1], simplexf[-1] = xe, fe
                    else:
                        simplex[-1], simplexf[-1] = xr, fr
                elif fr < simplexf[-2]:
                    simplex[-1], simplexf[-1] = xr, fr
                else:
                    # Contraction
                    if fr < simplexf[-1]:
                        xc = centroid + 0.5 * (simplex[-1] - centroid)
                        fc = func(xc)
                        calls += 1
                        if fc <= fr:
                            simplex[-1], simplexf[-1] = xc, fc
                        else:
                            # Shrink
                            for i in range(1, dim + 1):
                                simplex[i] = simplex[0] + 0.5 * (simplex[i] - simplex[0])
                                simplex[i] = np.clip(simplex[i], -5, 5)
                                simplexf[i] = func(simplex[i])
                                calls += 1
                    else:
                        # Shrink
                        for i in range(1, dim + 1):
                            simplex[i] = simplex[0] + 0.5 * (simplex[i] - simplex[0])
                            simplex[i] = np.clip(simplex[i], -5, 5)
                            simplexf[i] = func(simplex[i])
                            calls += 1

                # Track best
                if simplexf[0] < best_f:
                    best_x, best_f = simplex[0].copy(), simplexf[0]

        return best_x, best_f
# EVOLVE-BLOCK-END
