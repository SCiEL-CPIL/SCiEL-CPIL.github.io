"""Rewritten optimizer: (1+1)-ES with diverse restarts and final local refinement."""
# EVOLVE-BLOCK-START
import time
import numpy as np


class Optimizer:
    """(1+1)-ES with multiple restarts from diverse initial points + final Nelder-Mead polish."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        t0 = time.time()
        deadline = t0 + 290.0  # safety margin under 300s
        dim = self.dim
        lo, hi = -5.0, 5.0
        rng = np.random.default_rng(42)

        calls = [0]
        best_x = None
        best_f = np.inf

        def counted(x):
            nonlocal best_f, best_x
            x = np.clip(np.asarray(x, dtype=float), lo, hi)
            calls[0] += 1
            f = func(x)
            if f < best_f:
                best_f = f
                best_x = x.copy()
            return f

        def es_run(x0, budget_remaining, sigma0=1.5):
            """Run (1+1)-ES from x0 for budget_remaining evaluations."""
            nonlocal calls, best_f, best_x
            x = np.array(x0, dtype=float)
            f = counted(x)
            sigma = sigma0
            mu = 1.0 / (dim + 1)
            sigma_min = 1e-4
            sigma_max = 10.0

            while calls[0] < budget_remaining and time.time() < deadline:
                # Generate mutant
                direction = rng.standard_normal(dim)
                y = x + sigma * direction
                y = np.clip(y, lo, hi)
                fy = counted(y)

                if fy <= f:
                    x = y
                    f = fy
                    # Increase sigma on success
                    sigma *= (1 + mu)
                    sigma = min(sigma, sigma_max)
                else:
                    # Decrease sigma on failure
                    sigma *= (1 - mu)
                    sigma = max(sigma, sigma_min)

                # Check for convergence
                if sigma < sigma_min * 2:
                    break

            return x, f

        def nelder_mead(x0, budget_remaining):
            """Nelder-Mead local refinement from x0 for budget_remaining evaluations."""
            nonlocal calls, best_f, best_x
            x = np.array(x0, dtype=float)
            f0 = counted(x)

            # Build simplex
            simplex = np.empty((dim + 1, dim))
            simplex[0] = x
            fvals = np.empty(dim + 1)
            fvals[0] = f0

            for i in range(dim):
                p = x.copy()
                p[i] += 0.1
                p = np.clip(p, lo, hi)
                simplex[i + 1] = p
                fvals[i + 1] = counted(p)

            alpha, gamma, rho = 1.0, 2.0, 0.5

            while calls[0] < budget_remaining and time.time() < deadline:
                # Sort simplex by fvals
                order = np.argsort(fvals)
                simplex = simplex[order]
                fvals = fvals[order]

                if fvals[0] < best_f:
                    best_f = fvals[0]
                    best_x = simplex[0].copy()

                x_best = simplex[0]
                x_worst = simplex[-1]
                centroid = simplex[:-1].mean(axis=0)

                # Reflection
                x_r = centroid + alpha * (centroid - x_worst)
                x_r = np.clip(x_r, lo, hi)
                f_r = counted(x_r)

                if f_r < fvals[0]:
                    # Expansion
                    x_e = centroid + gamma * (centroid - x_worst)
                    x_e = np.clip(x_e, lo, hi)
                    f_e = counted(x_e)
                    if f_e < f_r:
                        simplex[-1] = x_e
                        fvals[-1] = f_e
                    else:
                        simplex[-1] = x_r
                        fvals[-1] = f_r
                elif f_r < fvals[-2]:
                    simplex[-1] = x_r
                    fvals[-1] = f_r
                else:
                    # Contraction
                    x_c = centroid + rho * (x_worst - centroid)
                    x_c = np.clip(x_c, lo, hi)
                    f_c = counted(x_c)
                    if f_c < fvals[-1]:
                        simplex[-1] = x_c
                        fvals[-1] = f_c
                    else:
                        # Shrink
                        for i in range(1, dim + 1):
                            simplex[i] = x_best + 0.5 * (simplex[i] - x_best)
                            fvals[i] = counted(simplex[i])

                # Check convergence
                if np.std(fvals) < 1e-10:
                    break

            return simplex[0], fvals[0]

        # Phase 1: Diverse exploration with (1+1)-ES restarts
        # Use 60% of budget for exploration
        explore_budget = int(self.budget * 0.6)

        # Generate diverse starting points
        start_points = []
        # Center
        start_points.append(np.zeros(dim))
        # Random points
        for _ in range(5):
            start_points.append(rng.uniform(lo, hi, dim))

        # Run (1+1)-ES from each starting point, splitting budget
        es_budget_per = max(10, explore_budget // len(start_points))

        for sp in start_points:
            if calls[0] >= explore_budget or time.time() >= deadline:
                break
            es_run(sp, explore_budget, sigma0=2.0)

        # Phase 2: Local refinement with Nelder-Mead from best point
        if calls[0] < self.budget and time.time() < deadline:
            refine_budget = self.budget - calls[0]
            if best_x is not None:
                nelder_mead(best_x, refine_budget)

        return best_x, best_f
# EVOLVE-BLOCK-END
