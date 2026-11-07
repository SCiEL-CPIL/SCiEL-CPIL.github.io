"""Nelder-Mead simplex with multi-restart global-to-local strategy for BBOB f20 (Schwefel)."""
import numpy as np
import time


class Optimizer:
    """Multi-restart Nelder-Mead simplex optimizer."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        deadline = time.time() + 280  # safety margin before 300s limit

        best_x = None
        best_f = np.inf
        calls_used = 0

        # Define a wrapper to count calls and check deadline
        def safe_func(x):
            nonlocal calls_used
            if calls_used >= budget:
                return np.inf
            if time.time() >= deadline:
                return np.inf
            calls_used += 1
            val = func(x)
            return val

        # Nelder-Mead simplex method
        def nelder_mead(func_eval, x0, initial_scale, max_iter):
            dim = len(x0)
            # Initialize simplex
            simplex = [x0.copy()]
            fvals = [func_eval(x0)]

            for i in range(dim):
                xi = x0.copy()
                xi[i] += initial_scale
                simplex.append(xi)
                fvals.append(func_eval(xi))

            if calls_used >= budget or time.time() >= deadline:
                return np.array(simplex[np.argmin(fvals)]), min(fvals)

            # Sort simplex by function value
            idx = np.argsort(fvals)
            simplex = [simplex[i] for i in idx]
            fvals = [fvals[i] for i in idx]

            alpha = 1.0   # reflection
            gamma = 2.0   # expansion
            rho = 0.5     # contraction
            sigma = 0.5   # shrink

            for _ in range(max_iter):
                if calls_used >= budget or time.time() >= deadline:
                    break

                # Sort
                idx = np.argsort(fvals)
                simplex = [simplex[i] for i in idx]
                fvals = [fvals[i] for i in idx]

                # Centroid of all but worst
                centroid = np.mean(simplex[:-1], axis=0)

                # Reflection
                xr = centroid + alpha * (centroid - simplex[-1])
                fr = func_eval(xr)

                if fr < fvals[0]:
                    # Expansion
                    xe = centroid + gamma * (centroid - simplex[-1])
                    fe = func_eval(xe)
                    if fe < fr:
                        simplex[-1] = xe
                        fvals[-1] = fe
                    else:
                        simplex[-1] = xr
                        fvals[-1] = fr
                elif fr < fvals[-2]:
                    # Accept reflection
                    simplex[-1] = xr
                    fvals[-1] = fr
                else:
                    # Contraction
                    if fr < fvals[-1]:
                        # Outside contraction
                        xc = centroid + rho * (simplex[-1] - centroid)
                        fc = func_eval(xc)
                        if fc <= fr:
                            simplex[-1] = xc
                            fvals[-1] = fc
                        else:
                            # Shrink
                            for i in range(1, dim + 1):
                                simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                                fvals[i] = func_eval(simplex[i])
                    else:
                        # Inside contraction
                        xc = centroid - rho * (centroid - simplex[-1])
                        fc = func_eval(xc)
                        if fc < fvals[-1]:
                            simplex[-1] = xc
                            fvals[-1] = fc
                        else:
                            # Shrink
                            for i in range(1, dim + 1):
                                simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                                fvals[i] = func_eval(simplex[i])

                # Check convergence
                if len(fvals) >= 2:
                    spread = max(fvals) - min(fvals)
                    if spread < 1e-12:
                        break

            # Return best point
            idx_min = np.argmin(fvals)
            return np.array(simplex[idx_min]), fvals[idx_min]

        # Multi-restart strategy: start with different initial points and scales
        rng = np.random.default_rng(42)

        # Generate diverse starting points
        # Mix of center, random, and corner-ish points
        starts = []
        # Center
        starts.append(np.zeros(dim))
        # Random points
        for _ in range(5):
            starts.append(rng.uniform(-4, 4, dim))
        # Points near corners of the box
        for _ in range(3):
            starts.append(rng.choice([-4, 0, 4], size=dim).astype(float))

        # Allocate budget: more iterations for earlier starts (they're more likely to find the basin)
        # But also give later starts a chance
        # Total budget is 500 calls. Each Nelder-Mead call uses (dim+1) + iterations
        # Let's do 3 restarts with decreasing budget allocation

        # Phase 1: Coarse search with large scale
        # Use first ~40% of budget for initial exploration
        phase1_budget = int(budget * 0.4)
        phase2_budget = int(budget * 0.35)
        phase3_budget = budget - phase1_budget - phase2_budget

        # Restart 1: center, large scale
        if calls_used < budget and time.time() < deadline:
            x_best, f_best = nelder_mead(safe_func, starts[0], 3.0, phase1_budget)
            if f_best < best_f:
                best_f = f_best
                best_x = x_best.copy()

        # Restart 2: best found so far or second start, medium scale
        if calls_used < budget and time.time() < deadline:
            start2 = best_x if best_x is not None else starts[1]
            x_best2, f_best2 = nelder_mead(safe_func, start2, 1.5, phase2_budget)
            if f_best2 < best_f:
                best_f = f_best2
                best_x = x_best2.copy()

        # Restart 3: fine-tune from best, small scale
        if calls_used < budget and time.time() < deadline:
            start3 = best_x if best_x is not None else starts[2]
            x_best3, f_best3 = nelder_mead(safe_func, start3, 0.5, phase3_budget)
            if f_best3 < best_f:
                best_f = f_best3
                best_x = x_best3.copy()

        # If we still have budget, do another fine-tune
        if calls_used < budget and time.time() < deadline and best_x is not None:
            remaining = budget - calls_used
            if remaining > 10:
                x_best4, f_best4 = nelder_mead(safe_func, best_x, 0.2, remaining)
                if f_best4 < best_f:
                    best_f = f_best4
                    best_x = x_best4.copy()

        return best_x, best_f
