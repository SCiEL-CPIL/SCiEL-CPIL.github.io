"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Latin Hypercube Sampling + Nelder-Mead Simplex Optimizer."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0  # Safety margin before 300s limit

        dim = self.dim
        budget = self.budget
        calls_used = 0
        best_x = None
        best_f = np.inf

        def call_func(x):
            nonlocal calls_used
            if calls_used >= budget:
                return np.inf
            calls_used += 1
            return func(x)

        def nelder_mead(x0, max_iter=200):
            nonlocal calls_used, deadline, best_f, best_x
            n = self.dim
            simplex = [x0.copy() for _ in range(n + 1)]
            for i in range(1, n + 1):
                simplex[i] = x0.copy()
                simplex[i][i - 1] += 0.5

            fvals = [call_func(x) for x in simplex]

            for _ in range(max_iter):
                if time.time() > deadline:
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
                xr = centroid + 1.0 * (centroid - simplex[-1])
                fr = call_func(xr)
                if fr < fvals[0]:
                    if fr > fvals[-2]:
                        # Expansion
                        xe = centroid + 2.0 * (centroid - simplex[-1])
                        fe = call_func(xe)
                        if fe < fr:
                            simplex[-1] = xe
                            fvals[-1] = fe
                        else:
                            simplex[-1] = xr
                            fvals[-1] = fr
                    else:
                        # Contraction
                        xc = centroid + 0.5 * (simplex[-1] - centroid)
                        fc = call_func(xc)
                        if fc < fvals[-1]:
                            simplex[-1] = xc
                            fvals[-1] = fc
                        else:
                            # Shrink
                            for i in range(1, n + 1):
                                simplex[i] = simplex[0] + 0.5 * (simplex[i] - simplex[0])
                                fvals[i] = call_func(simplex[i])
                        continue

                elif fr < fvals[-1]:
                    simplex[-1] = xr
                    fvals[-1] = fr
                else:
                    # Contraction
                    xc = centroid + 0.5 * (simplex[-1] - centroid)
                    fc = call_func(xc)
                    if fc < fvals[-1]:
                        simplex[-1] = xc
                        fvals[-1] = fc
                    else:
                        # Shrink
                        for i in range(1, n + 1):
                            simplex[i] = simplex[0] + 0.5 * (simplex[i] - simplex[0])
                            fvals[i] = call_func(simplex[i])
                        continue

                # Update global best
                min_idx = np.argmin(fvals)
                if fvals[min_idx] < best_f:
                    best_f = fvals[min_idx]
                    best_x = simplex[min_idx].copy()

                # Early stop if simplex is very small
                if np.max(fvals) - np.min(fvals) < 1e-10:
                    break

        # Phase 1: Global Search with Latin Hypercube Sampling
        # Allocate 30% of budget to global search, but at least 50 points
        global_budget = max(50, budget // 3)
        if budget - calls_used < global_budget:
            global_budget = budget - calls_used

        # Generate LHS points
        n_points = global_budget
        rng = np.random.default_rng(42)
        
        # Latin Hypercube Sampling
        # For each dimension, divide the range into n_points equal intervals
        # and pick one point uniformly at random from each interval
        lhs_points = np.zeros((n_points, dim))
        for d in range(dim):
            # Shuffle indices to create the permutation
            perm = rng.permutation(n_points)
            # For each point i, the value in dimension d is in the interval
            # [perm[i] / n_points, (perm[i] + 1) / n_points]
            # Map to [-4, 4]
            lhs_points[:, d] = (perm + rng.random(n_points)) / n_points * 8.0 - 4.0

        # Evaluate all LHS points
        for i in range(n_points):
            if time.time() > deadline or calls_used >= budget:
                break
            x = lhs_points[i]
            fx = call_func(x)
            if fx < best_f:
                best_f = fx
                best_x = x.copy()

        # Phase 2: Local Refinement with Nelder-Mead
        if best_x is not None and calls_used < budget and time.time() < deadline:
            nelder_mead(best_x, max_iter=1000)

        return best_x, best_f
# EVOLVE-BLOCK-END
