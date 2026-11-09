"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multistart Nelder-Mead with LHS initialization and deadline guard."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0  # safety margin before 330s limit
        dim = self.dim
        budget = self.budget
        rng = np.random.default_rng(42)

        best_x = None
        best_f = np.inf
        calls_used = 0

        # LHS (Latin Hypercube Sampling) for initial points
        # We'll use a few initial points spread across the box
        n_initial = min(20, budget // 10)
        
        # Generate LHS points
        lhs_points = []
        for i in range(n_initial):
            # For each dimension, pick a unique stratum
            point = np.zeros(dim)
            for d in range(dim):
                # Random permutation approach for LHS
                perm = rng.permutation(n_initial)
                point[d] = -5.0 + 10.0 * (perm[i] + 0.5) / n_initial
            lhs_points.append(point)
        
        # Evaluate initial points
        for x0 in lhs_points:
            if time.time() > deadline:
                break
            fx = func(x0)
            calls_used += 1
            if fx < best_f:
                best_f = fx
                best_x = x0.copy()

        # Now run Nelder-Mead from the best point
        if best_x is not None and calls_used < budget - 1:
            # Nelder-Mead implementation
            x = best_x.copy()
            f = best_f
            
            # Initialize simplex
            simplex = [x.copy()]
            fvals = [f]
            for i in range(dim):
                xi = x.copy()
                xi[i] += 0.1
                fi = func(xi)
                calls_used += 1
                simplex.append(xi)
                fvals.append(fi)
            
            # Nelder-Mead parameters
            alpha = 1.0  # reflection
            gamma = 2.0  # expansion
            rho = 0.5    # contraction
            sigma = 0.5  # shrink
            max_iter = budget - calls_used - 1
            
            for it in range(max_iter):
                if time.time() > deadline:
                    break
                
                # Sort simplex by function values
                idx = np.argsort(fvals)
                simplex = [simplex[i] for i in idx]
                fvals = [fvals[i] for i in idx]
                
                # Check for convergence
                if abs(fvals[0] - fvals[-1]) < 1e-12:
                    break
                
                # Centroid of all but worst point
                centroid = np.mean(simplex[:-1], axis=0)
                
                # Reflection
                xr = centroid + alpha * (centroid - simplex[-1])
                fr = func(np.clip(xr, -5, 5))
                calls_used += 1
                xr = np.clip(xr, -5, 5)
                
                if fr < fvals[0]:
                    # Expansion
                    xe = centroid + gamma * (xr - centroid)
                    fe = func(np.clip(xe, -5, 5))
                    calls_used += 1
                    xe = np.clip(xe, -5, 5)
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
                        xc = centroid + rho * (xr - centroid)
                        fc = func(np.clip(xc, -5, 5))
                        calls_used += 1
                        xc = np.clip(xc, -5, 5)
                        if fc <= fr:
                            simplex[-1] = xc
                            fvals[-1] = fc
                        else:
                            # Shrink
                            for i in range(1, len(simplex)):
                                simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                                fvals[i] = func(np.clip(simplex[i], -5, 5))
                                calls_used += 1
                            simplex[i] = np.clip(simplex[i], -5, 5)
                    else:
                        # Shrink
                        for i in range(1, len(simplex)):
                            simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                            fvals[i] = func(np.clip(simplex[i], -5, 5))
                            calls_used += 1
                        simplex[i] = np.clip(simplex[i], -5, 5)
                
                # Track best
                min_idx = np.argmin(fvals)
                if fvals[min_idx] < best_f:
                    best_f = fvals[min_idx]
                    best_x = simplex[min_idx].copy()
                
                if calls_used >= budget:
                    break
        
        return best_x, best_f
# EVOLVE-BLOCK-END
