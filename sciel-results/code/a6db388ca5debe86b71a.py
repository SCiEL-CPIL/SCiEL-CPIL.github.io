"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multistart Nelder-Mead for BBOB f17."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 295.0
        rng = np.random.default_rng(42)
        self.dim = self.dim
        budget = self.budget
        calls_used = 0
        best_x = None
        best_f = float('inf')

        def nelder_mead(x0, func, budget_left, rng):
            nonlocal calls_used
            n = len(x0)
            # Initialize simplex
            simplex = [x0.copy()]
            for i in range(n):
                xi = x0.copy()
                if abs(xi[i]) < 1e-10:
                    xi[i] += 0.25
                else:
                    xi[i] += 0.25 * (1.0 if xi[i] > 0 else -1.0)
                simplex.append(xi)
            simplex = [np.clip(p, -5, 5) for p in simplex]
            fvals = [func(p) for p in simplex]
            calls_used += n + 1
            alpha, gamma, rho = 1.0, 2.0, 0.5
            min_idx = 0
            max_idx = 0
            for _ in range(budget_left - (n + 1)):
                if time.time() > deadline:
                    break
                # Sort simplex by function value
                order = np.argsort(fvals)
                simplex = [simplex[i] for i in order]
                fvals = [fvals[i] for i in order]
                min_idx, max_idx = 0, n
                centroid = np.mean(simplex[:-1], axis=0)
                # Reflection
                xr = centroid + alpha * (centroid - simplex[-1])
                xr = np.clip(xr, -5, 5)
                fr = func(xr)
                calls_used += 1
                if fr < fvals[0]:
                    # Expansion
                    xe = centroid + gamma * (xr - centroid)
                    xe = np.clip(xe, -5, 5)
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
                    if fr < fvals[-1]:
                        # Outside contraction
                        xc = centroid + rho * (xr - centroid)
                        fc = func(xc)
                        calls_used += 1
                        if fc <= fr:
                            simplex[-1] = xc
                            fvals[-1] = fc
                        else:
                            # Shrink
                            for i in range(1, n + 1):
                                simplex[i] = simplex[0] + 0.5 * (simplex[i] - simplex[0])
                                simplex[i] = np.clip(simplex[i], -5, 5)
                                fvals[i] = func(simplex[i])
                                calls_used += 1
                    else:
                        # Inside contraction
                        xc = centroid - rho * (centroid - simplex[-1])
                        fc = func(xc)
                        calls_used += 1
                        if fc < fvals[-1]:
                            simplex[-1] = xc
                            fvals[-1] = fc
                        else:
                            # Shrink
                            for i in range(1, n + 1):
                                simplex[i] = simplex[0] + 0.5 * (simplex[i] - simplex[0])
                                simplex[i] = np.clip(simplex[i], -5, 5)
                                fvals[i] = func(simplex[i])
                                calls_used += 1
                if calls_used >= budget:
                    break
                # Check convergence
                if max(fvals) - min(fvals) < 1e-10:
                    break
            idx = np.argmin(fvals)
            return simplex[idx], fvals[idx]

        # Allocate budget: use first 100 calls for initial exploration, then restart NM
        # Strategy: multiple restarts with decreasing budget per restart
        n_restarts = 5
        per_restart = max(10, (budget - 10) // n_restarts)
        
        for r in range(n_restarts):
            if time.time() > deadline:
                break
            if calls_used >= budget:
                break
            
            # Generate starting point
            if r == 0:
                x0 = np.zeros(self.dim)
            else:
                x0 = rng.uniform(-4, 4, self.dim)
            
            # Reserve some calls for this restart
            remaining = budget - calls_used
            alloc = min(per_restart, remaining)
            if alloc < 10:
                break
            
            x_best, f_best = nelder_mead(x0, func, alloc, rng)
            
            if f_best < best_f:
                best_f = f_best
                best_x = x_best.copy()
            
            if calls_used >= budget:
                break
        
        # If we still have budget, do a final local refinement around best point
        if best_x is not None and calls_used < budget and time.time() < deadline:
            remaining = budget - calls_used
            if remaining >= 10:
                x_ref, f_ref = nelder_mead(best_x.copy(), func, remaining, rng)
                if f_ref < best_f:
                    best_f = f_ref
                    best_x = x_ref.copy()
        
        if best_x is None:
            best_x = np.zeros(self.dim)
            best_f = func(best_x)
            calls_used += 1
        
        return best_x, best_f
# EVOLVE-BLOCK-END
