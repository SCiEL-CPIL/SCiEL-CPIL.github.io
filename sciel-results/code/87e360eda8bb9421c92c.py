"""Optimizer using multi-start Nelder-Mead with adaptive restarts and time guard."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-start Nelder-Mead with adaptive restarts and time guard."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def _nelder_mead(self, func, x0, budget_left, time_limit, rng):
        """Nelder-Mead simplex method with budget and time guards."""
        n = self.dim
        # Build initial simplex
        simplex = [x0.copy()]
        for i in range(n):
            xi = x0.copy()
            xi[i] += 0.1 if x0[i] < 4.9 else -0.1
            simplex.append(xi)
        
        fvals = [func(np.clip(s, -5, 5)) for s in simplex]
        best_x = simplex[int(np.argmin(fvals))].copy()
        best_f = min(fvals)
        calls = n + 1
        
        # Nelder-Mead parameters
        alpha = 1.0  # reflection
        gamma = 2.0  # expansion
        rho = 0.5    # contraction
        sigma = 0.5  # shrink
        
        max_iterations = budget_left
        start_time = time.time()
        
        for iteration in range(max_iterations):
            # Check time
            if time.time() - start_time > time_limit:
                break
            
            # Sort simplex by function value
            order = np.argsort(fvals)
            simplex = [simplex[i] for i in order]
            fvals = [fvals[i] for i in order]
            
            # Check convergence
            if fvals[-1] - fvals[0] < 1e-10:
                break
            
            # Centroid of all but worst point
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
                    xc = centroid + rho * (simplex[-1] - centroid)
                    xc = np.clip(xc, -5, 5)
                    fc = func(xc)
                    calls += 1
                    if fc <= fr:
                        simplex[-1] = xc
                        fvals[-1] = fc
                    else:
                        # Shrink
                        for i in range(1, n + 1):
                            simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                            simplex[i] = np.clip(simplex[i], -5, 5)
                            fvals[i] = func(simplex[i])
                            calls += 1
                else:
                    # Shrink
                    for i in range(1, n + 1):
                        simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                        simplex[i] = np.clip(simplex[i], -5, 5)
                        fvals[i] = func(simplex[i])
                        calls += 1
            
            # Update best
            min_idx = int(np.argmin(fvals))
            if fvals[min_idx] < best_f:
                best_f = fvals[min_idx]
                best_x = simplex[min_idx].copy()
            
            # Check budget
            if calls >= budget_left:
                break
        
        return best_x, best_f, calls

    def __call__(self, func):
        rng = np.random.default_rng(0)
        dim = self.dim
        budget = self.budget
        
        # Time guard: 300s limit, use 280s as safety margin
        start_time = time.time()
        time_limit = 280.0
        
        # Phase 1: Random sampling to find good starting points
        n_samples = min(20, budget // 10)
        samples = []
        fvals = []
        
        for i in range(n_samples):
            x = rng.uniform(-5, 5, dim)
            fx = func(x)
            samples.append(x)
            fvals.append(fx)
        
        # Sort by function value
        order = np.argsort(fvals)
        samples = [samples[i] for i in order]
        fvals = [fvals[i] for i in order]
        
        best_x = samples[0].copy()
        best_f = fvals[0]
        calls_used = n_samples
        
        # Phase 2: Nelder-Mead from best starting points
        # Use remaining budget for Nelder-Mead
        remaining_budget = budget - calls_used
        
        # Try Nelder-Mead from top few starting points
        n_nm_starts = min(5, len(samples))
        
        for i in range(n_nm_starts):
            if time.time() - start_time > time_limit:
                break
            
            if remaining_budget < 10:
                break
            
            # Allocate budget to this Nelder-Mead run
            nm_budget = min(remaining_budget, max(10, remaining_budget // (n_nm_starts - i)))
            
            x_best, f_best, calls_made = self._nelder_mead(
                func, samples[i], nm_budget, time_limit, rng
            )
            
            calls_used += calls_made
            remaining_budget -= calls_made
            
            if f_best < best_f:
                best_f = f_best
                best_x = x_best
        
        return best_x, best_f
# EVOLVE-BLOCK-END
