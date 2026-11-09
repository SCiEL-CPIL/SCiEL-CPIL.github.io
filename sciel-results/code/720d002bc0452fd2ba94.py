"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-start Nelder-Mead with Sobol initialization and adaptive step sizes."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    @staticmethod
    def _sobol_points(n, dim, seed=0):
        """Generate n Sobol-like points in [0,1)^dim using Halton sequence (simpler, deterministic)."""
        rng = np.random.default_rng(seed)
        # Use a combination of structured + random for robustness
        points = np.zeros((n, dim))
        # First few points: structured grid-like
        for i in range(min(n, 4)):
            for j in range(dim):
                points[i, j] = (i + 1) / (min(n, 4) + 1) * 0.5 + rng.uniform(-0.1, 0.1)
        # Remaining: Halton-like with prime bases
        if n > 4:
            primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29]
            for i in range(4, n):
                for j in range(dim):
                    base = primes[j % len(primes)]
                    # Halton sequence
                    f, r = 1.0, i + 1
                    while r > 0:
                        f /= base
                        r = r // base
                    points[i, j] = f
        return np.clip(points, 0.01, 0.99)

    def _nelder_mead(self, func, x0, budget, max_iter=100, alpha=1.0, gamma=2.0, rho=0.5):
        """Nelder-Mead simplex method with adaptive step size."""
        dim = len(x0)
        # Initialize simplex
        simplex = np.zeros((dim + 1, dim))
        simplex[0] = x0
        for i in range(1, dim + 1):
            simplex[i] = x0.copy()
            simplex[i, i - 1] += 0.25  # initial step size
        
        fx = np.array([func(simplex[i]) for i in range(dim + 1)])
        best_idx = np.argmin(fx)
        best_x = simplex[best_idx].copy()
        best_fx = fx[best_idx]
        
        calls_used = dim + 1
        for _ in range(max_iter):
            if calls_used >= budget:
                break
            
            # Sort simplex by function value
            order = np.argsort(fx)
            simplex = simplex[order]
            fx = fx[order]
            
            # Update best
            if fx[0] < best_fx:
                best_fx = fx[0]
                best_x = simplex[0].copy()
            
            # Centroid of all but worst
            centroid = np.mean(simplex[:-1], axis=0)
            
            # Reflection
            xr = centroid + alpha * (centroid - simplex[-1])
            xr = np.clip(xr, -5, 5)
            fxr = func(xr)
            calls_used += 1
            
            if fxr < fx[0]:
                # Expansion
                xe = centroid + gamma * (centroid - simplex[-1])
                xe = np.clip(xe, -5, 5)
                fxe = func(xe)
                calls_used += 1
                if fxe < fxr:
                    simplex[-1] = xe
                    fx[-1] = fxe
                else:
                    simplex[-1] = xr
                    fx[-1] = fxr
            elif fxr < fx[-2]:
                simplex[-1] = xr
                fx[-1] = fxr
            else:
                # Contraction
                if fxr < fx[-1]:
                    xc = centroid + rho * (simplex[-1] - centroid)
                    fxc = func(xc)
                    calls_used += 1
                    if fxc <= fxr:
                        simplex[-1] = xc
                        fx[-1] = fxc
                    else:
                        # Shrink
                        for i in range(1, dim + 1):
                            simplex[i] = simplex[0] + 0.5 * (simplex[i] - simplex[0])
                            fx[i] = func(simplex[i])
                            calls_used += 1
                else:
                    # Shrink
                    for i in range(1, dim + 1):
                        simplex[i] = simplex[0] + 0.5 * (simplex[i] - simplex[0])
                        fx[i] = func(simplex[i])
                        calls_used += 1
            
            if calls_used >= budget:
                break
        
        return best_x, best_fx, calls_used

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        deadline = time.time() + 280  # safety margin
        
        # Phase 1: Multi-start exploration with Sobol points
        n_starts = 8
        sobol = self._sobol_points(n_starts, dim, seed=42)
        x_starts = sobol * 10.0 - 5.0  # map to [-5, 5]
        
        # Evaluate starting points
        starts = []
        for i in range(n_starts):
            x = x_starts[i]
            fx = func(x)
            starts.append((fx, x))
        
        # Sort by function value
        starts.sort()
        
        # Phase 2: Run Nelder-Mead from top 3 starts with allocated budgets
        # Allocate more budget to better starts
        allocations = [budget // 3, (budget // 3) - 1, budget - 2 * (budget // 3) - 1]
        # Ensure at least 10 calls each
        allocations = [max(10, a) for a in allocations]
        
        best_x, best_fx = starts[0][1], starts[0][0]
        
        for i in range(min(3, n_starts)):
            if time.time() > deadline:
                break
            alloc = min(allocations[i], budget - 2 * (dim + 1))  # account for initial evals
            if alloc < 10:
                break
            x, fx = self._nelder_mead(func, starts[i][1], alloc)
            if fx < best_fx:
                best_x, best_fx = x, fx
        
        # Phase 3: Final polish with (1+1)-ES from best point
        remaining = budget - sum(min(a, budget - 2 * (dim + 1)) for a in allocations[:3])
        remaining = max(0, remaining)
        
        if remaining > 5 and time.time() < deadline:
            x = best_x.copy()
            fx = best_fx
            sigma = 0.1
            for _ in range(min(remaining, 50)):
                if time.time() > deadline:
                    break
                y = np.clip(x + sigma * np.random.default_rng(0).standard_normal(dim), -5, 5)
                fy = func(y)
                if fy < fx:
                    x, fx = y, fy
                    sigma *= 1.3
                else:
                    sigma *= 0.8
            if fx < best_fx:
                best_x, best_fx = x, fx
        
        return best_x, best_fx
# EVOLVE-BLOCK-END
