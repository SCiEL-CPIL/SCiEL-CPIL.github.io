"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """(1+1)-ES with 1/5 success rule + Nelder-Mead polish."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = 325.0  # Safety margin before 330s kill
        budget = self.budget
        dim = self.dim

        def time_ok():
            return (time.time() - start_time) < deadline

        # Phase 1: (1+1)-ES for global exploration
        # Reserve calls for local polish
        es_budget = int(budget * 0.8)  # 400 calls for ES
        polish_budget = budget - es_budget  # 100 calls for polish

        rng = np.random.default_rng(0)
        
        # Try a few initial points, keep the best
        x = None
        fx = np.inf
        for _ in range(5):
            xc = rng.uniform(-5, 5, dim)
            fxc = func(xc)
            if fxc < fx:
                fx = fxc
                x = xc.copy()
        
        sigma = 1.5
        # 1/5 success rule: track successes over a window
        window_size = 10
        successes = 0
        window = 0
        sigma_min = 1e-6
        sigma_max = 10.0

        calls_used = 5  # already used 5 calls for init

        for i in range(es_budget):
            if not time_ok():
                break
            if calls_used >= budget:
                break
                
            # Generate candidate
            y = np.clip(x + sigma * rng.standard_normal(dim), -5, 5)
            fy = func(y)
            calls_used += 1
            
            if fy < fx:
                x = y
                fx = fy
                successes += 1
            window += 1
            
            # Adapt sigma every window_size steps
            if window >= window_size:
                success_rate = successes / window
                if success_rate > 0.2:
                    sigma *= 1.5
                elif success_rate < 0.2:
                    sigma *= 0.75
                else:
                    sigma *= 1.0
                sigma = np.clip(sigma, sigma_min, sigma_max)
                successes = 0
                window = 0

        # Phase 2: Nelder-Mead local polish
        if calls_used < budget and time_ok():
            x = self._nelder_mead(func, x, fx, budget - calls_used, dim, time_ok, start_time)
            # Update fx if improved
            fx = func(x)

        return x, fx

    def _nelder_mead(self, func, x0, fx0, max_iter, dim, time_ok, start_time):
        """Nelder-Mead local optimization with budget constraint."""
        # Initialize simplex
        simplex = [x0.copy()]
        fvals = [fx0]
        
        # Create initial simplex around x0
        step = 0.1
        for i in range(dim):
            xi = x0.copy()
            xi[i] += step
            xi = np.clip(xi, -5, 5)
            fi = func(xi)
            simplex.append(xi)
            fvals.append(fi)
        
        calls_used = dim + 1
        n = dim + 1  # simplex size
        
        # Nelder-Mead parameters
        alpha = 1.0
        gamma = 2.0
        rho = 0.5
        sigma_nm = 0.5
        
        for iteration in range(max_iter):
            if not time_ok() or calls_used >= max_iter:
                break
            
            # Sort simplex by function value
            idx = np.argsort(fvals)
            simplex = [simplex[i].copy() for i in idx]
            fvals = [fvals[i] for i in idx]
            
            # Centroid of all but worst
            centroid = np.mean(simplex[:-1], axis=0)
            
            # Reflection
            xr = centroid + alpha * (centroid - simplex[-1])
            xr = np.clip(xr, -5, 5)
            fr = func(xr)
            calls_used += 1
            
            if fr < fvals[0]:
                # Expansion
                xe = centroid + gamma * (centroid - simplex[-1])
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
                else:
                    # Inside contraction
                    xc = centroid - rho * (xr - centroid)
                xc = np.clip(xc, -5, 5)
                fc = func(xc)
                calls_used += 1
                if fc < min(fr, fvals[-1]):
                    simplex[-1] = xc
                    fvals[-1] = fc
                else:
                    # Shrink
                    for i in range(1, n):
                        simplex[i] = simplex[0] + sigma_nm * (simplex[i] - simplex[0])
                        simplex[i] = np.clip(simplex[i], -5, 5)
                        fvals[i] = func(simplex[i])
                        calls_used += 1
                        if calls_used >= max_iter:
                            break
                    if calls_used >= max_iter:
                        break
            
            # Check convergence
            fspread = max(fvals) - min(fvals)
            if fspread < 1e-12:
                break
        
        # Return best point
        best_idx = np.argmin(fvals)
        return simplex[best_idx]
# EVOLVE-BLOCK-END
