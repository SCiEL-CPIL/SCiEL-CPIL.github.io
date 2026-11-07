"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Hybrid optimizer: multi-start Nelder-Mead with L-BFGS-B polish for Discus."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        deadline = time.time() + 280  # safety margin before 300s limit
        rng = np.random.default_rng(42)

        # Global budget tracking
        calls = [0]

        def count_call():
            calls[0] += 1
            return calls[0]

        def safe_func(x):
            """Wrapper that counts calls and clips to box."""
            x = np.asarray(x, dtype=float)
            x = np.clip(x, -5.0, 5.0)
            val = func(x)
            count_call()
            return val

        # Phase 1: Multi-start Nelder-Mead (derivative-free basin finding)
        # Use about 60% of budget for global exploration
        nm_budget = int(budget * 0.6)
        n_starts = min(10, max(3, nm_budget // 50))
        
        best_x = None
        best_f = np.inf

        # Initial diverse starting points
        starts = []
        # Center
        starts.append(np.zeros(dim))
        # Random points
        for i in range(n_starts - 1):
            starts.append(rng.uniform(-4, 4, dim))
        
        # Shuffle starts
        perm = rng.permutation(len(starts))
        starts = [starts[i] for i in perm]

        nm_per_start = max(20, nm_budget // len(starts))

        for start in starts:
            if calls[0] >= budget or time.time() > deadline:
                break
            
            x0 = np.clip(start, -5.0, 5.0)
            f0 = safe_func(x0)
            
            if f0 < best_f:
                best_f = f0
                best_x = x0.copy()
            
            # Nelder-Mead simplex
            simplex = [x0.copy()]
            fvals = [f0]
            
            # Build simplex around x0
            for j in range(dim):
                dx = x0.copy()
                step = 0.5 if abs(x0[j]) > 0.5 else 0.5
                dx[j] += step
                dx = np.clip(dx, -5.0, 5.0)
                fval = safe_func(dx)
                simplex.append(dx)
                fvals.append(fval)
            
            # Nelder-Mead iterations
            max_nm_iter = nm_per_start
            for it in range(max_nm_iter):
                if calls[0] >= budget or time.time() > deadline:
                    break
                
                # Sort simplex
                idx = np.argsort(fvals)
                simplex = [simplex[i] for i in idx]
                fvals = [fvals[i] for i in idx]
                
                # Check for convergence
                if abs(fvals[0] - fvals[-1]) < 1e-12:
                    break
                
                # Centroid of all but worst
                centroid = np.mean(simplex[:-1], axis=0)
                worst = simplex[-1]
                
                # Reflection
                xr = centroid + (centroid - worst)
                xr = np.clip(xr, -5.0, 5.0)
                fr = safe_func(xr)
                
                if fr < fvals[0]:
                    # Expansion
                    xe = centroid + 2 * (centroid - worst)
                    xe = np.clip(xe, -5.0, 5.0)
                    fe = safe_func(xe)
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
                    xc = centroid + 0.5 * (worst - centroid)
                    xc = np.clip(xc, -5.0, 5.0)
                    fc = safe_func(xc)
                    if fc < fvals[-1]:
                        simplex[-1] = xc
                        fvals[-1] = fc
                    else:
                        # Shrink
                        for i in range(1, len(simplex)):
                            simplex[i] = (simplex[0] + simplex[i]) / 2
                            fvals[i] = safe_func(simplex[i])
                
                # Update best
                min_idx = np.argmin(fvals)
                if fvals[min_idx] < best_f:
                    best_f = fvals[min_idx]
                    best_x = simplex[min_idx].copy()

        # Phase 2: L-BFGS-B polish (exploit local structure)
        # Use remaining budget for gradient-based refinement
        if calls[0] < budget and time.time() < deadline:
            from scipy.optimize import minimize
            
            # Try L-BFGS-B from best point
            try:
                res = minimize(
                    lambda x: safe_func(x),
                    best_x,
                    method='L-BFGS-B',
                    bounds=[(-5.0, 5.0)] * dim,
                    options={'maxiter': budget - calls[0], 'maxfun': budget - calls[0]}
                )
                if res.fun < best_f:
                    best_f = res.fun
                    best_x = res.x.copy()
            except Exception:
                pass
            
            # If still have budget, try another L-BFGS-B from a slightly perturbed point
            if calls[0] < budget and time.time() < deadline:
                try:
                    x_perturb = best_x + 0.1 * rng.standard_normal(dim)
                    x_perturb = np.clip(x_perturb, -5.0, 5.0)
                    res2 = minimize(
                        lambda x: safe_func(x),
                        x_perturb,
                        method='L-BFGS-B',
                        bounds=[(-5.0, 5.0)] * dim,
                        options={'maxiter': budget - calls[0], 'maxfun': budget - calls[0]}
                    )
                    if res2.fun < best_f:
                        best_f = res2.fun
                        best_x = res2.x.copy()
                except Exception:
                    pass

        # Phase 3: Final local polish with (1+1)-ES if budget remains
        if calls[0] < budget and time.time() < deadline:
            sigma = 0.01
            for _ in range(budget - calls[0]):
                if calls[0] >= budget or time.time() > deadline:
                    break
                y = np.clip(best_x + sigma * rng.standard_normal(dim), -5.0, 5.0)
                fy = safe_func(y)
                if fy < best_f:
                    best_x = y
                    best_f = fy
                    sigma *= 1.2
                else:
                    sigma *= 0.85
                    if sigma < 1e-8:
                        break

        return best_x, best_f
# EVOLVE-BLOCK-END
