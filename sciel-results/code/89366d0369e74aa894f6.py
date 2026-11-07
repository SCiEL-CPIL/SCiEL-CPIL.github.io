"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-start Nelder-Mead optimizer for BBOB f20 (Schwefel)."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def _nelder_mead(self, func, x0, budget, rng):
        """Nelder-Mead simplex method starting from x0."""
        dim = self.dim
        # Initialize simplex: x0 and dim perturbed vertices
        simplex = [np.array(x0, dtype=float)]
        for i in range(dim):
            v = np.array(x0, dtype=float)
            step = 0.5 if v[i] < 4.5 else -0.5
            v[i] += step
            v = np.clip(v, -5, 5)
            simplex.append(v)
        
        fvals = [func(v) for v in simplex]
        
        # Nelder-Mead parameters
        alpha = 1.0   # reflection
        gamma = 2.0   # expansion
        rho = 0.5     # contraction
        sigma_nm = 0.5 # shrinkage
        
        calls = dim + 1
        max_iter = max(1, (budget - calls) // (dim + 1))
        
        for _ in range(max_iter):
            # Sort simplex by function value
            order = np.argsort(fvals)
            simplex = [simplex[i] for i in order]
            fvals = [fvals[i] for i in order]
            
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
                    # Outside contraction
                    xc = centroid + rho * (xr - centroid)
                else:
                    # Inside contraction
                    xc = centroid - rho * (centroid - simplex[-1])
                xc = np.clip(xc, -5, 5)
                fc = func(xc)
                calls += 1
                if fc < min(fr, fvals[-1]):
                    simplex[-1] = xc
                    fvals[-1] = fc
                else:
                    # Shrinkage
                    for i in range(1, dim + 1):
                        simplex[i] = simplex[0] + sigma_nm * (simplex[i] - simplex[0])
                        simplex[i] = np.clip(simplex[i], -5, 5)
                        fvals[i] = func(simplex[i])
                        calls += 1
                    break
        return min(fvals), simplex[np.argmin(fvals)]

    def __call__(self, func):
        rng = np.random.default_rng(0)
        dim = self.dim
        budget = self.budget
        
        start_time = time.time()
        deadline = start_time + 290.0  # Safety margin
        
        # Multi-start strategy: use top random samples as initial points
        n_starts = 5
        per_start_budget = budget // n_starts
        
        best_x = None
        best_f = float('inf')
        
        # Generate diverse starting points
        candidates = []
        # Some uniform random
        for i in range(3):
            x = rng.uniform(-5, 5, dim)
            candidates.append(x)
        # Some biased toward center (where minimum is more likely)
        for i in range(2):
            x = rng.uniform(-3, 3, dim)
            candidates.append(x)
        
        # Evaluate all candidates first
        cand_fvals = [func(x) for x in candidates]
        budget_remaining = budget - len(candidates)
        
        # Sort by function value and run Nelder-Mead on top starts
        order = np.argsort(cand_fvals)
        
        for idx in order:
            if budget_remaining <= 0:
                break
            if time.time() > deadline:
                break
                
            x0 = candidates[idx]
            f0 = cand_fvals[idx]
            
            # Allocate remaining budget proportionally
            start_budget = min(budget_remaining, max(50, budget_remaining // (n_starts - idx)))
            
            fx, x_best = self._nelder_mead(func, x0, start_budget, rng)
            budget_remaining -= (start_budget - (dim + 1))  # Approximate calls used
            
            if fx < best_f:
                best_f = fx
                best_x = x_best
        
        return best_x, best_f
# EVOLVE-BLOCK-END
