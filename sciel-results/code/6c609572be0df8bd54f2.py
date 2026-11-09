"""Rewritten optimizer: Deterministic Halton global search + Nelder-Mead local refinement."""
import time
import numpy as np


def _halton_sequence(n, dim, base):
    """Generate n Halton sequence points in [0,1]^dim using the given base."""
    points = np.zeros((n, dim))
    for d in range(dim):
        b = base[d]
        f = np.ones(n)
        for i in range(1, n + 1):
            f *= b
            points[i - 1, d] = (i * f) % 1.0
    return points


class Optimizer:
    """Global Halton sequence search followed by Nelder-Mead local refinement."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.calls = 0

    def __call__(self, func):
        t_start = time.time()
        deadline = t_start + 285.0
        dim = self.dim
        budget = self.budget
        self.calls = 0

        def _eval(x):
            if self.calls >= budget:
                return np.inf
            self.calls += 1
            return float(func(x))

        # Phase 1: Deterministic Halton sequence global search
        # Use 85% of budget for global exploration
        n_global = int(budget * 0.85)
        
        # Generate Halton sequence points
        # Use first dim primes as bases
        primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29]
        bases = primes[:dim]
        
        # Map [0,1]^dim to [-5,5]^dim
        halton_points = _halton_sequence(n_global, dim, bases)
        halton_points = halton_points * 10.0 - 5.0
        
        # Also add a few random points for robustness
        n_random = max(1, n_global // 10)
        rng = np.random.default_rng(42)
        random_points = rng.uniform(-5, 5, size=(n_random, dim))
        
        # Combine: Halton points first, then random points
        all_points = np.vstack([halton_points, random_points])
        
        best_x = np.zeros(dim)
        best_f = np.inf
        
        for i in range(len(all_points)):
            if self.calls >= budget or time.time() > deadline - 5.0:
                break
            
            x = all_points[i]
            x = np.clip(x, -5, 5)
            f = _eval(x)
            
            if f < best_f:
                best_f = f
                best_x = x.copy()
        
        # Phase 2: Nelder-Mead local refinement
        remaining_budget = budget - self.calls
        if remaining_budget > (dim + 2):
            # Build initial simplex around best_x
            simplex = [best_x.copy()]
            for d in range(dim):
                perturbed = best_x.copy()
                step = 0.5
                # Ensure we stay in bounds
                if perturbed[d] + step > 5.0:
                    step = -step
                if perturbed[d] + step < -5.0:
                    step = -step
                perturbed[d] += step
                simplex.append(perturbed)
            
            # Evaluate simplex
            fvals = []
            for s in simplex:
                if self.calls >= budget:
                    break
                fvals.append(_eval(s))
            
            if len(fvals) < dim + 1:
                return best_x, best_f

            # Nelder-Mead iterations
            alpha, gamma, rho, sigma_nm = 1.0, 2.0, 0.5, 0.5
            
            while self.calls < budget and time.time() < deadline:
                # Sort simplex
                order = np.argsort(fvals)
                simplex = [simplex[i] for i in order]
                fvals = [fvals[i] for i in order]
                
                if fvals[0] < best_f:
                    best_f = fvals[0]
                    best_x = simplex[0].copy()
                    
                # Check convergence
                if len(fvals) < 2:
                    break
                    
                # Centroid of all but worst
                centroid = np.mean(simplex[:-1], axis=0)
                
                # Reflection
                xr = centroid + alpha * (centroid - simplex[-1])
                xr = np.clip(xr, -5, 5)
                
                if self.calls >= budget: 
                    break
                fr = _eval(xr)
                
                if fvals[0] <= fr < fvals[-2]:
                    simplex[-1] = xr
                    fvals[-1] = fr
                elif fr < fvals[0]:
                    # Expansion
                    if self.calls >= budget: 
                        simplex[-1] = xr
                        fvals[-1] = fr
                        break
                    xe = centroid + gamma * (centroid - simplex[-1])
                    xe = np.clip(xe, -5, 5)
                    fe = _eval(xe)
                    if fe < fr:
                        simplex[-1] = xe
                        fvals[-1] = fe
                    else:
                        simplex[-1] = xr
                        fvals[-1] = fr
                else:
                    # Contraction
                    if self.calls >= budget: 
                        break
                    if fr < fvals[-1]:
                        # Outside contraction
                        xc = centroid + rho * (simplex[-1] - centroid)
                    else:
                        # Inside contraction
                        xc = centroid - rho * (simplex[-1] - centroid)
                    xc = np.clip(xc, -5, 5)
                    fc = _eval(xc)
                    
                    if fc < min(fr, fvals[-1]):
                        simplex[-1] = xc
                        fvals[-1] = fc
                    else:
                        # Shrink
                        for i in range(1, len(simplex)):
                            if self.calls >= budget: 
                                break
                            simplex[i] = simplex[0] + sigma_nm * (simplex[i] - simplex[0])
                            simplex[i] = np.clip(simplex[i], -5, 5)
                            fvals[i] = _eval(simplex[i])
        
        return best_x, best_f
