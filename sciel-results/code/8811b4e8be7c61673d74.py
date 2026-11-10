"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import qmc


class Optimizer:
    """Deterministic Sobol sampling + robust Nelder-Mead local refinement."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.calls_made = 0

    def _call_func(self, func, x):
        """Call function and track budget."""
        if self.calls_made >= self.budget:
            return np.inf
        self.calls_made += 1
        try:
            return func(x)
        except Exception:
            return np.inf

    def _nelder_mead(self, func, x0, lower=-5.0, upper=5.0, max_iter=80):
        """
        Robust Nelder-Mead implementation.
        Returns (best_x, best_f).
        """
        dim = self.dim
        
        # Initialize simplex with adaptive step size
        step = 1.0
        simplex = [x0.copy()]
        for i in range(dim):
            xi = x0.copy()
            xi[i] += step
            if xi[i] > upper:
                xi[i] = x0[i] - step
            if xi[i] < lower:
                xi[i] = x0[i] + step
            simplex.append(xi)
        
        # Evaluate initial simplex
        fvals = []
        for v in simplex:
            fv = self._call_func(func, v)
            fvals.append(fv)
            
        if self.calls_made >= self.budget:
            best_idx = np.argmin(fvals)
            return simplex[best_idx].copy(), fvals[best_idx]

        # Nelder-Mead parameters
        alpha = 1.0
        gamma = 2.0
        rho = 0.5
        sigma = 0.5
        
        best_x = simplex[np.argmin(fvals)].copy()
        best_f = np.min(fvals)
        
        for _ in range(max_iter):
            if self.calls_made >= self.budget:
                break
            
            # Sort simplex by function value
            order = np.argsort(fvals)
            simplex = [simplex[i] for i in order]
            fvals = [fvals[i] for i in order]
            
            if fvals[0] < best_f:
                best_f = fvals[0]
                best_x = simplex[0].copy()
                
            # Convergence check
            spread = np.max(fvals) - np.min(fvals)
            if spread < 1e-10 * (1 + abs(fvals[0])):
                break
            
            # Centroid of all but worst point
            centroid = np.mean(simplex[:-1], axis=0)
            
            # Reflection
            x_r = centroid + alpha * (centroid - simplex[-1])
            x_r = np.clip(x_r, lower, upper)
            
            if self.calls_made >= self.budget:
                break
            f_r = self._call_func(func, x_r)
            
            if f_r < fvals[0]:
                if self.calls_made < self.budget:
                    x_e = centroid + gamma * (x_r - centroid)
                    x_e = np.clip(x_e, lower, upper)
                    f_e = self._call_func(func, x_e)
                    
                    if f_e < f_r:
                        simplex[-1] = x_e
                        fvals[-1] = f_e
                    else:
                        simplex[-1] = x_r
                        fvals[-1] = f_r
                else:
                    simplex[-1] = x_r
                    fvals[-1] = f_r
            elif f_r < fvals[-2]:
                simplex[-1] = x_r
                fvals[-1] = f_r
            else:
                if f_r < fvals[-1]:
                    x_c = centroid + rho * (x_r - centroid)
                else:
                    x_c = centroid - rho * (simplex[-1] - centroid)
                
                x_c = np.clip(x_c, lower, upper)
                
                if self.calls_made >= self.budget:
                    break
                f_c = self._call_func(func, x_c)
                
                if f_c < min(f_r, fvals[-1]):
                    simplex[-1] = x_c
                    fvals[-1] = f_c
                else:
                    for i in range(1, len(simplex)):
                        simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                        simplex[i] = np.clip(simplex[i], lower, upper)
                        
                        if self.calls_made >= self.budget:
                            break
                        fvals[i] = self._call_func(func, simplex[i])
                    
                    if self.calls_made >= self.budget:
                        break

            if self.calls_made >= self.budget:
                break

        return best_x, best_f

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        lower = -5.0
        upper = 5.0
        self.calls_made = 0

        # 1. Global Exploration with Sobol Sequence
        n_sobol = 32
        sampler = qmc.Sobol(d=dim, scramble=True, seed=42)
        sobol_points = sampler.random(n_sobol)
        sobol_points = sobol_points * (upper - lower) + lower

        # Evaluate all Sobol points and keep track of top candidates
        candidates = []
        for i in range(n_sobol):
            if self.calls_made >= budget:
                break
            x = sobol_points[i]
            fx = self._call_func(func, x)
            candidates.append((fx, x.copy()))

        # Sort by function value
        candidates.sort(key=lambda t: t[0])

        best_x = candidates[0][1]
        best_f = candidates[0][0]

        # 2. Local Refinement with Nelder-Mead on top candidates
        # Try refining the best 3 candidates
        n_refine = min(3, len(candidates))
        for i in range(n_refine):
            if self.calls_made >= budget:
                break
            if i == 0:
                x_start = candidates[0][1]
            else:
                x_start = candidates[i][1]
            
            refined_x, refined_f = self._nelder_mead(func, x_start, lower, upper)
            if refined_f < best_f:
                best_f = refined_f
                best_x = refined_x

        return best_x, best_f
# EVOLVE-BLOCK-END
