"""Optimized solver for BBOB f13 (Sharp ridge) using Sobol multi-start + Hybrid Local Search."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.stats import qmc


class Optimizer:
    """Multi-start local search with Sobol initialization and Hybrid NM/L-BFGS-B refinement."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        bounds = [(-5.0, 5.0)] * dim

        # Track function evaluations to stay within budget
        self._eval_count = 0
        self._best_f = np.inf
        self._best_x = None

        def tracked_func(x):
            if self._eval_count >= budget:
                return self._best_f
            x = np.clip(np.asarray(x, dtype=np.float64), -5.0, 5.0)
            self._eval_count += 1
            fx = float(func(x))
            if fx < self._best_f:
                self._best_f = fx
                self._best_x = x.copy()
            return fx

        # Phase 1: Sobol sampling for broad coverage
        # 32 points for dim=5 is a good balance for early progress and budget saving
        n_sobol = 32
        halton = qmc.Sobol(d=dim, scramble=True, seed=42)
        m = max(1, int(np.ceil(np.log2(n_sobol))))
        sobol_samples = halton.random(2 ** m)[:n_sobol]
        x_sobol = -5.0 + 10.0 * sobol_samples

        all_points = []
        all_values = []

        for x0 in x_sobol:
            if self._eval_count >= budget:
                break
            fx = tracked_func(x0)
            all_points.append(x0.copy())
            all_values.append(fx)

        # Phase 2: Sequential refinement
        # Sort by objective value
        order = np.argsort(all_values)
        
        # Hybrid Strategy:
        # 1. Refine the absolute best candidate with Nelder-Mead (robust for ridges)
        # 2. Refine the next few candidates with L-BFGS-B (fast polish)
        
        remaining = budget - self._eval_count
        
        # Refine top 1 with Nelder-Mead
        if len(order) > 0 and remaining > 20:
            x_start = all_points[order[0]].copy()
            try:
                # Nelder-Mead is derivative-free, good for ill-conditioned ridges
                # maxiter controls number of iterations, not function calls directly
                # NM typically uses ~1.5 * dim * maxiter calls? No, simplex update is 1 call per vertex update.
                # Let's allow a generous number of iterations.
                result = minimize(
                    tracked_func,
                    x_start,
                    method='Nelder-Mead',
                    options={'maxiter': 100, 'xatol': 1e-8, 'fatol': 1e-12}
                )
                if result.fun < self._best_f:
                    self._best_f = result.fun
                    self._best_x = result.x.copy()
            except Exception:
                pass

        # Refine next 3 candidates with L-BFGS-B
        remaining = budget - self._eval_count
        for idx in order[1:4]:
            if self._eval_count >= budget or remaining <= 10:
                break
            
            x_start = all_points[idx].copy()
            
            # L-BFGS-B for efficient convergence on smooth parts
            maxiter = max(1, min(50, (budget - self._eval_count) // 3))
            
            try:
                result = minimize(
                    tracked_func,
                    x_start,
                    method='L-BFGS-B',
                    bounds=bounds,
                    options={'maxiter': maxiter, 'ftol': 1e-14, 'gtol': 1e-10, 'maxls': 50}
                )
                
                if result.fun < self._best_f:
                    self._best_f = result.fun
                    self._best_x = result.x.copy()
                    
            except Exception:
                pass
                
            remaining = budget - self._eval_count

        return self._best_x, self._best_f
# EVOLVE-BLOCK-END
