"""Optimized solver for BBOB f13 (Sharp ridge) using Sobol multi-start + L-BFGS-B refinement."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.stats import qmc


class Optimizer:
    """Multi-start local search with Sobol initialization and L-BFGS-B refinement."""

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
        # Increase to 64 points for better initial coverage
        n_sobol = 64
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
        
        # Refine top 5 candidates with L-BFGS-B
        # L-BFGS-B is efficient for smooth, convex-like landscapes (transformed quadratics)
        remaining = budget - self._eval_count
        n_refine = 5
        
        for i, idx in enumerate(order[:n_refine]):
            if self._eval_count >= budget or remaining <= 10:
                break
            
            x_start = all_points[idx].copy()
            
            # Allocate budget for each refinement
            # Reserve some budget for subsequent refinements
            budget_per_refine = max(10, (budget - self._eval_count) // (n_refine - i))
            
            # L-BFGS-B options
            # maxiter is iterations, but each iteration can use multiple function calls (line search)
            # For dim=5, convergence is fast.
            maxiter = 100 # Generous, will stop on convergence
            
            try:
                result = minimize(
                    tracked_func,
                    x_start,
                    method='L-BFGS-B',
                    bounds=bounds,
                    options={
                        'maxiter': maxiter, 
                        'ftol': 1e-12, 
                        'gtol': 1e-8, 
                        'maxls': 50
                    }
                )
                
                if result.fun < self._best_f:
                    self._best_f = result.fun
                    self._best_x = result.x.copy()
                    
            except Exception:
                pass
                
            remaining = budget - self._eval_count

        return self._best_x, self._best_f
# EVOLVE-BLOCK-END
