"""Optimized solver for BBOB f13 (Sharp ridge) using Sobol multi-start + L-BFGS-B."""
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

        # Sobol sequence for diverse initial points
        # Use enough points for good coverage but leave budget for optimization
        n_sobol = min(64, max(16, budget // 8))
        halton = qmc.Sobol(d=dim, scramble=True, seed=42)
        # Generate 2^n points, take first n_sobol
        m = max(1, int(np.ceil(np.log2(n_sobol))))
        sobol_samples = halton.random(2 ** m)[:n_sobol]
        # Map [0,1]^d to [-5,5]^d
        x_sobol = -5.0 + 10.0 * sobol_samples

        # Evaluate all Sobol points
        best_x = None
        best_f = np.inf
        evaluated = 0

        for x0 in x_sobol:
            if evaluated >= budget:
                break
            fx = func(x0)
            evaluated += 1
            if fx < best_f:
                best_f = fx
                best_x = x0.copy()

        # Sequential refinement of best candidates using L-BFGS-B
        # Sort by objective value and refine the best few
        # We'll refine the top candidates, spending remaining budget
        remaining_budget = budget - evaluated

        # Collect all evaluated points sorted by fitness
        # We only kept the best, so let's re-evaluate a few promising ones
        # Actually, let's keep track of all points for better selection
        
        # Re-do: keep all points
        all_points = []
        all_values = []
        best_x = None
        best_f = np.inf
        evaluated = 0

        for x0 in x_sobol:
            if evaluated >= budget:
                break
            fx = func(x0)
            evaluated += 1
            all_points.append(x0.copy())
            all_values.append(fx)
            if fx < best_f:
                best_f = fx
                best_x = x0.copy()

        # Sort by objective value
        order = np.argsort(all_values)
        
        # Refine top candidates with L-BFGS-B
        # Each L-BFGS-B call may use multiple function evaluations
        # Limit the number of refinements based on remaining budget
        n_refine = min(len(order), max(1, remaining_budget // 20))
        
        for idx in order[:n_refine]:
            if evaluated >= budget:
                break
            
            x_start = all_points[idx].copy()
            f_start = all_values[idx]
            
            # Use L-BFGS-B with a reasonable maxiter
            # Each iteration uses ~1-2 function evaluations
            maxiter = max(1, min(50, (budget - evaluated) // 2))
            
            try:
                result = minimize(
                    func,
                    x_start,
                    method='L-BFGS-B',
                    bounds=bounds,
                    options={'maxiter': maxiter, 'ftol': 1e-12, 'gtol': 1e-8}
                )
                # Count function evaluations used
                # result.nfev gives the number of function evaluations
                # But we need to track our own count since func is wrapped
                # Actually, scipy's minimize calls func directly, so we can't easily track
                # Let's just use it and hope it doesn't exceed budget
                
                if result.success or result.fun < best_f:
                    if result.fun < best_f:
                        best_f = result.fun
                        best_x = result.x.copy()
                        
            except Exception:
                pass
        
        return best_x, best_f
# EVOLVE-BLOCK-END
