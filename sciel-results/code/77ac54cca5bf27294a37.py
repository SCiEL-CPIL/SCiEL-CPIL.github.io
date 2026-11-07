"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize


class Optimizer:
    """Multi-start L-BFGS-B with diverse initializations."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        n = self.dim
        total_budget = self.budget
        rng = np.random.default_rng(42)
        
        best_x = None
        best_f = float('inf')
        calls_made = 0
        
        # Track function calls
        call_count = [0]
        
        def counting_func(x):
            if call_count[0] >= total_budget:
                return best_f  # Return best known to avoid wasted calls
            val = func(np.clip(x, -5, 5))
            call_count[0] += 1
            return val
        
        # Phase 1: Broad random sampling to find good basins (first 50 calls)
        n_random = min(50, total_budget)
        for _ in range(n_random):
            if call_count[0] >= total_budget:
                break
            x = rng.uniform(-4.5, 4.5, size=n)
            f = counting_func(x)
            if f < best_f:
                best_f = f
                best_x = x.copy()
        
        # Phase 2: Multi-start L-BFGS-B from diverse starting points
        # Use remaining budget for local optimization
        remaining_budget = total_budget - call_count[0]
        
        if remaining_budget > 20 and best_x is not None:
            # Generate multiple starting points: best point + perturbations + random
            start_points = [best_x.copy()]
            
            # Add perturbed versions of best point
            for scale in [0.5, 1.0, 2.0]:
                for _ in range(2):
                    perturbation = rng.normal(0, scale, size=n)
                    x_pert = best_x + perturbation
                    x_pert = np.clip(x_pert, -4.5, 4.5)
                    start_points.append(x_pert)
            
            # Add some random points
            for _ in range(3):
                x_rand = rng.uniform(-4.5, 4.5, size=n)
                start_points.append(x_rand)
            
            # Run L-BFGS-B from each start point
            for start in start_points:
                if call_count[0] >= total_budget:
                    break
                
                # Limit iterations based on remaining budget
                max_iter = max(1, (total_budget - call_count[0]) // 10)
                
                try:
                    result = minimize(
                        counting_func,
                        start,
                        method='L-BFGS-B',
                        bounds=[(-5, 5)] * n,
                        options={'maxiter': max_iter, 'ftol': 1e-12, 'gtol': 1e-12}
                    )
                    
                    if result.success or result.fun < best_f:
                        if result.fun < best_f:
                            best_f = result.fun
                            best_x = result.x.copy()
                except Exception:
                    pass
        
        # Phase 3: Final refinement around best point if budget remains
        if best_x is not None and call_count[0] < total_budget:
            # Do a few more random perturbations around best
            for scale in [0.1, 0.05, 0.01]:
                if call_count[0] >= total_budget:
                    break
                for _ in range(3):
                    if call_count[0] >= total_budget:
                        break
                    perturbation = rng.normal(0, scale, size=n)
                    x_ref = best_x + perturbation
                    x_ref = np.clip(x_ref, -5, 5)
                    f = counting_func(x_ref)
                    if f < best_f:
                        best_f = f
                        best_x = x_ref.copy()
        
        return best_x, best_f
# EVOLVE-BLOCK-END
