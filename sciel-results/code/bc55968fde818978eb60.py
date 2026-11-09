"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize


class Optimizer:
    """Nelder-Mead simplex optimizer with adaptive restarts."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        # Use a deterministic starting point for reproducibility
        # Start at origin which is often close to the optimum in BBOB
        x0 = np.zeros(self.dim)
        
        # We will use Nelder-Mead which is derivative-free
        # Set maxiter to be slightly less than budget to account for overhead
        # Nelder-Mead calls func multiple times per iteration
        
        # To be safe with the 500 call budget, we limit iterations.
        # A typical Nelder-Mead step might take 2-5 function evaluations.
        # Let's aim for ~100-150 iterations max to stay well under 500 calls.
        
        # However, Nelder-Mead can be slow to converge if the initial simplex is bad.
        # Let's use a simpler, more robust approach: CMA-ES or a custom adaptive method.
        # Since SciPy is available, let's try a combination or a robust local search.
        
        # Actually, for BBOB f13 (Sharp Ridge), a global-ish search might be needed if the ridge is narrow.
        # But the shift is hidden. A good strategy is to start with a coarse grid or random restarts
        # and then refine.
        
        # Let's implement a simple Adaptive Random Search with Local Refinement.
        # 1. Perform a few random restarts to find a promising region.
        # 2. Use Nelder-Mead from the best point found.
        
        rng = np.random.default_rng(42)
        
        best_x = x0.copy()
        best_f = func(best_x)
        
        # Phase 1: Coarse exploration with random restarts
        # Use ~50% of budget for exploration
        explore_budget = self.budget // 2
        n_restarts = 10
        evals_per_restart = explore_budget // n_restarts if n_restarts > 0 else 0
        
        for i in range(n_restarts):
            if i == 0:
                x = x0.copy()
            else:
                x = rng.uniform(-5, 5, self.dim)
            
            fx = func(x)
            if fx < best_f:
                best_x = x.copy()
                best_f = fx
            
            # Simple local improvement from this point
            # Use a small number of Nelder-Mead steps
            try:
                res = minimize(
                    func, x, method='Nelder-Mead',
                    options={'maxiter': max(1, evals_per_restart // 5), 'xatol': 1e-4, 'fatol': 1e-8}
                )
                if res.fun < best_f:
                    best_x = res.x.copy()
                    best_f = res.fun
            except Exception:
                pass
        
        # Phase 2: Fine-tuning from the best point found
        # Use the remaining budget for a more intensive local search
        remaining_budget = self.budget - (explore_budget + 10) # Approximate
        if remaining_budget > 10:
            try:
                res = minimize(
                    func, best_x, method='Nelder-Mead',
                    options={'maxiter': remaining_budget // 5, 'xatol': 1e-6, 'fatol': 1e-10}
                )
                if res.fun < best_f:
                    best_x = res.x.copy()
                    best_f = res.fun
            except Exception:
                pass
        
        return best_x, best_f
# EVOLVE-BLOCK-END
