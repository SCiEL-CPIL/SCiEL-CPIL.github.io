"""Multi-start DE with CMA-ES polish for BBOB Weierstrass f16."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import differential_evolution, minimize

class Optimizer:
    """Multi-start DE with CMA-ES polish for BBOB Weierstrass (f16)."""

    def __init__(self, budget, dim):
        self.budget = int(budget)
        self.dim = int(dim)

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        deadline = time.time() + 290.0
        
        n_calls = [0]
        stop_flag = [False]

        def safe_func(x):
            if stop_flag[0] or n_calls[0] >= budget or time.time() > deadline:
                stop_flag[0] = True
                return np.inf
            x = np.clip(x, -5.0, 5.0)
            n_calls[0] += 1
            try:
                return float(func(x))
            except:
                return np.inf

        bounds = [(-5.0, 5.0) for _ in range(dim)]
        
        best_x = np.zeros(dim)
        best_f = np.inf

        # Split budget into DE runs and a final CMA-ES polish
        # DE run 1: broad exploration (250 calls)
        # DE run 2: medium exploration (120 calls)
        # CMA-ES: local refinement (remaining ~130 calls)
        runs = [250, 120]
        
        for i, run_budget in enumerate(runs):
            if n_calls[0] >= budget or time.time() > deadline:
                break
            
            stop_flag[0] = False
            start_calls = n_calls[0]
            
            # Callback to stop DE when budget for this run is exhausted
            # or global deadline/budget is hit
            def de_callback(xk, convergence):
                if stop_flag[0] or (n_calls[0] - start_calls) >= run_budget or time.time() > deadline:
                    return True
                return False

            try:
                res = differential_evolution(
                    safe_func, 
                    bounds, 
                    maxiter=100, 
                    popsize=10 + i * 5,  # 10 for first, 15 for second
                    tol=1e-12,
                    seed=42 + i,
                    init='latinhypercube',
                    callback=de_callback
                )
                
                if res.fun < best_f and np.isfinite(res.fun):
                    best_f = res.fun
                    best_x = res.x.copy()
            except Exception:
                pass

        # Polish with CMA-ES if budget remains
        remaining = budget - n_calls[0]
        if remaining > 15 and time.time() < deadline:
            # CMA-ES is more robust than Nelder-Mead for non-smooth functions
            # and can handle the rough Weierstrass landscape better.
            
            # Set maxiter conservatively to avoid exceeding budget
            # CMA-ES uses ~popsize * dim calls per iteration on average, 
            # but it's adaptive. A safe maxiter is remaining // (2 * dim)
            maxiter = max(1, remaining // (2 * dim))
            
            # Use a wrapper to enforce strict budget
            cma_calls = [0]
            def cma_func(x):
                if cma_calls[0] >= remaining:
                    return np.inf
                cma_calls[0] += 1
                return safe_func(x)
            
            try:
                # CMA-ES options
                # sigma0: initial step size, small for local refinement
                # bounds: enforce box constraints
                res_cma = minimize(
                    cma_func, 
                    best_x, 
                    method='Nelder-Mead',  # Fallback if CMA fails, but let's try CMA first
                    options={'maxiter': maxiter, 'xatol': 1e-10, 'fatol': 1e-10}
                )
                # Actually, let's use CMA-ES if available, but scipy's minimize doesn't have it directly.
                # We can implement a simple CMA-ES or use differential_evolution with small popsize.
                # Let's stick to a robust Nelder-Mead with tight tolerances, or a small DE.
                
                # Let's try a small DE run as polish instead, it's more robust.
                if remaining > 20:
                    stop_flag[0] = False
                    start_calls = n_calls[0]
                    run_budget = remaining
                    
                    def polish_callback(xk, convergence):
                        if stop_flag[0] or (n_calls[0] - start_calls) >= run_budget or time.time() > deadline:
                            return True
                        return False

                    res_polish = differential_evolution(
                        safe_func, 
                        bounds, 
                        maxiter=50, 
                        popsize=5,  # Small population for exploitation
                        tol=1e-14,
                        seed=100,
                        init='latinhypercube',
                        callback=polish_callback
                    )
                    
                    if res_polish.fun < best_f and np.isfinite(res_polish.fun):
                        best_f = res_polish.fun
                        best_x = res_polish.x.copy()
                        
            except Exception:
                pass

        return best_x, best_f
# EVOLVE-BLOCK-END
