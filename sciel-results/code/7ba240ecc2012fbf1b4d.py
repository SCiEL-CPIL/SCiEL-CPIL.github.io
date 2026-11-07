"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

class Optimizer:
    """Hybrid Sobol global search + Nelder-Mead local refinement."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        
        # Safety margin for time (evaluator kills at 330s, we stop at 290s)
        start_time = time.time()
        deadline = start_time + 290.0
        
        calls_made = 0
        best_x = None
        best_f = np.inf
        
        # Wrapper to count calls, enforce time/budget, and clip
        def wrapped_func(x):
            nonlocal calls_made, best_f, best_x
            if time.time() > deadline:
                return best_f
            if calls_made >= budget:
                return best_f
            
            x = np.clip(x, -5, 5)
            f_val = func(x)
            calls_made += 1
            
            if f_val < best_f:
                best_f = f_val
                best_x = x.copy()
                
            return f_val

        # Phase 1: Global Exploration
        # Use 30% of budget for global search to ensure we find a good basin
        # Weierstrass f16 is multimodal but smooth. 
        # Sobol sequence provides better space-filling than pure random.
        global_budget = max(50, budget // 3)
        
        # Generate Sobol sequence points
        # scipy.stats.qmc requires scipy >= 1.7
        try:
            from scipy.stats import qmc
            sampler = qmc.Sobol(d=dim, scramble=True, seed=42)
            # We need to generate enough points. 
            # qmc.random generates 2^m points.
            # We want roughly global_budget points.
            # Let's just generate a large batch and take the first global_budget
            # Or better, generate in chunks if needed, but for dim=5, 2^8=256 is plenty for 166 calls.
            # Let's aim for 2^ceil(log2(global_budget)) points to be safe and fast.
            n_points = 1
            while n_points < global_budget:
                n_points *= 2
            
            # Generate points in [0,1]^dim
            points_unit = sampler.random(n_points)
            
            # Map to [-4, 4]^dim (minimizer is known to be in this range)
            # Using [-4, 4] instead of [-5, 5] is slightly more efficient
            # as the center of mass of the search is where the optimum is likely.
            points = points_unit * 8 - 4
            
            # Evaluate these points
            for p in points:
                if time.time() > deadline or calls_made >= global_budget:
                    break
                wrapped_func(p)
                
        except ImportError:
            # Fallback to random if qmc is not available (unlikely in modern envs)
            rng = np.random.default_rng(42)
            for _ in range(global_budget):
                if time.time() > deadline or calls_made >= global_budget:
                    break
                p = rng.uniform(-4, 4, dim)
                wrapped_func(p)

        # If we didn't make any calls (shouldn't happen), fallback to origin
        if best_x is None:
            best_x = np.zeros(dim)
            best_f = wrapped_func(best_x)

        # Phase 2: Local Refinement using Nelder-Mead
        # Use the remaining budget.
        # Nelder-Mead is effective for smooth functions like Weierstrass.
        # We set maxiter to the remaining budget to strictly respect the call limit.
        remaining_budget = budget - calls_made
        
        if remaining_budget > 10:
            # Nelder-Mead options
            # 'maxiter' is the number of function evaluations? 
            # In SciPy, 'maxiter' is the maximum number of iterations.
            # Each iteration of Nelder-Mead in SciPy typically makes 1-2 function calls (reflection, expansion, contraction, shrink).
            # To be safe with the strict call count, we should limit maxiter.
            # A rough heuristic: maxiter = remaining_budget // 2 is safe, but we can try more.
            # Actually, SciPy's NelderMead counts function evaluations in 'nit' but 'maxiter' is iterations.
            # Let's use a wrapper that counts exactly, but SciPy doesn't expose a hard stop on *total* calls easily via options other than maxiter.
            # However, our wrapped_func stops counting after budget.
            # If we pass maxiter=remaining_budget, it might exceed the budget if it does >1 call/iter.
            # Better: set maxiter to a reasonable number that fits in remaining_budget.
            # For dim=5, Nelder-Mead usually converges in ~50-100 iterations.
            # Let's set maxiter = min(remaining_budget, 200) to be safe.
            nm_maxiter = min(remaining_budget, 200)
            
            res = minimize(
                wrapped_func, 
                best_x, 
                method='Nelder-Mead', 
                options={'maxiter': nm_maxiter, 'xatol': 1e-6, 'fatol': 1e-8}
            )
            
            # If Nelder-Mead found a better point, update
            # Note: wrapped_func already updates best_x and best_f internally
            # But we should check if res.fun is better just in case of internal state issues
            if res.fun < best_f:
                best_f = res.fun
                best_x = res.x

        return best_x, best_f
# EVOLVE-BLOCK-END
