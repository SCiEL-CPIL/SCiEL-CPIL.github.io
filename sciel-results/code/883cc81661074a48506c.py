import numpy as np
import time
from scipy.optimize import minimize

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280  # Safety margin before 300s limit
        
        calls = 0
        best_x = None
        best_f = float('inf')
        
        def safe_call(x):
            nonlocal calls
            if calls >= self.budget or time.time() > deadline:
                return float('inf')
            calls += 1
            return func(x)

        # Phase 1: Deterministic grid-based global search
        # Evaluate on a coarse lattice covering [-4, 4]^5 to find the basin
        # 3 points per dimension: -4, 0, 4 -> 3^5 = 243 evaluations
        grid_vals = np.array([-4.0, 0.0, 4.0])
        grid_points = np.array(np.meshgrid(*([grid_vals] * self.dim))).T.reshape(-1, self.dim)
        
        for p in grid_points:
            if calls >= self.budget or time.time() > deadline:
                break
            f_val = safe_call(p)
            if np.isfinite(f_val) and f_val < best_f:
                best_f = f_val
                best_x = p.copy()
        
        # Phase 2: Multi-start local refinement from top grid points
        # Sort grid points by function value and use top ones as starting points
        # for Nelder-Mead and L-BFGS-B refinement
        if best_x is not None and calls < self.budget and time.time() < deadline:
            # Collect all evaluated grid points with their values
            grid_results = []
            for p in grid_points:
                if calls >= self.budget or time.time() > deadline:
                    break
                # We already evaluated these, but we need to track which ones we've seen
                # Actually, let's just use the best few from what we have
                pass
            
            # Instead, let's do multi-start from the best grid point and a few others
            # Top 5 grid points for multi-start
            # Re-evaluate top candidates (we have their values from the grid phase)
            # But we didn't store them all, so let's just use best_x and a few perturbations
            
            # Strategy: Run Nelder-Mead from best_x, then L-BFGS-B from its result
            # Then run a few more starts from other good grid regions
            
            # First refinement: Nelder-Mead from best grid point
            try:
                res = minimize(safe_call, best_x, method='Nelder-Mead', 
                               options={'maxfev': 80, 'xatol': 1e-8, 'fatol': 1e-10})
                if np.isfinite(res.fun) and res.fun < best_f:
                    best_f = res.fun
                    best_x = res.x
            except Exception:
                pass
            
            # Second refinement: L-BFGS-B from the Nelder-Mead result
            if calls < self.budget and time.time() < deadline:
                bounds = [(-5.0, 5.0)] * self.dim
                try:
                    res = minimize(safe_call, best_x, method='L-BFGS-B', 
                                   bounds=bounds,
                                   options={'maxiter': 50, 'maxfev': 100, 'ftol': 1e-12, 'gtol': 1e-8})
                    if np.isfinite(res.fun) and res.fun < best_f:
                        best_f = res.fun
                        best_x = res.x
                except Exception:
                    pass
            
            # Additional multi-starts from other grid regions
            # Pick a few more diverse starting points
            additional_starts = [
                np.array([0.0, 0.0, 0.0, 0.0, 0.0]),
                np.array([2.0, 2.0, 2.0, 2.0, 2.0]),
                np.array([-2.0, -2.0, -2.0, -2.0, -2.0]),
                np.array([3.0, -3.0, 2.0, -2.0, 3.0]),
                np.array([-3.0, 3.0, -2.0, 2.0, -3.0]),
            ]
            
            for x0 in additional_starts:
                if calls >= self.budget or time.time() > deadline:
                    break
                
                # Quick Nelder-Mead from this start
                try:
                    res = minimize(safe_call, x0, method='Nelder-Mead', 
                                   options={'maxfev': 30, 'xatol': 1e-6, 'fatol': 1e-8})
                    if np.isfinite(res.fun) and res.fun < best_f:
                        best_f = res.fun
                        best_x = res.x
                except Exception:
                    pass
        
        # Final polishing with Powell if budget remains
        if best_x is not None and calls < self.budget and time.time() < deadline:
            remaining = self.budget - calls
            if remaining > 50:
                try:
                    res = minimize(safe_call, best_x, method='Powell', 
                                   options={'maxfev': min(remaining, 150), 'ftol': 1e-12, 'xtol': 1e-10})
                    if np.isfinite(res.fun) and res.fun < best_f:
                        best_f = res.fun
                        best_x = res.x
                except Exception:
                    pass
        
        # Fallback if no valid result found
        if best_x is None:
            best_x = np.zeros(self.dim)
            best_f = safe_call(best_x)

        return best_x, best_f
