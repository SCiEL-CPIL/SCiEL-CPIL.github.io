"""
Optimizer for Bent Cigar (f12) combining Sobol global search with 
Adaptive Nelder-Mead local refinement.
"""
import numpy as np
import time
from scipy.stats import qmc
from scipy.optimize import minimize


class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290  # Safety margin for 300s limit
        
        dim = self.dim
        budget = self.budget
        
        # State tracking
        calls_used = 0
        best_x = None
        best_fx = np.inf
        
        # Objective wrapper to count calls and track best
        def wrapped_obj(x):
            nonlocal calls_used, best_x, best_fx
            if time.time() >= deadline or calls_used >= budget:
                raise Exception("Limit exceeded")
            
            x = np.asarray(x, dtype=np.float64)
            if x.shape != (dim,):
                x = x.reshape(-1)
            
            calls_used += 1
            
            # Evaluate
            fx = func(x)
            
            # Update best
            if fx < best_fx:
                best_fx = fx
                best_x = x.copy()
                
            return fx

        # --- Phase 1: Global Search with Sobol Sequence ---
        # Use a Sobol sequence for low-discrepancy sampling
        # 32 points is sufficient to locate the basin in 5D
        n_global = 32
        sobol = qmc.Sobol(d=dim, scramble=True, seed=42)
        samples = sobol.random(n_global)
        
        # Map [0, 1]^d to [-5, 5]^d
        samples = samples * 10.0 - 5.0
        
        # Evaluate all global samples
        for x in samples:
            if time.time() >= deadline or calls_used >= budget:
                break
            try:
                wrapped_obj(x)
            except Exception:
                break
        
        # If we didn't find any point (shouldn't happen), return center
        if best_x is None:
            x0 = np.zeros(dim)
            try:
                wrapped_obj(x0)
            except Exception:
                return None, np.inf
        else:
            x0 = best_x.copy()
            
        # --- Phase 2: Local Refinement with Powell's Method ---
        # Powell's method is effective for anisotropic functions like Bent Cigar
        # because it adapts step sizes along the principal axes of the function's contours.
        remaining_budget = budget - calls_used
        
        if remaining_budget > 0 and time.time() < deadline:
            try:
                res = minimize(
                    wrapped_obj,
                    x0,
                    method="Powell",
                    options={
                        "maxiter": 1000,
                        "maxfev": remaining_budget,
                        "xrtol": 1e-10,
                        "ftol": 1e-10,
                    },
                )
                # Update best if Powell found something better
                xf = np.asarray(res.x, dtype=np.float64)
                ff = res.fun
                if np.all(np.isfinite(xf)) and np.isfinite(ff):
                    if ff < best_fx:
                        best_fx = ff
                        best_x = np.clip(xf, -5.0, 5.0)
            except Exception:
                pass

        return best_x, best_fx
