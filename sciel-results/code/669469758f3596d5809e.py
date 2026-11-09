"""
Optimizer for Bent Cigar (f12) combining Sobol global search with 
(1+1)-Evolution Strategy for local refinement.
"""
import numpy as np
import time
from scipy.stats import qmc


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
        # 64 points to ensure we find the basin of attraction
        n_global = 64
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
            
        # --- Phase 2: Local Refinement with (1+1)-ES ---
        # The Bent Cigar function is highly anisotropic. (1+1)-ES with 
        # sigma adaptation is well-suited for this landscape.
        
        current_x = x0.copy()
        current_fx = best_fx
        sigma = 1.0  # Initial step size
        
        # 1/5th rule constants
        sigma_up = 1.5
        sigma_down = 0.87
        
        rng = np.random.default_rng(42)
        
        while calls_used < budget and time.time() < deadline:
            # Generate offspring
            y = current_x + sigma * rng.standard_normal(dim)
            
            # Clip to box [-5, 5]
            y = np.clip(y, -5.0, 5.0)
            
            # Evaluate offspring
            try:
                fy = wrapped_obj(y)
            except Exception:
                break
            
            # Update strategy parameters
            if fy < current_fx:
                # Success
                current_x = y
                current_fx = fy
                sigma *= sigma_up
            else:
                # Failure
                sigma *= sigma_down
            
            # Prevent sigma from becoming too small or too large
            if sigma < 1e-12:
                break
            if sigma > 10.0:
                sigma = 10.0

        # Final update of best if ES found a better point
        if current_fx < best_fx:
            best_fx = current_fx
            best_x = current_x

        return best_x, best_fx
