"""
Optimizer for Bent Cigar (f12) with anisotropy-aware adaptive search.
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
        deadline = start_time + 285  # Safety margin for 300s limit
        
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
        # Use more points to ensure we find the basin
        n_global = 128
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
        
        # If we didn't find any point, return center
        if best_x is None:
            x0 = np.zeros(dim)
            wrapped_obj(x0)
        else:
            x0 = best_x.copy()
            
        # --- Phase 2: Anisotropy-aware Adaptive Coordinate Descent ---
        # Bent Cigar: f(x) = x[0]^2 + 1e6 * sum(x[i]^2 for i>=1)
        # Dim 0 is flat, dims 1-4 are steep
        # Initialize with different step sizes per dimension
        # Dim 0: larger initial step, slower shrink
        # Dims 1-4: smaller initial step, faster shrink
        
        h = np.zeros(dim)
        h[0] = 2.0  # Larger step for the flat dimension
        for i in range(1, dim):
            h[i] = 0.1  # Smaller step for steep dimensions
        
        # Shrink factors per dimension
        shrink = np.zeros(dim)
        shrink[0] = 0.7  # Slower shrink for flat dim
        for i in range(1, dim):
            shrink[i] = 0.3  # Faster shrink for steep dims
        
        # Expand factor
        expand = 1.3
        
        current_x = x0.copy()
        current_fx = best_fx
        
        # Track which dimensions are "converged" (step too small)
        converged = np.zeros(dim, dtype=bool)
        min_h = 1e-8
        
        # Main Loop
        while calls_used < budget:
            if time.time() >= deadline:
                break
            
            improved = False
            
            for i in range(dim):
                if time.time() >= deadline or calls_used >= budget:
                    break
                
                if converged[i]:
                    continue
                
                if h[i] < min_h:
                    converged[i] = True
                    continue
                
                # Try positive and negative steps
                x_plus = current_x.copy()
                x_plus[i] += h[i]
                x_plus = np.clip(x_plus, -5, 5)
                
                x_minus = current_x.copy()
                x_minus[i] -= h[i]
                x_minus = np.clip(x_minus, -5, 5)
                
                # Evaluate
                fx_plus = wrapped_obj(x_plus)
                fx_minus = wrapped_obj(x_minus)
                
                # Determine best move
                if fx_plus < current_fx:
                    current_x = x_plus
                    current_fx = fx_plus
                    improved = True
                    # Success: Expand step size
                    h[i] = min(h[i] * expand, 5.0)
                elif fx_minus < current_fx:
                    current_x = x_minus
                    current_fx = fx_minus
                    improved = True
                    # Success: Expand step size
                    h[i] = min(h[i] * expand, 5.0)
                else:
                    # Failure: Shrink step size
                    h[i] *= shrink[i]
                    if h[i] < min_h:
                        converged[i] = True
            
            if improved:
                # If we improved, we might want to re-examine converged dims
                # occasionally in case we need to fine-tune
                if not improved and np.all(converged):
                    break
            else:
                # No improvement this pass
                if np.all(converged):
                    break
                
                # If not all converged but no improvement, try a small kick
                # on the non-converged dimensions
                for i in range(dim):
                    if not converged[i] and h[i] > min_h:
                        h[i] *= 1.5  # Kick to escape local issues
                
                # If we've done many passes without improvement, break
                # (safety check)
                if calls_used > budget - 10:
                    break

        return best_x, best_fx
