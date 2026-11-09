"""Black-box continuous optimizer for Rastrigin f3 on [-5,5]^5."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

class Optimizer:
    """Structure-exploiting grid search + Nelder-Mead for minimizing func on [-5,5]^dim."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        deadline = time.time() + 320.0  # Safety margin before 330s kill

        best_x = None
        best_f = float('inf')
        calls = 0

        def safe_func(x):
            nonlocal calls, best_f, best_x
            if calls >= budget:
                return float('inf')
            x_clipped = np.clip(x, -5.0, 5.0)
            fx = func(x_clipped)
            calls += 1
            if fx < best_f:
                best_f = fx
                best_x = x_clipped.copy()
            return fx

        # Phase 1: Coarse-to-fine 1D grid search exploiting Rastrigin's separability
        # Rastrigin: f(x) = 10*A + sum(x_i^2 - 10*cos(2*pi*x_i))
        # Period 2 in each dimension. The global minimum is at integer multiples of 2 (shifted).
        # We do a coarse grid to find the basin, then refine.

        # Coarse grid: 11 points per dimension (step ~1.0)
        # This covers [-5, 5] with step 1.0
        coarse_points = np.linspace(-5.0, 5.0, 11)
        
        # Evaluate all combinations of coarse grid (11^5 = 16105 is too many)
        # Instead, we do a sequential 1D search: fix all dims at 0, sweep each dim
        # But that's not quite right for shifted functions.
        
        # Better approach: Use a small set of starting points and do coordinate-wise refinement
        # Start with a few diverse points, then refine each coordinate
        
        # Generate initial diverse points
        rng = np.random.default_rng(42)
        init_points = []
        
        # Center point
        init_points.append(np.zeros(dim))
        
        # Halton sequence for diversity
        def halton(index, base):
            f = 1.0
            r = 0.0
            while index > 0:
                f /= base
                r += f * (index % base)
                index //= base
            return r
        
        for i in range(1, 11):
            p = np.array([halton(i, base) for base in [2, 3, 5, 7, 11][:dim]])
            p = p * 10.0 - 5.0
            init_points.append(p)
        
        # Evaluate initial points
        for p in init_points:
            if time.time() > deadline - 5.0 or calls >= budget:
                break
            fp = safe_func(p)
        
        if best_x is None:
            best_x = np.zeros(dim)
            safe_func(best_x)

        # Phase 2: Coordinate-wise refinement
        # For each coordinate, do a fine 1D search around the current best
        # This exploits the separability of Rastrigin
        
        # Fine grid refinement: for each dim, search [-1, 1] around current value with step 0.1
        for iteration in range(3):  # Multiple rounds of refinement
            if time.time() > deadline - 5.0 or calls >= budget:
                break
            
            for d in range(dim):
                if time.time() > deadline - 5.0 or calls >= budget:
                    break
                
                # Search range: current value ± 1.0, step 0.1
                center = best_x[d]
                search_range = np.linspace(center - 1.0, center + 1.0, 21)
                search_range = np.clip(search_range, -5.0, 5.0)
                
                best_val = best_f
                best_idx = None
                
                for idx, val in enumerate(search_range):
                    if time.time() > deadline - 5.0 or calls >= budget:
                        break
                    if abs(val - center) < 1e-6:  # Skip if same as current
                        continue
                    
                    test_x = best_x.copy()
                    test_x[d] = val
                    f_val = safe_func(test_x)
                    
                    if f_val < best_val:
                        best_val = f_val
                        best_idx = idx
                
                # If we found a better value, update best_x
                if best_idx is not None:
                    best_x[d] = search_range[best_idx]
                    best_f = best_val

        # Phase 3: Nelder-Mead polish
        if time.time() < deadline - 5.0 and calls < budget:
            nm_calls = 0
            def nm_func(x):
                nonlocal nm_calls, calls, best_f, best_x
                if calls >= budget:
                    return best_f
                nm_calls += 1
                x_clipped = np.clip(x, -5.0, 5.0)
                fx = func(x_clipped)
                calls += 1
                if fx < best_f:
                    best_f = fx
                    best_x = x_clipped.copy()
                return fx

            try:
                res = minimize(nm_func, best_x, method='Nelder-Mead', 
                               options={'maxiter': 200, 'xatol': 1e-10, 'fatol': 1e-14})
                
                if res.fun < best_f:
                    best_f = res.fun
                    best_x = np.clip(res.x, -5.0, 5.0)
            except Exception:
                pass

        return best_x, best_f
# EVOLVE-BLOCK-END
