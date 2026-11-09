"""Black-box continuous optimizer for Rastrigin f3 on [-5,5]^5."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """Multi-start Nelder-Mead optimizer for minimizing func on [-5,5]^dim."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        start_time = time.time()
        deadline = start_time + 320.0  # Safety margin before 330s kill

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

        # Generate deterministic starting points:
        # 1. Center
        # 2. Vertices of a 5-simplex inscribed in [-5,5]^5
        # 3. A few additional spread points

        start_points = []
        
        # Center
        start_points.append(np.zeros(dim))
        
        # Vertices of a regular 5-simplex
        # The vertices of a regular simplex in d dimensions can be generated
        # using a Hadamard matrix or similar construction.
        # For dim=5, we can use the following construction:
        # v_i = e_i - (1/(d+1)) * 1, scaled appropriately.
        
        # A simpler approach: use the vertices of a hypercube and select a subset,
        # or use a specific simplex construction.
        
        # Let's use the standard simplex construction:
        # v_0 = 0
        # v_i = (1 - 1/(d+1)) * e_i + (1/(d+1)) * (1 - e_i) ... this is getting complex.
        
        # Alternative: Use a fixed set of well-spread points.
        # We'll use a combination of:
        # - Origin
        # - Points at +/- 3 in each dimension (2*dim = 10 points)
        # - Corners of a smaller cube
        
        # Point at origin
        # start_points already has origin
        
        # Points along axes
        for i in range(dim):
            for sign in [1, -1]:
                p = np.zeros(dim)
                p[i] = sign * 3.0
                start_points.append(p)
        
        # Points at corners of a smaller cube [-2,2]^dim
        # This gives 2^5 = 32 points, which might be too many.
        # Let's take a subset or use a different strategy.
        
        # Instead, let's use a Halton sequence for better space-filling properties
        # but only for the first few points to get diversity.
        
        def halton(index, base):
            f = 1.0
            r = 0.0
            while index > 0:
                f /= base
                r += f * (index % base)
                index //= base
            return r

        # Add 5 more Halton points for diversity
        for i in range(1, 6):
            p = np.array([halton(i, base) for base in [2, 3, 5, 7, 11][:dim]])
            p = p * 10.0 - 5.0
            start_points.append(p)
        
        # Evaluate all starting points
        for p in start_points:
            if time.time() > deadline - 10.0 or calls >= budget:
                break
            safe_func(p)
        
        if best_x is None:
            # Fallback: evaluate origin
            best_x = np.zeros(dim)
            best_f = safe_func(best_x)
        
        # Now run Nelder-Mead from the best starting point
        # We need to carefully manage the budget for Nelder-Mead
        
        nm_calls = 0
        nm_budget = budget - calls - 10  # Reserve 10 calls for safety
        
        def nm_func(x):
            nonlocal nm_calls, calls, best_f, best_x
            if calls >= budget or nm_calls >= nm_budget:
                return best_f
            nm_calls += 1
            x_clipped = np.clip(x, -5.0, 5.0)
            fx = func(x_clipped)
            calls += 1
            if fx < best_f:
                best_f = fx
                best_x = x_clipped.copy()
            return fx
        
        # Run Nelder-Mead
        try:
            from scipy.optimize import minimize
            res = minimize(nm_func, best_x, method='Nelder-Mead',
                          options={'maxiter': 200, 'xatol': 1e-10, 'fatol': 1e-12})
            if res.fun < best_f:
                best_f = res.fun
                best_x = np.clip(res.x, -5.0, 5.0)
        except Exception:
            pass
        
        # If we still have budget, try a few more Nelder-Mead runs from other good points
        # This can help if the first NM got stuck in a local minimum
        
        # Sort start_points by their function values (we need to re-evaluate or track)
        # For simplicity, let's just try NM from a few other promising points
        
        # Actually, let's try a simple restart strategy:
        # If we have budget left, try NM from the second-best starting point
        
        if time.time() < deadline - 5.0 and calls < budget - 20:
            # Find the second-best starting point
            start_values = []
            for p in start_points:
                if time.time() > deadline - 5.0:
                    break
                # We need to evaluate these points, but we already did in the first loop
                # So we can't easily get their values without re-evaluating
                # Let's just skip this for now and rely on the first NM run
            
            pass
        
        return best_x, best_f
# EVOLVE-BLOCK-END
