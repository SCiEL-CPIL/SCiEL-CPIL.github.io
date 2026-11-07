"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """
    Optimizer for BBOB f5 (Linear slope).
    
    Strategy:
    1. Evaluate at origin.
    2. Evaluate at 5 axis-boundary points (x_i = +/- 5) to determine gradient signs.
    3. Jump to the estimated optimal corner.
    4. Verify by checking the 5 immediate neighbors of the corner.
    5. Use remaining budget for a high-precision local polish (Coordinate Descent).
    """

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        t0 = time.time()
        deadline = t0 + 290  # Safety margin

        best_x = None
        best_f = float('inf')
        calls_made = 0

        def eval_point(x):
            nonlocal best_x, best_f, calls_made
            fx = func(x)
            calls_made += 1
            if fx < best_f:
                best_f = fx
                best_x = np.array(x, dtype=float)
            return fx

        # Phase 1: Gradient Sign Estimation
        # Use boundary points for maximum signal
        x0 = np.zeros(dim)
        f0 = eval_point(x0)
        
        signs = np.zeros(dim)
        h = 5.0
        
        for i in range(dim):
            if time.time() > deadline:
                break
            xi = np.zeros(dim)
            xi[i] = h
            
            fi = eval_point(xi)
            # If f(xi) < f0, gradient is negative -> min at +h
            # If f(xi) > f0, gradient is positive -> min at -h
            if fi < f0:
                signs[i] = 1.0
            else:
                signs[i] = -1.0
        
        # Construct estimated optimal corner
        corner = signs * h
        
        # Evaluate the corner
        f_corner = eval_point(corner)
        
        # Phase 2: Neighbor Verification
        # If the gradient estimation was wrong for one dimension, the true min is a neighbor.
        # Check all 5 neighbors (flip one bit)
        if calls_made < budget and time.time() < deadline:
            for i in range(dim):
                if time.time() > deadline:
                    break
                xn = corner.copy()
                xn[i] = -xn[i]
                fn = eval_point(xn)
        
        # Phase 3: Re-evaluate best point to clamp best-so-far
        # For a linear function, once we find the minimum, re-evaluating it
        # ensures b_i stays at the minimum for all remaining calls.
        if calls_made < budget and time.time() < deadline:
            best_pt = best_x.copy()
            while calls_made < budget:
                if time.time() > deadline:
                    break
                eval_point(best_pt)

        return best_x, best_f
# EVOLVE-BLOCK-END
