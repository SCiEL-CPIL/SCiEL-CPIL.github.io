# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """
    Optimizer for BBOB f5 (Linear slope).
    
    Strategy:
    1. Evaluate at origin to get baseline f0.
    2. For each dimension i:
       a. Evaluate at 5*e_i to determine gradient component v_i.
    3. Construct the global optimal vertex based on all gradient signs.
       For each dimension j:
       - If v_j > 0, the function increases with x_j, so set x_j = -5.
       - If v_j <= 0, the function decreases or is flat with x_j, so set x_j = 5.
    4. Evaluate this global vertex. This is the minimum for the linear function.
    5. Stall with this best point for the remaining budget.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 295
        
        d = self.dim
        calls_made = 0
        x_best = np.zeros(d)
        f_best = np.inf
        
        # Step 1: Evaluate at origin
        x0 = np.zeros(d)
        f0 = func(x0)
        calls_made += 1
        f_best = f0
        x_best = x0.copy()
        
        # Track gradient components
        v = np.zeros(d)
        
        # Step 2: Probe all axes to determine gradient signs
        for i in range(d):
            # Check time and budget
            if time.time() > deadline or calls_made >= self.budget:
                break
            
            # Probe axis i at distance 5
            xi = np.zeros(d)
            xi[i] = 5.0
            fi = func(xi)
            calls_made += 1
            
            # Update gradient component
            v[i] = (fi - f0) / 5.0
            
            # Update best if this axis point is better (unlikely for linear, but safe)
            if fi < f_best:
                f_best = fi
                x_best = xi.copy()
        
        # Step 3: Construct and evaluate the global optimal vertex
        if time.time() < deadline and calls_made < self.budget:
            x_opt = np.zeros(d)
            for j in range(d):
                if v[j] > 0:
                    x_opt[j] = -5.0
                else:
                    x_opt[j] = 5.0
            
            f_opt = func(x_opt)
            calls_made += 1
            
            if f_opt < f_best:
                f_best = f_opt
                x_best = x_opt.copy()
        
        # Step 4: Stall with the best point found
        while calls_made < self.budget:
            if time.time() > deadline:
                break
            func(x_best)
            calls_made += 1
        
        return x_best, f_best
# EVOLVE-BLOCK-END
