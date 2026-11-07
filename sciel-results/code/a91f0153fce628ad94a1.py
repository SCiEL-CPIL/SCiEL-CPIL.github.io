"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """
    Optimizes the BBOB f5 (Linear slope) function by sequentially
    determining the optimal direction for each dimension.
    
    Strategy:
    1. Start at the center (0, ..., 0).
    2. For each dimension i:
       a. Probe the positive direction (+5) from the current best point.
       b. Estimate the gradient component for dimension i.
       c. Determine the optimal bound (-5 or +5) for dimension i.
       d. If the optimal bound is different from the probe direction,
          evaluate the new point. Otherwise, reuse the probe value.
       e. Update the current best point to include this optimized dimension.
    
    This ensures that the best-so-far value improves (or stays same) 
    at every step, maximizing the Area Under the Convergence Curve (AOCC).
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        deadline = time.time() + 290  # Safety margin for 300s limit
        
        # 1. Evaluate center
        x_curr = np.zeros(dim)
        f_curr = func(x_curr)
        f_best = f_curr
        x_best = x_curr.copy()
        calls_made = 1
        
        # 2. Incrementally determine gradient and optimize each dimension
        # For linear f(x) = g^T x + c:
        # We maintain x_curr with optimized dims 0..i-1 and 0 for i..dim-1.
        # To find g_i, we probe x_curr + 5*e_i.
        # f(x_curr + 5*e_i) = f(x_curr) + 5*g_i (since other dims are fixed or 0)
        # So g_i = (f(x_curr + 5*e_i) - f(x_curr)) / 5.
        
        for i in range(dim):
            if time.time() > deadline:
                break
            
            # Probe positive direction for dimension i
            x_probe = x_curr.copy()
            x_probe[i] += 5.0
            f_probe = func(x_probe)
            calls_made += 1
            
            if f_probe < f_best:
                f_best = f_probe
                x_best = x_probe.copy()
            
            # Calculate gradient component
            grad_i = (f_probe - f_curr) / 5.0
            
            # Determine optimal value for dimension i
            if grad_i > 0:
                x_curr[i] = -5.0
            else:
                x_curr[i] = 5.0
            
            # Evaluate the new point with optimized dimension i
            # Note: if grad_i < 0, x_probe was already x_curr (with 5.0 set), so f_probe is the value.
            # But to be safe and consistent, we evaluate x_curr.
            # Optimization: if grad_i < 0, x_curr[i] == 5.0, so x_curr == x_probe. We can reuse f_probe.
            # If grad_i > 0, x_curr[i] == -5.0, so x_curr != x_probe. We must evaluate.
            
            if grad_i < 0:
                f_curr = f_probe # x_curr is x_probe
            else:
                f_curr = func(x_curr)
                calls_made += 1
                
            if f_curr < f_best:
                f_best = f_curr
                x_best = x_curr.copy()
        
        # 3. Fill remaining budget with the optimal point
        # x_curr is now the optimal vertex
        while calls_made < self.budget:
            if time.time() > deadline:
                break
            f_val = func(x_curr)
            if f_val < f_best:
                f_best = f_val
            calls_made += 1
        
        return x_best, f_best
# EVOLVE-BLOCK-END
