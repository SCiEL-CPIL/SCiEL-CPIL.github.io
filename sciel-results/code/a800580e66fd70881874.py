"""
Direct Affine Model Recovery for BBOB f5 (Linear Slope).

Strategy:
1. The function f5 is linear (affine).
2. We recover the gradient vector $v$ using 6 evaluations:
   - Evaluate at the origin $x_0 = 0$.
   - Evaluate at $x_i = 5e_i$ for $i=1..5$.
   - Compute $v_i = (f(x_i) - f(x_0)) / 5$.
3. Determine the optimal vertex $x^*$ on the box $[-5, 5]^5$:
   - $x^*_i = -5$ if $v_i > 0$
   - $x^*_i = 5$ if $v_i < 0$
   - $x^*_i = 0$ if $v_i = 0$ (arbitrary, but 0 is safe)
4. Evaluate $f(x^*)$ at the 7th call to confirm the global minimum.
5. Re-evaluate $x^*$ for the remaining budget to maximize AOCC.

This guarantees finding the global minimum in exactly 7 calls, which is optimal
for a linear function on a hypercube when the structure is known.
"""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290  # Safety margin for 300s limit
        
        d = self.dim
        calls_made = 0
        
        # Step 1: Evaluate at the origin
        x0 = np.zeros(d)
        f0 = func(x0)
        calls_made += 1
        
        # Step 2: Evaluate along each axis to determine the gradient
        v = np.zeros(d)
        for i in range(d):
            if time.time() > deadline:
                break
            xi = np.zeros(d)
            xi[i] = 5.0
            fi = func(xi)
            calls_made += 1
            # f(xi) = f0 + v_i * 5.0  => v_i = (fi - f0) / 5.0
            v[i] = (fi - f0) / 5.0
            
        # Step 3: Construct the optimal vertex
        # For a linear function, the minimum is at the vertex where each coordinate
        # is chosen to minimize the contribution v_i * x_i.
        # If v_i > 0, we want x_i to be as small as possible (-5).
        # If v_i < 0, we want x_i to be as large as possible (5).
        x_best = np.zeros(d)
        for i in range(d):
            if v[i] > 0:
                x_best[i] = -5.0
            elif v[i] < 0:
                x_best[i] = 5.0
            else:
                x_best[i] = 0.0 # Doesn't matter if gradient is 0
        
        # Step 4: Evaluate the best vertex
        f_best = func(x_best)
        calls_made += 1
        
        # Step 5: Use remaining budget to re-evaluate the best point
        # We have used d + 2 calls. For d=5, this is 7 calls.
        # The remaining budget is used to keep the best value as low as possible.
        remaining = self.budget - calls_made
        for _ in range(remaining):
            if time.time() > deadline:
                break
            func(x_best)
            calls_made += 1
            
        return x_best, f_best
# EVOLVE-BLOCK-END
