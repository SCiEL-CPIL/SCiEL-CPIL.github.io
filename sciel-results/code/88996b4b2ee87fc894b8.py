"""
Direct Corner Search for BBOB f5 (Linear slope).

Strategy:
1. The function is linear (affine). The minimum of a linear function over a
   hypercube [-5, 5]^d is always located at one of the 2^d vertices (corners).
2. Instead of estimating the gradient (which costs 2d+1 calls and risks
   estimation errors or noise), we explicitly evaluate all 32 corners.
3. This guarantees finding the exact global minimum within the first 32 calls.
4. The remaining budget is used to "stall" by re-evaluating the best point found
   (or a point extremely close to it) to maintain a low error value b_i for the
   rest of the 500 calls, maximizing the AOCC score.
"""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """
    Optimizer that exploits the linearity of f5 by searching all box vertices.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290  # Safety margin for 300s limit
        
        # Generate all 2^dim vertices of the box [-5, 5]^dim
        # For dim=5, this is 32 points.
        vertices = np.array([[(-1)**(i >> j) * 5.0 for j in range(self.dim)] for i in range(2**self.dim)])
        
        best_x = None
        best_f = np.inf
        calls_made = 0
        
        # Phase 1: Evaluate all vertices to find the global minimum
        for x in vertices:
            if time.time() > deadline:
                break
            fx = func(x)
            calls_made += 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
                
        # If we didn't finish the vertices (unlikely for dim=5), 
        # we still have a best_x from whatever we evaluated.
        if best_x is None:
            # Fallback if something went wrong (e.g., exception in func?)
            # Just pick a random point
            best_x = np.random.uniform(-5, 5, self.dim)
            best_f = func(best_x)
            calls_made += 1

        # Phase 2: Stall to maximize AOCC
        # AOCC is based on the best value seen so far (b_i).
        # Since we found the global minimum in the first 32 calls,
        # b_i will be close to f_opt for all i >= 32.
        # To ensure we don't accidentally increase the error (if func is noisy, 
        # though the contract implies deterministic), or simply to consume the budget,
        # we re-evaluate the best point found.
        # The contract says "func(x) clips... and returns a float".
        # It doesn't explicitly say it's noise-free, but BBOB functions are deterministic.
        # Re-evaluating the same point returns the same value.
        
        # We must be careful not to exceed 500 calls.
        # Current calls: ~32.
        # Remaining: ~468.
        
        while calls_made < self.budget:
            if time.time() > deadline:
                break
            # Re-evaluate the best point to "keep" the best value
            # This is a no-op in terms of optimization but counts as a call.
            # Note: If the evaluator tracks unique points or penalizes repeats, 
            # this might be suboptimal, but standard black-box optimizers 
            # often just return the best found. The scoring is based on 
            # "best value after call i", so repeating the best value 
            # keeps b_i constant at the minimum.
            func(best_x)
            calls_made += 1
            
        return best_x, best_f
# EVOLVE-BLOCK-END
