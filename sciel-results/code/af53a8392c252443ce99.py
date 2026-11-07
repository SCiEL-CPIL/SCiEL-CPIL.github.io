"""Rewritten optimizer for BBOB f5 (Linear slope) using exact affine model fitting."""
# EVOLVE-BLOCK-START
import numpy as np


class Optimizer:
    """Exact linear-model optimizer for f(x) = w^T x + b on [-5,5]^d.
    
    Strategy:
      1. Evaluate at d+1 points forming an affine basis (origin + axis points).
      2. Solve for gradient w and offset b exactly.
      3. The minimum of a linear function over a box is at the vertex opposing the gradient.
      4. Evaluate the vertex to update the global best.
      
    This finds the exact optimum in d+2 calls.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        d = self.dim
        budget = self.budget
        
        # Step 1: Build affine basis
        # Evaluate at origin
        x0 = np.zeros(d)
        f0 = func(x0)
        
        # Evaluate at unit vectors to determine gradient components
        # f(e_i) = w_i + b  =>  w_i = f(e_i) - f0
        w = np.zeros(d)
        for i in range(d):
            xi = np.zeros(d)
            xi[i] = 1.0
            fi = func(xi)
            w[i] = fi - f0
        
        # Step 2: Determine the optimal vertex
        # For a linear function f(x) = w^T x + b, the minimum over [-5, 5]^d
        # is at x_i = -5 if w_i > 0, and x_i = 5 if w_i < 0.
        # If w_i == 0, the value is independent of x_i.
        
        # Vectorized calculation of x_opt
        # sign(w) is -1 if w < 0, 1 if w > 0, 0 if w == 0
        # We want x_i = -5 * sign(w_i)
        # If w_i > 0, sign is 1, x_i = -5. Correct.
        # If w_i < 0, sign is -1, x_i = 5. Correct.
        # If w_i == 0, sign is 0, x_i = 0. Correct.
        
        x_opt = -5.0 * np.sign(w)
        
        # Step 3: Evaluate the optimal vertex
        # This call updates the evaluator's "best value so far" to the true minimum.
        f_opt_est = func(x_opt)
        
        # We have used d + 2 calls.
        # For d=5, this is 7 calls.
        # The remaining budget is unused, which is fine as the optimum is found.
        
        return x_opt, f_opt_est
# EVOLVE-BLOCK-END
