"""Rewritten optimizer for BBOB f5 (Linear slope) using exact affine model fitting."""
# EVOLVE-BLOCK-START
import numpy as np


class Optimizer:
    """Exact linear-model optimizer for f(x) = w^T x + b on [-5,5]^d.
    
    Strategy:
      1. Evaluate at d+1 points: origin plus unit basis vectors.
      2. Compute gradient w_i = f(e_i) - f(0) for each i.
      3. The minimum of a linear function over [-5,5]^d is at the vertex 
         where each coordinate is -5 if w_i > 0, +5 if w_i < 0, 0 if w_i == 0.
      4. Evaluate at that vertex to get the optimal value.
      5. Return immediately — no further calls needed for a true linear function.
      
    Total calls: d + 2 = 7 for d=5.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        d = self.dim
        budget = self.budget
        
        # Evaluate at origin to get offset b
        x0 = np.zeros(d)
        f0 = func(x0)
        
        # Evaluate at unit vectors to get gradient components
        w = np.zeros(d)
        for i in range(d):
            xi = np.zeros(d)
            xi[i] = 1.0
            fi = func(xi)
            w[i] = fi - f0
        
        # Compute the optimal vertex: for a linear function f(x) = w^T x + b,
        # the minimum over [-5,5]^d is at x_i = -5 if w_i > 0, +5 if w_i < 0.
        x_opt = np.zeros(d)
        for i in range(d):
            if w[i] > 0:
                x_opt[i] = -5.0
            elif w[i] < 0:
                x_opt[i] = 5.0
            else:
                x_opt[i] = 0.0
        
        # Evaluate at the optimal vertex
        f_opt_est = func(x_opt)
        
        return x_opt, f_opt_est
# EVOLVE-BLOCK-END
