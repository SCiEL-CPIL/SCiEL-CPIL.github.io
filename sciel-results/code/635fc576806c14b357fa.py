"""Optimizer for BBOB f5 (Linear slope) using exact affine model fitting."""
# EVOLVE-BLOCK-START
import numpy as np


class Optimizer:
    """
    Optimizer for linear functions f(x) = w^T x + b on the box [-5, 5]^d.
    
    Algorithm:
    1. Determine the affine parameters w and b using d+1 evaluations at the origin and unit vectors.
    2. Compute the optimal vertex x_opt where each coordinate is -5 if w_i > 0, else 5.
    3. Evaluate f(x_opt) to register the minimum value with the evaluator.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        d = self.dim
        
        # 1. Evaluate at the origin to get the offset b
        x_origin = np.zeros(d)
        f_origin = func(x_origin)
        
        # 2. Evaluate at unit vectors to get the gradient components w_i
        w = np.zeros(d)
        for i in range(d):
            x_unit = np.zeros(d)
            x_unit[i] = 1.0
            f_unit = func(x_unit)
            w[i] = f_unit - f_origin
            
        # 3. Determine the optimal vertex
        # For a linear function, the minimum over a box is at a vertex.
        # x_i = -5 if w_i > 0 (decreasing direction)
        # x_i =  5 if w_i < 0 (increasing direction, so min is at -w_i direction? No.)
        # f(x) = sum(w_i * x_i).
        # To minimize w_i * x_i:
        # If w_i > 0, choose smallest x_i -> -5.
        # If w_i < 0, choose largest x_i ->  5.
        
        x_opt = np.where(w > 0, -5.0, 5.0)
        
        # 4. Evaluate the optimal vertex to update the best score
        f_opt = func(x_opt)
        
        return x_opt, f_opt
# EVOLVE-BLOCK-END
