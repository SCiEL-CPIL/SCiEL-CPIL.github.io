"""Rewritten optimizer for BBOB f5 (Linear slope) using exact affine model fitting."""
# EVOLVE-BLOCK-START
import numpy as np


class Optimizer:
    """Exact linear-model optimizer for f(x) = w^T x + b on [-5,5]^d."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        d = self.dim
        
        # Step 1: Evaluate at origin to get offset b
        x0 = np.zeros(d)
        b = func(x0)
        
        # Step 2: Evaluate at unit vectors to get gradient w
        w = np.zeros(d)
        for i in range(d):
            xi = np.zeros(d)
            xi[i] = 1.0
            w[i] = func(xi) - b
        
        # Step 3: Determine the optimal vertex
        # Minimize w^T x subject to x in [-5, 5]^d
        # x_i = -5 if w_i > 0, else 5 if w_i < 0
        # If w_i == 0, x_i can be anything, we pick 0.
        signs = np.sign(w)
        # If sign is 0, we want 0. If 1, we want -5. If -1, we want 5.
        # x_i = -5 * sign(w_i) works for non-zero w_i.
        # If w_i == 0, sign is 0, so x_i = 0.
        x_opt = -5.0 * signs
        
        # Step 4: Evaluate the optimal vertex
        f_opt = func(x_opt)
        
        return x_opt, f_opt
# EVOLVE-BLOCK-END
