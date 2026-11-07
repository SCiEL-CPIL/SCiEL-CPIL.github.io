"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Exact affine model fitting for BBOB f5 (Linear slope)."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0

        best_x = None
        best_f = np.inf

        def update_best(x, f):
            nonlocal best_x, best_f
            if f < best_f:
                best_f = f
                best_x = x.copy()

        dim = self.dim
        budget = self.budget
        calls_used = 0

        # Phase 1: Fit exact affine model f(x) = w^T x + b
        # Evaluate at origin and at unit basis vectors scaled to reach boundary
        # Use step = 5.0 so that x_center + step*e_i hits the boundary
        # This gives maximum signal for gradient estimation

        x0 = np.zeros(dim)
        f0 = func(x0)
        calls_used += 1
        update_best(x0, f0)

        if time.time() > deadline:
            return best_x, best_f

        step = 0.1
        w = np.zeros(dim)

        for i in range(dim):
            if time.time() > deadline:
                break
            xi = x0.copy()
            xi[i] = step
            fi = func(xi)
            calls_used += 1
            update_best(xi, fi)
            # f(xi) - f0 = step * w_i  =>  w_i = (fi - f0) / step
            w[i] = (fi - f0) / step

        # Compute the optimal vertex of [-5, 5]^d for the linear model
        # f(x) = w^T x + b
        # To minimize, for each dimension i:
        #   if w[i] > 0, set x[i] = -5
        #   if w[i] < 0, set x[i] = +5
        #   if w[i] == 0, any value works; use 0
        x_opt = np.zeros(dim)
        for i in range(dim):
            if w[i] > 0:
                x_opt[i] = -5.0
            elif w[i] < 0:
                x_opt[i] = 5.0
            else:
                x_opt[i] = 0.0

        # Evaluate the computed optimum
        f_opt_val = func(x_opt)
        calls_used += 1
        update_best(x_opt, f_opt_val)

        return best_x, best_f
# EVOLVE-BLOCK-END
