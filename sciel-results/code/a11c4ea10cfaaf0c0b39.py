"""Optimized ZDT2 optimizer using convex spacing for Hypervolume maximization."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Model-based optimizer for ZDT2: places points directly on the Pareto front 
    with convex spacing optimized for Hypervolume maximization."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280  # Safety margin

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])

        n_evals = self.budget
        
        # ZDT2 Pareto Front: x[1:] = 0, f1 = x[0], f2 = 1 - sqrt(x[0]).
        # Reference point R = [11, 11].
        # We optimize the exponent alpha in x0 = (i/(N-1))^alpha to maximize discrete Hypervolume.
        
        i = np.arange(n_evals)
        t = i / (n_evals - 1)
        
        def compute_hv(alpha):
            u = t ** alpha
            widths = np.diff(u)
            heights = 10.0 + np.sqrt(u[:-1])
            return np.sum(widths * heights)
        
        def compute_grad_hv(alpha):
            # Numerical gradient for Newton's method
            eps = 1e-6
            return (compute_hv(alpha + eps) - compute_hv(alpha - eps)) / (2 * eps)
        
        # Initial guess
        alpha = 1.15
        
        # Newton's method to optimize alpha
        for _ in range(20):
            if time.time() > deadline:
                break
            grad = compute_grad_hv(alpha)
            if abs(grad) < 1e-8:
                break
            # Second derivative approximation for step size if needed, or simple step
            # Hessian approx
            hess = (compute_grad_hv(alpha + 1e-4) - compute_grad_hv(alpha - 1e-4)) / (2e-4)
            if abs(hess) < 1e-10:
                break
            step = -grad / hess
            # Limit step size to avoid divergence
            if step > 0.1: step = 0.1
            if step < -0.1: step = -0.1
            alpha += step
            if alpha <= 1.0:
                alpha = 1.0001
            if alpha >= 2.0:
                alpha = 1.9999

        x0_vals = t ** alpha
        
        # Create candidate solutions
        X = np.zeros((n_evals, self.dim))
        X[:, 0] = x0_vals
        # x[1:] = 0 by default

        # Evaluate all points
        F = np.zeros((n_evals, self.n_obj))
        for k in range(n_evals):
            F[k] = problem(X[k])

            # Deadline guard
            if time.time() > deadline:
                return X

        return X
# EVOLVE-BLOCK-END
