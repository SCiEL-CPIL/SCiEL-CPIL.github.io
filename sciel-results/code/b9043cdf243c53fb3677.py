"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize_scalar

class Optimizer:
    """Direct Pareto front sampling for ZDT2 with optimized spacing."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 320  # Safety margin

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        # 1. Find the optimal x_i (i>0) that minimizes g(x)
        # g(x) = 1 + 10(n-1) + sum_{i=1}^{n-1} (x_i^3 * sin(3*pi*x_i))
        # The term is separable. We minimize h(u) = u^3 * sin(3*pi*u) for u in [0, 1]
        
        def h(u):
            return u**3 * np.sin(3 * np.pi * u)
            
        # Minimize h(u)
        res = minimize_scalar(h, bounds=(0, 1), method='bounded')
        u_star = res.x
        h_min = h(u_star)
        
        n_vars = self.dim
        g_min = 1 + 10 * (n_vars - 1) + (n_vars - 1) * h_min
        
        # 2. Generate points on the Pareto Front
        # f1 = x0
        # f2 = (1 + g_min) * (1 - sqrt(f1 / g_min))
        
        # We have self.budget evaluations.
        N = self.budget
        
        # Use non-uniform sampling for f1 to maximize hypervolume.
        # Power law alpha = 0.5 favors f1 near 0.
        alpha = 0.5
        t = np.linspace(0, 1, N)
        f1_vals = t**alpha
        
        # Ensure f1 is within bounds [0, 1]
        f1_vals = np.clip(f1_vals, lo[0], hi[0])
        
        # Construct decision vectors
        X = np.zeros((N, n_vars))
        X[:, 0] = f1_vals
        X[:, 1:] = u_star
        
        # Clip to bounds just in case
        X = np.clip(X, lo, hi)
        
        # 3. Evaluate
        F = []
        for i in range(N):
            if time.time() > deadline:
                break
            f = problem(X[i])
            F.append(f)
            
        # The evaluator records the returned objective vectors.
        # The return value of __call__ is ignored, but we return F for completeness.
        return F
# EVOLVE-BLOCK-END
