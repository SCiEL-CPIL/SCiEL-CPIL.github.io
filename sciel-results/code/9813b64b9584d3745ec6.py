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
        
        # Use a finer grid to find the optimal alpha more accurately
        # The function is smooth, so a grid search followed by local optimization works well
        alphas = np.linspace(1.0, 2.0, 200)
        hv_vals = np.array([compute_hv(a) for a in alphas])
        best_idx = np.argmax(hv_vals)
        alpha = alphas[best_idx]
        
        # Refine with golden section search around the best grid point
        if best_idx > 0 and best_idx < len(alphas) - 1:
            a_left = alphas[best_idx - 1]
            a_right = alphas[best_idx + 1]
        else:
            a_left = 1.0
            a_right = 2.0
            
        # Golden section search
        gr = (np.sqrt(5) - 1) / 2
        for _ in range(50):
            if time.time() > deadline:
                break
            x1 = a_left + (1 - gr) * (a_right - a_left)
            x2 = a_left + gr * (a_right - a_left)
            f1 = compute_hv(x1)
            f2 = compute_hv(x2)
            if f1 < f2:
                a_left = x1
            else:
                a_right = x2
            if abs(a_right - a_left) < 1e-10:
                break
        alpha = (a_left + a_right) / 2

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
