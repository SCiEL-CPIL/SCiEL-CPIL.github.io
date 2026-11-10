"""Multi-objective optimizer on WFG4 with Sobol sequence-based sampling."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.stats import qmc


class Optimizer:
    """Sobol sequence-based optimizer for WFG4."""

    def __init__(self, budget, dim, n_obj):
        self.budget = int(budget)
        self.dim = int(dim)
        self.n_obj = int(n_obj)

    def __call__(self, problem, bounds):
        deadline = time.time() + 280.0
        dim, n_obj = self.dim, self.n_obj
        lo = np.array([b[0] for b in bounds], dtype=float)
        hi = np.array([b[1] for b in bounds], dtype=float)
        span = hi - lo

        # Generate Sobol sequence points for systematic coverage
        n_samples = self.budget
        sobol = qmc.Sobol(d=dim, scramble=True)
        
        # Sobol requires power of 2, find next power of 2 >= n_samples
        n_pow2 = 1
        while n_pow2 < n_samples:
            n_pow2 *= 2
        
        # Generate Sobol samples in [0,1]^d
        sobol_samples = sobol.random(n_pow2)
        # Take first n_samples points
        sobol_samples = sobol_samples[:n_samples]
        
        # Scale to bounds
        X = lo + sobol_samples * span
        
        # Evaluate all points
        F = np.zeros((n_samples, n_obj))
        for i in range(n_samples):
            if time.time() > deadline:
                break
            F[i] = problem(X[i])
        
        return X
# EVOLVE-BLOCK-END
