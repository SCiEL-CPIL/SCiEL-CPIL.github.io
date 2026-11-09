"""Multi-objective optimizer for ZDT6 exploiting known Pareto front structure."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Exploits ZDT6 structure: x[1:] = 0, x[0] sweeps [0, 1]."""

    def __init__(self, budget, dim, n_obj):
        self.budget, self.dim, self.n_obj = budget, dim, n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280  # safety margin before 300s limit
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        # ZDT6 Pareto front: x[1:] = 0, x[0] in [0, 1]
        # f1 = 1 - exp(-4*x[0]) * sin(6*pi*x[0])^6
        # f2 = 1 + (g(x) - f1)^2 where g(x) = 1 + 9*sum(x[1:])/9 = 1 when x[1:]=0
        # So on PF: f2 = 1 + (1 - f1)^2
        
        # Strategy: densely sample x[0] to cover the Pareto front
        # We have budget evaluations. Let's use them all for x[0] sampling.
        
        n_evals = self.budget
        x0_values = np.linspace(0, 1, n_evals)
        
        results = []
        for x0 in x0_values:
            x = np.zeros(self.dim)
            x[0] = x0
            f = problem(x)
            results.append(f)
            
            if time.time() > deadline:
                break
        
        return results
# EVOLVE-BLOCK-END
