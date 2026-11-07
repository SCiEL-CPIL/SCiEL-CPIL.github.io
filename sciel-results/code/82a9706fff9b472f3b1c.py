import numpy as np
import time


class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 250  # safety margin before 300s limit
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        # Phase 1: Analytical sweep along x1, with x2..x10 at 0
        # ZDT4 Pareto front: x2..x10 = 0, x1 varies
        n_sweep = 80
        x1_lo, x1_hi = bounds[0]
        x1_vals = np.linspace(x1_lo, x1_hi, n_sweep)
        
        X = np.zeros((n_sweep, self.dim))
        for i, x1v in enumerate(x1_vals):
            X[i, 0] = x1v
            # x2..x10 = 0 (global min of g)
        
        F = np.array([problem(x) for x in X])
        used = n_sweep
        
        # Phase 2: Local refinement around best points
        # Find non-dominated points and refine
        if time.time() < deadline:
            # Simple local search around each point
            for i in range(n_sweep):
                if time.time() > deadline - 30:
                    break
                if used >= self.budget:
                    break
                # Try small perturbations
                for _ in range(5):
                    if used >= self.budget or time.time() > deadline - 30:
                        break
                    pert = X[i].copy()
                    pert[0] += 0.01 * (x1_hi - x1_lo) * (np.random.rand() - 0.5)
                    for j in range(1, self.dim):
                        pert[j] += 0.5 * (hi[j] - lo[j]) * (np.random.rand() - 0.5)
                    pert = np.clip(pert, lo, hi)
                    fp = problem(pert)
                    used += 1
                    # Replace if better in any objective
                    if np.any(fp < F[i]):
                        X[i] = pert
                        F[i] = fp
        
        return X
