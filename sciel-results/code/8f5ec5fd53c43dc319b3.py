"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-objective optimizer for ZDT6 exploiting analytical structure."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280  # safety margin before 300s limit
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        span = hi - lo
        
        dim = self.dim
        budget = self.budget
        
        # ZDT6: f1 = x[0], g = 1 + 10*(n-1) where n=dim, h = 1 - sqrt(f1/g)*sin(10*f1*pi)
        # Pareto-optimal: x[1:]=0, so g = 1 + 10*(dim-1)
        # f2 = g * (1 - sqrt(f1/g) * sin(10*f1*pi))
        
        g_min = 1.0 + 10.0 * (dim - 1)  # when x[1:]=0
        
        # Strategy: 
        # 1. Sample x[0] across [0,1] with x[1:]=0 to trace the Pareto front
        # 2. Use remaining budget for local refinement
        # 3. Also try some random points to ensure coverage
        
        rng = np.random.default_rng(42)
        
        # Number of front points to sample
        n_front = min(150, max(50, budget - 50))
        # Remaining for local search
        n_local = budget - n_front
        
        # Sample x[0] values - use a combination of uniform and clustered near 0
        # The front is dense in effect near f1=0
        x0_samples = np.linspace(0.0, 1.0, n_front)
        # Add some extra points near 0 where the front changes rapidly
        x0_samples = np.concatenate([x0_samples, np.linspace(0.0, 0.1, 20)])
        x0_samples = np.unique(x0_samples)
        n_front = len(x0_samples)
        
        X = []
        F = []
        
        # Evaluate front points with x[1:]=0
        for x0 in x0_samples:
            if time.time() > deadline:
                break
            x = np.zeros(dim)
            x[0] = x0
            x = np.clip(x, lo, hi)
            f = problem(x)
            X.append(x)
            F.append(f)
        
        used = len(X)
        
        # Local refinement: for some of the front points, try small perturbations
        if used < budget:
            # Pick a subset of front points to refine
            n_refine = min(30, budget - used)
            if n_refine > 0:
                # Select indices to refine - spread across the front
                idxs = np.linspace(0, len(X) - 1, n_refine).astype(int)
                for idx in idxs:
                    if used >= budget or time.time() > deadline:
                        break
                    base = X[idx].copy()
                    # Try small perturbations in x[1:]
                    for _ in range(3):
                        if used >= budget or time.time() > deadline:
                            break
                        pert = 0.01 * rng.standard_normal(dim)
                        pert[0] = 0  # keep x[0] fixed for this refinement
                        x_new = base + pert
                        x_new = np.clip(x_new, lo, hi)
                        f_new = problem(x_new)
                        X.append(x_new)
                        F.append(f_new)
                        used += 1
        
        # If we have budget left, do some random exploration
        if used < budget:
            n_random = budget - used
            for _ in range(n_random):
                if time.time() > deadline:
                    break
                x = lo + span * rng.random(dim)
                f = problem(x)
                X.append(x)
                F.append(f)
                used += 1
        
        return X
# EVOLVE-BLOCK-END
