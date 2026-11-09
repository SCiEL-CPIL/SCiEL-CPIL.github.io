# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start = time.time()
        deadline = start + 290  # safety margin before 330s kill
        rng = np.random.default_rng(0)
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        dim = self.dim

        # ZDT4: x1 is on [-5,5], x2..x10 on [-5,5]
        # f1 = x1, f2 = g(x2..x10)*(1 + sqrt(x1/g))
        # Pareto front: g=0, f1 in [0,1] (but x1 in [-5,5], so f1=x1)
        # Actually for ZDT4, the PF is f1=x1 for x1 in [0,1] when g=0
        # But x1 can be negative; f1=x1 always. The PF is typically x1 in [0,1].

        # Strategy: 
        # 1. Find good x2..x10 that minimize g
        # 2. Sample x1 across [0,1] (the PF region)
        # 3. Use remaining budget to refine

        def evaluate(x):
            return problem(x)

        # Phase 1: Initialize with diverse x1 and random x2..x10
        # Use a small population
        pop_size = 30
        X = np.zeros((pop_size, dim))
        # x1: spread across [0, 1] plus some beyond
        x1_vals = np.linspace(0, 1, pop_size // 2)
        x1_vals = np.concatenate([x1_vals, rng.uniform(-1, 2, pop_size - len(x1_vals))])
        X[:, 0] = x1_vals
        X[:, 1:] = lo[1:] + (hi[1:] - lo[1:]) * rng.random((pop_size, dim - 1))
        
        F = np.array([evaluate(x) for x in X])
        used = pop_size

        # Phase 2: Local search on x2..x10 to minimize g
        # For each point, try to improve x2..x10
        # Use a simple hill-climbing / local perturbation
        
        # First, identify the best g values (second objective related)
        # g(x2..x10) = 10*(dim-1) + sum(x_i^2 - 10*cos(4*pi*x_i)) for i=2..10
        # We want to minimize this.
        
        # For each point, do local search on x2..x10
        for i in range(pop_size):
            if time.time() > deadline - 5:
                break
            x = X[i].copy()
            f = F[i].copy()
            # Try to improve x2..x10
            for _ in range(5):
                if time.time() > deadline - 5:
                    break
                if used >= self.budget:
                    break
                # Perturb x2..x10
                pert = x.copy()
                scale = 0.1
                pert[1:] += scale * (hi[1:] - lo[1:]) * rng.standard_normal(dim - 1)
                pert[1:] = np.clip(pert[1:], lo[1:], hi[1:])
                fp = evaluate(pert)
                used += 1
                # Keep if it improves f2 (since f1 is fixed for this x1)
                if fp[1] < f[1]:
                    x = pert
                    f = fp
                    X[i] = x
                    F[i] = f
        
        # Phase 3: Use remaining budget to refine the front
        # Sample x1 uniformly in [0,1] and use best x2..x10 found
        # Find the best x2..x10 (lowest f2)
        if used < self.budget:
            # Get the best g value (lowest f2 among points with x1 in [0,1])
            mask = (X[:, 0] >= 0) & (X[:, 0] <= 1)
            if np.any(mask):
                best_idx = np.argmin(F[mask, 1])
                best_x2_10 = X[mask][best_idx, 1:]
                best_f2 = F[mask, 1][best_idx]
            else:
                best_idx = np.argmin(F[:, 1])
                best_x2_10 = X[best_idx, 1:]
                best_f2 = F[best_idx, 1]
            
            # Now sample x1 across [0,1] with the best x2..x10
            n_samples = min(self.budget - used, 50)
            x1_grid = np.linspace(0, 1, n_samples)
            for x1 in x1_grid:
                if used >= self.budget:
                    break
                if time.time() > deadline - 5:
                    break
                x = np.zeros(dim)
                x[0] = x1
                x[1:] = best_x2_10
                fx = evaluate(x)
                used += 1
                # Replace worst point
                # Simple: replace the point with highest f2
                worst = np.argmax(F[:, 1])
                X[worst] = x
                F[worst] = fx
        
        # Phase 4: Final polish - local search on x2..x10 for points near PF
        if used < self.budget and time.time() < deadline - 10:
            # For each point, try to further reduce f2
            for i in range(pop_size):
                if used >= self.budget:
                    break
                if time.time() > deadline - 5:
                    break
                x = X[i].copy()
                f = F[i].copy()
                # Small perturbations on x2..x10
                for _ in range(3):
                    if used >= self.budget:
                        break
                    if time.time() > deadline - 5:
                        break
                    pert = x.copy()
                    scale = 0.05
                    pert[1:] += scale * (hi[1:] - lo[1:]) * rng.standard_normal(dim - 1)
                    pert[1:] = np.clip(pert[1:], lo[1:], hi[1:])
                    fp = evaluate(pert)
                    used += 1
                    if fp[1] < f[1]:
                        x = pert
                        f = fp
                        X[i] = x
                        F[i] = f
        
        return X
# EVOLVE-BLOCK-END
