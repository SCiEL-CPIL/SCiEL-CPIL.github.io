"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Fast multi-objective optimizer for DTLZ1."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 285
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        budget = self.budget
        dim = self.dim
        n_obj = self.n_obj
        
        rng = np.random.default_rng(5)
        
        all_X = []
        all_F = []
        used = 0
        
        # Population for evolution
        pop_size = min(30, budget // 5)
        if pop_size < 5:
            pop_size = 5
        
        # Initialize
        X = lo + (hi - lo) * rng.random((pop_size, dim))
        for x in X:
            f = problem(x)
            all_X.append(x)
            all_F.append(f)
            used += 1
        F = np.array(all_F[-pop_size:])
        
        # Evolution
        while used < budget and time.time() < deadline:
            # Find non-dominated in current population
            nd = []
            for i in range(len(F)):
                dominated = False
                for j in range(len(F)):
                    if i == j:
                        continue
                    if all(F[j][k] <= F[i][k] for k in range(n_obj)) and \
                       any(F[j][k] < F[i][k] for k in range(n_obj)):
                        dominated = True
                        break
                if not dominated:
                    nd.append(i)
            if not nd:
                nd = [0]
            
            # Generate one child
            parent_idx = nd[rng.integers(len(nd))]
            x = X[parent_idx].copy()
            mask = rng.random(dim) < 0.3
            if np.any(mask):
                x[mask] += 0.1 * (hi - lo) * rng.standard_normal(np.sum(mask))
            x = np.clip(x, lo, hi)
            f = problem(x)
            used += 1
            all_X.append(x)
            all_F.append(f)
            
            # Update population
            worst = int(np.argmax(np.sum(F, axis=1)))
            X[worst] = x
            F[worst] = f
        
        return all_X
# EVOLVE-BLOCK-END
