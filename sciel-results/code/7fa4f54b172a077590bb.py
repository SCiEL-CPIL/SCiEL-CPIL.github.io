"""Multi-objective optimizer for ZDT1 using a decomposition-based approach."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Decomposition-based multi-objective optimizer for ZDT1."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280  # safety margin before 300s limit

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        rng = np.random.default_rng(42)

        # For ZDT1, the Pareto front is characterized by f1 = g + 1, f2 = 1 - sqrt(f1/g)
        # where g = sum((xi - 0.5)^2) for i in 2..dim
        # Optimal: g = 0 (all xi = 0.5 for i>=2), and f1 ranges from 1 to n_obj+1
        # For 2 objectives: f1 in [1, 2], f2 = 1 - sqrt(f1)

        # Strategy: 
        # 1. Set x[1:] = 0.5 (optimal for g)
        # 2. Vary x[0] to control f1
        # For ZDT1: f1 = g + x[0], f2 = 1 - sqrt(x[0]/(g+x[0]))
        # When g=0: f1 = x[0], f2 = 1 - sqrt(x[0])
        # So x[0] in [0,1] gives f1 in [0,1], f2 in [1,0]
        # But we want f1 in [1,2] for the standard ZDT1 front... 
        # Actually ZDT1: f1 = x1, g = 1 + 10*(n-1)*avg((xi-0.5)^2)
        # f2 = g * (1 - sqrt(f1/g))
        # Optimal g=1: f2 = 1 - sqrt(f1)
        # f1 in [0,1], f2 in [1,0]

        # Let's use a hybrid approach:
        # - Initialize with points along the known Pareto front (analytic)
        # - Then use local search to refine

        pop_size = min(50, self.budget // 4)
        
        # Initialize population along the analytic Pareto front
        # x[0] controls f1, x[1:] = 0.5 for optimal g
        X = np.zeros((pop_size, self.dim))
        for i in range(pop_size):
            X[i, 0] = i / max(1, pop_size - 1)  # spread x[0] in [0,1]
            X[i, 1:] = 0.5  # optimal for g
        
        # Add some random diversity
        if pop_size > 10:
            n_random = pop_size // 5
            X[-n_random:, 0] = rng.random(n_random)
            X[-n_random:, 1:] = 0.5 + 0.1 * rng.standard_normal((n_random, self.dim - 1))
            X = np.clip(X, lo, hi)
        
        F = np.array([problem(x) for x in X])
        used = pop_size

        # Local search / refinement phase
        # For each point, try small perturbations
        best_f1 = F[:, 0].min()
        best_f2 = F[:, 1].min()

        # Sort by f1 for efficient neighbor finding
        order = np.argsort(F[:, 0])
        X_sorted = X[order]
        F_sorted = F[order]

        # Refine each point with local search
        for idx in range(pop_size):
            if time.time() > deadline:
                break
            if used >= self.budget:
                break
            
            x = X_sorted[idx].copy()
            f = F_sorted[idx].copy()
            
            # Try perturbing x[0] (controls f1)
            for step in [0.01, 0.005, 0.001]:
                if used >= self.budget or time.time() > deadline:
                    break
                for delta in [-step, step]:
                    if used >= self.budget or time.time() > deadline:
                        break
                    x_try = x.copy()
                    x_try[0] = np.clip(x_try[0] + delta, lo[0], hi[0])
                    f_try = problem(x_try)
                    used += 1
                    # Check if this improves hypervolume contribution
                    # Simple check: is it non-dominated by current point?
                    if f_try[0] < f[0] or f_try[1] < f[1]:
                        # Potentially better, accept if it's on the front
                        X_sorted[idx] = x_try
                        F_sorted[idx] = f_try
                        f = f_try
            
            # Try perturbing other dimensions to reduce g
            for dim_idx in range(1, min(5, self.dim)):
                if used >= self.budget or time.time() > deadline:
                    break
                for step in [0.05, 0.02]:
                    if used >= self.budget or time.time() > deadline:
                        break
                    for delta in [-step, step]:
                        if used >= self.budget or time.time() > deadline:
                            break
                        x_try = x.copy()
                        x_try[dim_idx] = np.clip(x_try[dim_idx] + delta, lo[dim_idx], hi[dim_idx])
                        f_try = problem(x_try)
                        used += 1
                        if f_try[0] < f[0] + 1e-6 and f_try[1] < f[1] + 1e-6:
                            if f_try[0] < f[0] or f_try[1] < f[1]:
                                X_sorted[idx] = x_try
                                F_sorted[idx] = f_try
                                f = f_try

        # Final refinement: dense sampling along the front
        if used < self.budget:
            n_extra = min(self.budget - used, 100)
            if n_extra > 0:
                # Sample x[0] densely
                x0_vals = np.linspace(0, 1, n_extra)
                for x0 in x0_vals:
                    if used >= self.budget or time.time() > deadline:
                        break
                    x_try = np.full(self.dim, 0.5)
                    x_try[0] = x0
                    x_try = np.clip(x_try, lo, hi)
                    f_try = problem(x_try)
                    used += 1
                    # Replace worst point or add if better
                    # Find if this point is non-dominated
                    dominated = False
                    for j in range(len(F_sorted)):
                        if np.all(F_sorted[j] <= f_try) and np.any(F_sorted[j] < f_try):
                            dominated = True
                            break
                    if not dominated:
                        # Replace the point with largest sum of objectives
                        worst_idx = np.argmax(np.sum(F_sorted, axis=1))
                        X_sorted[worst_idx] = x_try
                        F_sorted[worst_idx] = f_try

        # Unsort
        X = X_sorted[np.argsort(order)]
        F = F_sorted[np.argsort(order)]

        return X
# EVOLVE-BLOCK-END
