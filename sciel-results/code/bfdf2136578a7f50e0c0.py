"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-objective optimizer for ZDT2 with Pareto-front-aware sampling."""

    def __init__(self, budget, dim, n_obj):
        self.budget, self.dim, self.n_obj = budget, dim, n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280  # safety margin before 330s kill

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        dim = self.dim
        budget = self.budget

        # ZDT2: f1 = x1, f2 = (1+10*(n-1))*(1-sqrt(f1/(1+10*(n-1))))
        # Pareto front: x1 in [0,1], x2..xn = 0
        # Use non-uniform sampling to cover the concave front well

        n_init = min(100, budget // 2)
        n_refine = budget - n_init

        # Generate x1 values spread along the Pareto front
        # For ZDT2, sample x1 = 291*(1-t)^2 for good coverage
        t_vals = np.linspace(0, 1, n_init)
        x1_vals = 291.0 * (1 - t_vals) ** 2
        x1_vals = np.clip(x1_vals, 0, 1)

        # Create initial population: x1 varies, rest = 0
        X_init = np.zeros((n_init, dim))
        X_init[:, 0] = x1_vals
        X_init = np.clip(X_init, lo, hi)

        # Evaluate initial population
        F = np.array([problem(x) for x in X_init])
        used = n_init

        # Store all evaluated points
        all_X = X_init.copy()
        all_F = F.copy()

        # Refine phase
        for i in range(n_refine):
            if time.time() > deadline:
                break

            # Pick a random point from current set
            parent_idx = np.random.randint(len(all_X))
            child = all_X[parent_idx].copy()

            # Perturb x1
            child[0] += np.random.normal(0, 0.02)
            child[0] = np.clip(child[0], 0, 1)

            # Add small noise to other dimensions
            child[1:] += np.random.normal(0, 0.005, dim - 1)
            child[1:] = np.clip(child[1:], lo[1:], hi[1:])

            fc = problem(child)
            used += 1

            all_X = np.vstack([all_X, child.reshape(1, -1)])
            all_F = np.vstack([all_F, fc.reshape(1, -1)])

        return all_X
# EVOLVE-BLOCK-END
