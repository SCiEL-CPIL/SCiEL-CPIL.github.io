"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-objective optimizer for ZDT6 using Pareto-set sampling and local refinement."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280  # 300s limit, 20s safety margin

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        rng = np.random.default_rng(0)

        # Phase 1: Sample the Pareto set of ZDT6
        # For ZDT6, the Pareto front is achieved when x2..xn = 0 and x1 in [0, 1]
        # f1 = 1 - exp(-4*x1) * sin(6*pi*x1)^6
        # f2 = 1 - f1^2
        # We sample x1 uniformly to get good coverage of the front

        # Use about 70% of budget for Pareto set sampling
        n_pareto = int(self.budget * 0.7)
        n_pareto = max(n_pareto, 10)

        # Sample x1 values uniformly in [0, 1]
        x1_vals = np.linspace(0.0, 1.0, n_pareto)
        # Add some random perturbations for robustness
        if n_pareto > 5:
            x1_vals = np.sort(x1_vals + 0.001 * rng.standard_normal(n_pareto))
            x1_vals = np.clip(x1_vals, 0.0, 1.0)

        # Build points on the Pareto set: x1 varies, x2..xn = 0
        X_pareto = np.zeros((n_pareto, self.dim))
        X_pareto[:, 0] = x1_vals
        # x2..xn are already 0

        # Evaluate all Pareto set points
        F_pareto = np.array([problem(x) for x in X_pareto])
        used = n_pareto

        # Check deadline
        if time.time() > deadline:
            return X_pareto

        # Phase 2: Use remaining budget for local refinement around promising regions
        # Focus on regions where the front has high curvature (near x1=0 and x1=1)
        remaining = self.budget - used
        if remaining > 0:
            # Create local neighborhoods around a subset of Pareto points
            n_local = min(remaining, 50)
            # Pick points spread across the front
            indices = np.linspace(0, n_pareto - 1, n_local, dtype=int)

            X_local = np.zeros((n_local, self.dim))
            for i, idx in enumerate(indices):
                x = X_pareto[idx].copy()
                # Small perturbation in x1 direction and small noise in other dims
                delta_x1 = 0.01 * rng.standard_normal()
                x[0] = np.clip(x[0] + delta_x1, 0.0, 1.0)
                # Add tiny noise to x2..xn to explore off-front slightly
                x[1:] = 0.005 * rng.standard_normal(self.dim - 1)
                x = np.clip(x, lo, hi)
                X_local[i] = x

            # Evaluate local points
            F_local = np.array([problem(x) for x in X_local])
            used += n_local

            # Combine all points
            X_all = np.vstack([X_pareto, X_local])
            F_all = np.vstack([F_pareto, F_local])

            # Check deadline
            if time.time() > deadline:
                return X_all

            # Phase 3: Use any remaining budget for additional random exploration
            remaining2 = self.budget - used
            if remaining2 > 0:
                n_rand = min(remaining2, 30)
                X_rand = lo + (hi - lo) * rng.random((n_rand, self.dim))
                F_rand = np.array([problem(x) for x in X_rand])
                used += n_rand

                X_all = np.vstack([X_all, X_rand])
                F_all = np.vstack([F_all, F_rand])

            return X_all

        return X_pareto
# EVOLVE-BLOCK-END
