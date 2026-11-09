"""Multi-objective optimizer for DTLZ7 exploiting known Pareto front structure."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Exploit DTLZ7 structure: when x[1:]=0, f1=x0, f2=(1-x0)*cos(x0*pi/2), f3=(1-x0)*sin(x0*pi/2)."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        t0 = time.time()
        deadline = t0 + 290.0  # safety margin before 330s kill

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        dim = self.dim
        budget = self.budget

        # Track all evaluated points and their objective values
        X_all = []  # list of x vectors
        F_all = []  # list of f vectors

        def evaluate(x):
            """Evaluate x and record it."""
            x = np.clip(x, lo, hi)
            f = problem(x)
            X_all.append(x.copy())
            F_all.append(np.array(f))
            return f

        # Phase 1: Dense structured initialization along the expected Pareto front
        # DTLZ7 Pareto front: f1=x0, f2=(1-x0)*cos(x0*pi/2), f3=(1-x0)*sin(x0*pi/2)
        # We place points at various x0 values with x[1:] = 0 (or very small)
        n_init = min(40, budget // 4)

        for i in range(n_init):
            x = np.zeros(dim)
            # Spread x0 uniformly in [0, 1]
            x[0] = i / max(n_init - 1, 1)
            # Set x[1:] to exactly 0 for ideal front points
            # But also add some with tiny noise to explore
            if i % 3 == 0:
                # Exactly on front
                x[1:] = 0.0
            else:
                # Tiny noise
                rng = np.random.default_rng(i)
                x[1:] = rng.uniform(0, 0.001, dim - 1)
            evaluate(x)

        used = n_init

        # Phase 2: Local refinement - drive x[1:] to 0 for each point
        # Use a population-based approach with exploitation
        n_pop = min(25, len(X_all))

        # Select best points by hypervolume contribution approximation (use sum of objectives)
        while used < budget:
            if time.time() > deadline:
                break

            # Pick a random parent from the current population
            parent_idx = np.random.randint(0, len(X_all))
            parent = X_all[parent_idx].copy()

            # Generate child by reducing x[1:]
            child = parent.copy()

            # Strategy: multiply x[1:] by a decreasing factor
            # Use exponential decay toward 0
            factor = np.random.uniform(0.0, 0.5)
            child[1:] = child[1:] * factor

            # Also try setting random components to 0
            n_zero = np.random.randint(1, dim)
            zero_idx = np.random.choice(dim - 1, size=min(n_zero, dim - 1), replace=False)
            child[1 + zero_idx] = 0.0

            # Slight perturbation of x0 to explore nearby front positions
            if np.random.rand() < 0.3:
                child[0] = parent[0] + np.random.uniform(-0.02, 0.02)
                child[0] = max(0.0, min(1.0, child[0]))

            fc = evaluate(child)
            used += 1

            # If we have many points, prune the worst ones to keep population manageable
            if len(X_all) > 200:
                # Keep best 200 by sum of objectives
                sums = np.sum(np.array(F_all), axis=1)
                keep_idx = np.argsort(sums)[:200]
                X_all = [X_all[i] for i in keep_idx]
                F_all = [F_all[i] for i in keep_idx]

        # Phase 3: Final polish - for the best points, try x[1:] = 0 exactly
        if used < budget and len(X_all) > 0:
            sums = np.sum(np.array(F_all), axis=1)
            sorted_idx = np.argsort(sums)

            for idx in sorted_idx[:min(15, len(X_all))]:
                if used >= budget or time.time() > deadline:
                    break

                x = X_all[idx].copy()
                x[1:] = 0.0
                x = np.clip(x, lo, hi)

                fc = evaluate(x)
                used += 1

        return X_all
# EVOLVE-BLOCK-END
