"""Multi-objective optimizer for RE37 using Sobol sweep + local refinement."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Sobol quasi-random sweep then local Gaussian refinement around non-dominated points."""

    def __init__(self, budget, dim, n_obj):
        self.budget, self.dim, self.n_obj = budget, dim, n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 290  # safety margin before 300s kill

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        dim = self.dim
        n_obj = self.n_obj
        budget = self.budget

        # --- Phase 1: Sobol quasi-random sweep ---
        from scipy.stats import qmc

        # Use a Sobol sequence for space-filling coverage
        sobol = qmc.Sobol(d=dim, scramble=True, seed=42)
        n_sobol = min(128, budget // 2)
        sobol_samples = sobol.random(n_sobol)
        X_sobol = lo + sobol_samples * (hi - lo)

        X = X_sobol.copy()
        F = np.array([problem(x) for x in X])
        used = len(X)

        # --- Helper: find non-dominated indices ---
        def non_dominated_indices(F):
            n = len(F)
            nd = np.ones(n, dtype=bool)
            for i in range(n):
                if not nd[i]:
                    continue
                # Check if any other point dominates i
                # j dominates i if F[j] <= F[i] in all objectives and < in at least one
                dominated = np.any(
                    np.all(F <= F[i], axis=1) & np.any(F < F[i], axis=1)
                )
                # Exclude self
                dominated_self = np.all(F[i] <= F[i], axis=1) & np.any(F[i] < F[i], axis=1)
                # If any other point dominates i, mark i as dominated
                # More efficient: check all others
                for j in range(n):
                    if j == i:
                        continue
                    if np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                        nd[i] = False
                        break
            return nd

        # --- Phase 2: Local Gaussian perturbations around non-dominated points ---
        # Budget remaining for local search
        remaining = budget - used

        # Find non-dominated points from the sweep
        nd_idx = non_dominated_indices(F)
        nd_points = X[nd_idx]
        nd_F = F[nd_idx]

        if len(nd_points) == 0:
            nd_points = X
            nd_F = F

        # Local search: perturb each non-dominated point
        # Allocate budget: give more to points that seem more promising
        # Use a simple strategy: cycle through non-dominated points with decreasing step size

        # Adaptive step sizes: start large, shrink
        base_step = 0.15 * (hi - lo)
        shrink_factor = 0.7

        local_budget = remaining
        step = base_step.copy()

        # Sort non-dominated points by sum of objectives (lower is better) to prioritize
        order = np.argsort(np.sum(nd_F, axis=1))
        nd_points = nd_points[order]
        nd_F = nd_F[order]

        # For each non-dominated point, do local perturbations
        # Use a pool of current best non-dominated set
        all_X = X.copy()
        all_F = F.copy()
        all_used = used

        # Keep a record of all evaluated points
        # We'll maintain a growing set and always perturb from the current non-dominated front

        while all_used < budget:
            if time.time() > deadline:
                break

            # Recompute non-dominated set from all evaluated points
            nd_mask = non_dominated_indices(all_F)
            nd_X = all_X[nd_mask]
            nd_F_curr = all_F[nd_mask]

            if len(nd_X) == 0:
                nd_X = all_X
                nd_F_curr = all_F

            # Pick a parent from non-dominated set (round-robin with some randomness)
            # Use deterministic cycling to ensure coverage
            parent_idx = all_used % len(nd_X)
            parent = nd_X[parent_idx].copy()

            # Generate a perturbation with current step size
            perturbation = np.random.standard_normal(dim) * step
            child = parent + perturbation
            child = np.clip(child, lo, hi)

            fc = problem(child)
            all_used += 1

            # Add to population
            all_X = np.vstack([all_X, child.reshape(1, -1)])
            all_F = np.vstack([all_F, fc.reshape(1, -1)])

            # Periodically shrink step size
            if all_used % 20 == 0:
                step *= shrink_factor

        return all_X
# EVOLVE-BLOCK-END
