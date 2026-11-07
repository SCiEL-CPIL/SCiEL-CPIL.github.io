"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-objective optimizer for ZDT4 using structured exploration and local search."""

    def __init__(self, budget, dim, n_obj):
        self.budget, self.dim, self.n_obj = budget, dim, n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280  # 300s limit, 20s safety margin

        rng = np.random.default_rng(42)
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        dim = self.dim
        budget = self.budget

        # Phase 1: Structured random sampling to cover the space
        # ZDT4: x1 controls f1, others control f2
        # Focus on x1 in range that gives good f1 values
        n_init = min(40, budget // 5)
        X = np.zeros((n_init, dim))
        F = np.zeros((n_init, self.n_obj))

        for i in range(n_init):
            x = lo + (hi - lo) * rng.random(dim)
            # Bias x1 toward lower values for better f1
            if rng.random() < 0.7:
                x[0] = lo[0] + (hi[0] - lo[0]) * (0.1 * rng.random())
            X[i] = x
            F[i] = problem(x)

        used = n_init

        # Phase 2: Local search around best points
        # Find non-dominated points
        def is_dominated(f, F_arr):
            for j in range(len(F_arr)):
                if np.all(F_arr[j] <= f) and np.any(F_arr[j] < f):
                    return True
            return False

        # Keep track of all evaluated points
        all_X = [X.copy()]
        all_F = [F.copy()]

        while used < budget:
            # Check time
            if time.time() > deadline:
                break

            # Select a parent: prefer non-dominated, diverse
            # Compute dominance
            n_pts = len(all_F[0])
            non_dom_mask = np.zeros(n_pts, dtype=bool)
            for i in range(n_pts):
                non_dom_mask[i] = not is_dominated(all_F[0][i], all_F[0])

            if not np.any(non_dom_mask):
                non_dom_mask = np.ones(n_pts, dtype=bool)

            # Pick a parent from non-dominated set
            parent_idx = rng.choice(np.where(non_dom_mask)[0])
            parent = all_X[0][parent_idx].copy()

            # Local search: small perturbation
            step_size = 0.05 * (hi - lo)
            child = parent + step_size * rng.standard_normal(dim)
            child = np.clip(child, lo, hi)

            # Also try targeted moves
            r = rng.random()
            if r < 0.3:
                # Move x1 toward better f1
                child[0] = lo[0] + (hi[0] - lo[0]) * rng.random() * 0.2
            elif r < 0.6:
                # Perturb a random subset of variables
                n_pert = rng.integers(1, max(2, dim // 2))
                idx = rng.choice(dim, size=n_pert, replace=False)
                child[idx] = parent[idx] + 0.1 * (hi[idx] - lo[idx]) * rng.standard_normal(n_pert)
                child[idx] = np.clip(child[idx], lo[idx], hi[idx])
            else:
                # Larger jump
                child = parent + 0.2 * (hi - lo) * rng.standard_normal(dim)
                child = np.clip(child, lo, hi)

            fc = problem(child)
            used += 1

            # Add to archive
            all_X[0] = np.vstack([all_X[0], child.reshape(1, -1)])
            all_F[0] = np.vstack([all_F[0], fc.reshape(1, -1)])

            # Prune dominated points to keep archive manageable
            if len(all_F[0]) > 100:
                # Keep non-dominated and some random
                new_mask = np.zeros(len(all_F[0]), dtype=bool)
                for i in range(len(all_F[0])):
                    new_mask[i] = not is_dominated(all_F[0][i], all_F[0])
                if np.sum(new_mask) < 50:
                    # Add some random dominated points for diversity
                    dom_idx = np.where(~new_mask)[0]
                    extra = rng.choice(dom_idx, size=min(20, len(dom_idx)), replace=False)
                    new_mask[extra] = True
                all_X[0] = all_X[0][new_mask]
                all_F[0] = all_F[0][new_mask]

        return all_X[0]
# EVOLVE-BLOCK-END
