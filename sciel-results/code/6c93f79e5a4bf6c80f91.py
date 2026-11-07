"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-objective optimizer with adaptive local refinement and deadline guard."""

    def __init__(self, budget, dim, n_obj):
        self.budget, self.dim, self.n_obj = budget, dim, n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 300.0  # 300s limit with margin
        rng = np.random.default_rng(0)
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        span = hi - lo

        # Phase 1: Latin Hypercube Sampling for diverse initial population
        n_init = min(40, self.budget // 5)
        X = np.zeros((n_init, self.dim))
        for d in range(self.dim):
            perm = rng.permutation(n_init)
            X[:, d] = (perm + rng.random(n_init)) / n_init
        X = lo + X * span
        F = np.array([problem(x) for x in X])
        used = n_init

        # Archive management
        archive_X = list(X)
        archive_F = list(F)

        def is_dominated(f, archive):
            for af in archive:
                if np.all(af <= f) and np.any(af < f):
                    return True
            return False

        def update_archive(x, f):
            nonlocal archive_X, archive_F
            # Remove dominated points
            new_archive_X = []
            new_archive_F = []
            for ax, af in zip(archive_X, archive_F):
                if not (np.all(f <= af) and np.any(f < af)):
                    new_archive_X.append(ax)
                    new_archive_F.append(af)
            # Add new point if not dominated
            if not is_dominated(f, new_archive_F):
                new_archive_X.append(x)
                new_archive_F.append(f)
            archive_X = new_archive_X
            archive_F = new_archive_F

        # Phase 2: Adaptive local refinement
        step_size = 0.1
        polish_evals = 30
        main_budget = self.budget - polish_evals

        while used < main_budget:
            # Check deadline
            if time.time() > deadline - 10:
                break

            # Find non-dominated points for parent selection
            nondom_mask = np.array([not is_dominated(f, archive_F) for f in archive_F])
            if np.any(nondom_mask):
                parent_idx = rng.choice(np.where(nondom_mask)[0], size=1)[0]
                parent_x = archive_X[parent_idx]
            else:
                parent_idx = rng.integers(len(archive_X))
                parent_x = archive_X[parent_idx]

            # Adaptive mutation
            noise = step_size * span * rng.standard_normal(self.dim)
            child_x = np.clip(parent_x + noise, lo, hi)
            child_f = problem(child_x)
            used += 1

            update_archive(child_x, child_f)

            # Adaptive step size: reduce when close to convergence
            if used % 10 == 0 and len(archive_F) > 5:
                step_size *= 0.95
                step_size = max(step_size, 0.01)

        # Phase 3: Final high-precision polish pass
        if used < self.budget and time.time() < deadline - 5:
            polish_step = 0.02
            for _ in range(min(polish_evals, self.budget - used)):
                if time.time() > deadline - 2:
                    break

                # Select best archive point to refine
                if len(archive_X) > 0:
                    # Use simple score: sum of objectives
                    scores = np.array([np.sum(f) for f in archive_F])
                    best_idx = int(np.argmin(scores))
                    parent_x = archive_X[best_idx]
                else:
                    parent_x = lo + span * 0.5

                noise = polish_step * span * rng.standard_normal(self.dim)
                child_x = np.clip(parent_x + noise, lo, hi)
                child_f = problem(child_x)
                used += 1

                update_archive(child_x, child_f)

        # Return the archive points
        if archive_X:
            return np.array(archive_X)
        else:
            return np.array([lo + span * 0.5])
# EVOLVE-BLOCK-END
