"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-Objective Simulated Annealing with non-dominated archive for WFG7."""

    def __init__(self, budget, dim, n_obj):
        self.budget, self.dim, self.n_obj = budget, dim, n_obj

    def __call__(self, problem, bounds):
        t_start = time.time()
        deadline = t_start + 315.0  # safety margin before 330s kill

        rng = np.random.default_rng(42)
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        dim = self.dim
        n_obj = self.n_obj
        budget = self.budget
        rng_span = hi - lo

        # --- Non-dominated archive management ---
        archive_F = []  # list of objective vectors
        archive_X = []  # list of decision vectors

        def is_dominated(f, archive):
            """Check if f is dominated by any point in archive."""
            for af in archive:
                if np.all(af <= f) and np.any(af < f):
                    return True
            return False

        def remove_dominated(archive_F, archive_X, idx_to_remove):
            """Remove dominated points from archive."""
            for idx in sorted(idx_to_remove, reverse=True):
                archive_F.pop(idx)
                archive_X.pop(idx)

        def add_to_archive(f, x):
            """Add point to archive if not dominated; remove dominated points."""
            # Check if new point is dominated
            for af in archive_F:
                if np.all(af <= f) and np.any(af < f):
                    return False
            # Remove points dominated by new point
            to_remove = []
            for i, af in enumerate(archive_F):
                if np.all(f <= af) and np.any(f < af):
                    to_remove.append(i)
            remove_dominated(archive_F, archive_X, to_remove)
            archive_F.append(f)
            archive_X.append(x)
            return True

        # --- Initial population to seed archive ---
        n_init = 30
        init_X = lo + rng_span * rng.random((n_init, dim))
        used = 0
        for i in range(n_init):
            f = problem(init_X[i])
            used += 1
            add_to_archive(f, init_X[i].copy())

        # Pick initial current solution from archive
        cur_idx = rng.integers(len(archive_X))
        cur_x = archive_X[cur_idx].copy()
        cur_f = archive_F[cur_idx].copy()

        # --- SA parameters ---
        T_init = 1.0
        T_min = 1e-6
        cooling = 0.998
        T = T_init

        # Reference point for dominance-based acceptance
        ref_point = np.array([3.0, 5.0, 7.0])

        # --- Main SA loop ---
        while used < budget:
            if time.time() > deadline:
                break

            # Generate perturbation: mix of large and small steps
            if rng.random() < 0.3:
                # Large jump: restart-like
                step_size = 0.5
            elif rng.random() < 0.5:
                # Medium step
                step_size = 0.15
            else:
                # Fine local search
                step_size = 0.03

            # Perturb current solution
            new_x = cur_x + step_size * rng.standard_normal(dim)
            new_x = np.clip(new_x, lo, hi)

            # Evaluate
            new_f = problem(new_x)
            used += 1

            # Acceptance criterion
            # 1. If new point is non-dominated w.r.t. current -> accept
            if not (np.all(cur_f <= new_f) and np.any(cur_f < new_f)):
                # New is not worse than current
                accept = True
            elif not (np.all(new_f <= cur_f) and np.any(new_f < cur_f)):
                # Neither dominates the other (incomparable)
                # Use hypervolume contribution as proxy
                # Accept with probability based on temperature
                # Compute a simple scalar: distance to reference
                d_cur = np.sum(np.maximum(0, ref_point - cur_f))
                d_new = np.sum(np.maximum(0, ref_point - new_f))
                delta = d_new - d_cur
                if delta <= 0:
                    accept = True
                else:
                    accept = rng.random() < np.exp(-delta / max(T, T_min))
            else:
                # New is dominated by current
                delta = np.sum(np.maximum(0, new_f - cur_f))
                if delta <= 0:
                    accept = True
                else:
                    accept = rng.random() < np.exp(-delta / max(T, T_min))

            if accept:
                cur_x = new_x
                cur_f = new_f

            # Add to archive regardless (archive is global)
            add_to_archive(cur_f, cur_x.copy())

            # Periodically re-seed current from archive to explore different regions
            if used % 25 == 0 and len(archive_X) > 1:
                # Pick a random archive member as new starting point
                pick_idx = rng.integers(len(archive_X))
                cur_x = archive_X[pick_idx].copy()
                cur_f = archive_F[pick_idx].copy()

            # Cool down
            T = max(T * cooling, T_min)

        return cur_x
# EVOLVE-BLOCK-END
