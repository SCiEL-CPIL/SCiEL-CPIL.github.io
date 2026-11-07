"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Latin Hypercube initialized archive-based multi-objective optimizer for WFG9."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 325.0  # safety margin before 330s kill

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        dim = self.dim
        n_obj = self.n_obj
        budget = self.budget

        rng = np.random

        # Latin Hypercube Sampling for initial population
        # This ensures uniform coverage of the search space
        pop_size = min(20, budget // 4)
        if pop_size < 10:
            pop_size = max(5, pop_size)

        # Generate LHS points
        X = np.zeros((pop_size, dim))
        for d in range(dim):
            # Create strata and randomize within each stratum
            perm = rng.permutation(pop_size)
            for i in range(pop_size):
                # Random point within the stratum
                X[perm[i], d] = lo[d] + (hi[d] - lo[d]) * (i + rng.random()) / pop_size

        F = np.array([problem(x) for x in X])
        used = pop_size

        # Archive of non-dominated points
        archive_X = list(X)
        archive_F = list(F)

        def is_dominated(f, archive):
            """Check if f is dominated by any point in archive."""
            for af in archive:
                if np.all(af <= f) and np.any(af < f):
                    return True
            return False

        def dominates(f, archive):
            """Return indices of archive points dominated by f."""
            dominated = []
            for i, af in enumerate(archive):
                if np.all(f <= af) and np.any(f < af):
                    dominated.append(i)
            return dominated

        # Main evolution loop
        while used < budget:
            if time.time() > deadline - 5.0:
                break

            # Find non-dominated points in current archive
            nd_mask = []
            for i, f in enumerate(archive_F):
                dominated = False
                for j, af in enumerate(archive_F):
                    if i == j:
                        continue
                    if np.all(af <= f) and np.any(af < f):
                        dominated = True
                        break
                nd_mask.append(not dominated)

            nd_indices = [i for i, m in enumerate(nd_mask) if m]

            # Select parent from non-dominated set
            if nd_indices:
                parent_idx = nd_indices[rng.randint(len(nd_indices))]
                parent = archive_X[parent_idx]
            else:
                parent = archive_X[rng.randint(len(archive_X))]

            # Adaptive mutation step size
            progress = used / budget
            base_step = 0.15
            min_step = 0.02
            step_size = base_step * (1.0 - progress * 0.8) + min_step

            # Generate child with mutation
            if rng.random() < 0.8:
                # Gaussian mutation
                child = parent + step_size * (hi - lo) * rng.standard_normal(dim)
            else:
                # Larger exploration jump
                child = parent + 0.3 * (hi - lo) * rng.standard_normal(dim)

            child = np.clip(child, lo, hi)

            # Evaluate child
            fc = problem(child)
            used += 1

            # Update archive
            # Remove points dominated by child
            dominated_indices = dominates(fc, archive_F)
            for idx in sorted(dominated_indices, reverse=True):
                archive_X.pop(idx)
                archive_F.pop(idx)

            # Add child if not dominated
            if not is_dominated(fc, archive_F):
                archive_X.append(child)
                archive_F.append(fc)

            # If archive gets too large, prune using hypervolume contribution
            if len(archive_F) > 50:
                # Simple pruning: remove points with least hypervolume contribution
                # Use a greedy approach to keep diverse points
                keep = []
                keep_F = []
                for i in range(len(archive_F)):
                    if len(keep) >= 50:
                        break
                    # Check if this point is dominated by any kept point
                    dominated_by_kept = False
                    for kf in keep_F:
                        if np.all(kf <= archive_F[i]) and np.any(kf < archive_F[i]):
                            dominated_by_kept = True
                            break
                    if not dominated_by_kept:
                        keep.append(archive_X[i])
                        keep_F.append(archive_F[i])

                archive_X = keep
                archive_F = keep_F

        return archive_X
# EVOLVE-BLOCK-END
