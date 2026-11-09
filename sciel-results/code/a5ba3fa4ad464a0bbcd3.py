"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Simple evolutionary multi-objective search with adaptive mutation."""

    def __init__(self, budget, dim, n_obj):
        self.budget, self.dim, self.n_obj = budget, dim, n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        DEADLINE = 310  # safety margin before 330s kill

        rng = np.random.default_rng(0)
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        dim = self.dim
        budget = self.budget

        # Initial population
        pop_size = min(20, max(10, budget // 10))
        X = lo + (hi - lo) * rng.random((pop_size, dim))
        F = np.array([problem(x) for x in X])
        used = pop_size

        # Archive of best non-dominated points (in decision space)
        archive_X = [X[i].copy() for i in range(pop_size)]
        archive_F = [F[i].copy() for i in range(pop_size)]

        def is_non_dominated(f, archive_F_list):
            """Check if f is not dominated by any point in archive."""
            for af in archive_F_list:
                if np.all(af <= f) and np.any(af < f):
                    return False
            return True

        def update_archive(x, f):
            """Update archive with new point if non-dominated."""
            if is_non_dominated(f, archive_F):
                # Remove dominated points from archive
                new_archive_X = []
                new_archive_F = []
                for ax, af in zip(archive_X, archive_F):
                    if not (np.all(f <= af) and np.any(f < af)):
                        new_archive_X.append(ax)
                        new_archive_F.append(af)
                new_archive_X.append(x.copy())
                new_archive_F.append(f.copy())
                return new_archive_X, new_archive_F
            return archive_X, archive_F

        # Adaptive mutation scale
        base_scale = 0.15
        min_scale = 0.02

        while used < budget:
            if time.time() - start_time > DEADLINE:
                break

            # Find non-dominated points in current population
            non_dom_mask = np.zeros(len(F), dtype=bool)
            for i in range(len(F)):
                dominated = False
                for j in range(len(F)):
                    if i != j and np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                        dominated = True
                        break
                if not dominated:
                    non_dom_mask[i] = True

            # Select parent from non-dominated points (or all if none)
            if np.any(non_dom_mask):
                parent_idx = rng.integers(len(non_dom_mask))
                parent_idx = np.where(non_dom_mask)[0][parent_idx]
            else:
                # Pick the point with lowest sum of objectives
                parent_idx = int(np.argmin(np.sum(F, axis=1)))

            # Adaptive mutation scale: shrink as we use more budget
            progress = used / budget
            scale = base_scale * (1 - progress * 0.7) + min_scale * progress

            # Generate child with Gaussian mutation
            child = X[parent_idx].copy()
            # Randomly select which dimensions to mutate
            n_mutate = rng.integers(1, dim + 1)
            mut_dims = rng.choice(dim, size=n_mutate, replace=False)
            for d in mut_dims:
                child[d] += scale * (hi[d] - lo[d]) * rng.standard_normal()
            child = np.clip(child, lo, hi)

            fc = problem(child)
            used += 1

            # Update archive
            archive_X, archive_F = update_archive(child, fc)

            # Replace worst point in population
            worst = int(np.argmax(np.sum(F, axis=1)))
            X[worst] = child
            F[worst] = fc

        return X
# EVOLVE-BLOCK-END
