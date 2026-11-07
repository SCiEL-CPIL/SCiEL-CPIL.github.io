"""Multi-objective optimizer on RE37 using hypervolume-based differential evolution."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """Multi-Objective Differential Evolution with hypervolume-based archive."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 315.0  # Safety margin

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        dim = self.dim
        n_obj = self.n_obj
        budget = self.budget

        ref = np.array([1.1, 1.3, 1.1])

        # Check if a dominates b (a is better than b in all objectives, strictly in at least one)
        def dominates(a, b):
            return np.all(a <= b) and np.any(a < b)

        # Initialize with random points
        pop_size = 20
        X = lo + (hi - lo) * np.random.random((pop_size, dim))
        F = np.array([problem(x) for x in X])
        used = pop_size

        # Build initial archive (non-dominated points)
        archive = list(range(pop_size))
        archive_set = set(archive)
        for i in list(archive_set):
            for j in archive_set:
                if i != j and dominates(F[j], F[i]):
                    archive_set.discard(i)
                    break
        archive = list(archive_set)

        # DE parameters
        CR = 0.9  # Crossover rate
        F_de = 0.8  # Differential weight
        max_archive = 40

        while used < budget:
            # Check time
            if time.time() > deadline:
                break

            # Ensure we have enough archive members
            if len(archive) < 3:
                while len(archive) < 3 and used < budget:
                    x_new = lo + (hi - lo) * np.random.random(dim)
                    f_new = problem(x_new)
                    used += 1
                    X = np.vstack([X, [x_new]])
                    F = np.vstack([F, [f_new]])
                    # Check if non-dominated
                    is_dom = False
                    for idx in archive:
                        if dominates(F[idx], f_new):
                            is_dom = True
                            break
                    if not is_dom:
                        # Remove dominated archive members
                        new_archive = []
                        for idx in archive:
                            if not dominates(f_new, F[idx]):
                                new_archive.append(idx)
                        archive = new_archive
                        archive.append(len(F) - 1)
                if len(archive) < 3:
                    break

            # Select 3 distinct indices from archive
            a, b, c = np.random.choice(archive, size=3, replace=False)
            base = np.random.choice(archive)

            # Standard DE mutation in decision space
            x_base = X[base]
            x_a = X[a]
            x_b = X[b]

            v = x_base + F_de * (x_a - x_b)

            # Binomial crossover
            trial = x_base.copy()
            for j in range(dim):
                if np.random.random() < CR:
                    trial[j] = v[j]

            # Clip to bounds
            trial = np.clip(trial, lo, hi)

            # Evaluate
            f_trial = problem(trial)
            used += 1

            # Update archive
            dominated = False
            for idx in archive:
                if dominates(F[idx], f_trial):
                    dominated = True
                    break

            if not dominated:
                # Remove archive members dominated by trial
                new_archive = []
                for idx in archive:
                    if not dominates(f_trial, F[idx]):
                        new_archive.append(idx)
                archive = new_archive
                X = np.vstack([X, [trial]])
                F = np.vstack([F, [f_trial]])
                archive.append(len(F) - 1)

                # Prune archive if too large
                if len(archive) > max_archive:
                    # Simple pruning: remove the point with worst individual objective values
                    # or just truncate (keep first max_archive)
                    # Better: use a simple distance-based pruning
                    # For now, keep the most recently added ones (they tend to be better)
                    archive = archive[-max_archive:]

        return X
# EVOLVE-BLOCK-END
