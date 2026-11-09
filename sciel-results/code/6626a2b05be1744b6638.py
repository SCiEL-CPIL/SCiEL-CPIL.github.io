"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


class Optimizer:
    """Differential Evolution with external archive for ZDT6 multi-objective optimization."""

    def __init__(self, budget, dim, n_obj):
        self.budget, self.dim, self.n_obj = budget, dim, n_obj

    def __call__(self, problem, bounds):
        rng = np.random.default_rng(42)
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        dim = self.dim
        budget = self.budget

        # Population size for DE
        pop_size = min(50, budget // 4)
        if pop_size < 4:
            pop_size = 4

        # Initialize population
        X = lo + (hi - lo) * rng.random((pop_size, dim))
        F = np.array([problem(x) for x in X])
        used = pop_size

        # External archive for non-dominated solutions
        archive = []
        archive_F = []

        def is_dominated(f, archive_F):
            """Check if f is dominated by any point in archive_F."""
            for af in archive_F:
                if np.all(af <= f) and np.any(af < f):
                    return True
            return False

        def update_archive(f, archive, archive_F):
            """Update archive with new point if it is non-dominated."""
            if is_dominated(f, archive_F):
                return
            # Remove dominated points from archive
            new_archive = []
            new_archive_F = []
            for i, af in enumerate(archive_F):
                if not (np.all(f <= af) and np.any(f < af)):
                    new_archive.append(archive[i])
                    new_archive_F.append(af)
            new_archive.append(f.copy())
            new_archive_F.append(f.copy())
            return new_archive, new_archive_F

        # Initialize archive with initial non-dominated points
        for i in range(pop_size):
            archive, archive_F = update_archive(F[i], archive, archive_F)

        # DE parameters
        F_factor = 0.8  # Differential weight
        CR = 0.9        # Crossover rate

        while used < budget:
            # Select three distinct random indices
            idx = rng.integers(0, pop_size)
            while True:
                i1 = rng.integers(0, pop_size)
                if i1 != idx:
                    break
            while True:
                i2 = rng.integers(0, pop_size)
                if i2 != idx and i2 != i1:
                    break
            while True:
                i3 = rng.integers(0, pop_size)
                if i3 != idx and i3 != i1 and i3 != i2:
                    break

            # Create trial vector using DE/rand/1
            v = X[i1] + F_factor * (X[i2] - X[i3])
            v = np.clip(v, lo, hi)

            # Binomial crossover
            j_rand = rng.integers(0, dim)
            cross_mask = rng.random(dim) < CR
            cross_mask[j_rand] = True
            trial = np.where(cross_mask, v, X[idx])

            # Evaluate trial
            F_trial = problem(trial)
            used += 1

            # Update archive
            archive, archive_F = update_archive(F_trial, archive, archive_F)

            # Selection: keep better of parent and trial (non-dominated selection)
            # If trial dominates parent, replace
            if np.all(F_trial <= F[idx]) and np.any(F_trial < F[idx]):
                X[idx] = trial
                F[idx] = F_trial
            # If parent dominates trial, keep parent
            # If non-dominated, keep both (but we only have one slot, so use archive)
            # Simple approach: if non-dominated, keep trial with some probability
            elif not (np.all(F[idx] <= F_trial) and np.any(F[idx] < F_trial)):
                # Non-dominated: keep trial with 50% probability to maintain diversity
                if rng.random() < 0.5:
                    X[idx] = trial
                    F[idx] = F_trial

        return X
# EVOLVE-BLOCK-END
