# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-objective optimizer for DTLZ4 with adaptive mutation and non-dominated selection."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 290.0  # 10s safety margin before 300s limit

        rng = np.random.default_rng(42)
        lo = np.array([b[0] for b in bounds], dtype=np.float64)
        hi = np.array([b[1] for b in bounds], dtype=np.float64)
        dim = self.dim
        n_obj = self.n_obj
        budget = self.budget

        # Initialize with a larger population for better initial coverage
        pop_size = min(50, max(20, budget // 4))
        X = lo + (hi - lo) * rng.random((pop_size, dim))
        F = np.array([problem(x) for x in X], dtype=np.float64)
        used = pop_size

        # Track best solutions found
        archive_X = [X[i] for i in range(pop_size)]
        archive_F = [F[i] for i in range(pop_size)]

        def dominates(a, b):
            """Return True if a dominates b (a <= b in all objectives and a < b in at least one)."""
            return np.all(a <= b) and np.any(a < b)

        def is_dominated(f, F_set):
            """Check if f is dominated by any point in F_set."""
            for g in F_set:
                if dominates(g, f):
                    return True
            return False

        def update_archive(x, f, archive_X, archive_F):
            """Add x to archive if it's not dominated; remove dominated points."""
            if is_dominated(f, archive_F):
                return archive_X, archive_F
            # Remove points dominated by f
            new_X = []
            new_F = []
            for i in range(len(archive_X)):
                if not dominates(f, archive_F[i]):
                    new_X.append(archive_X[i])
                    new_F.append(archive_F[i])
            new_X.append(x)
            new_F.append(f)
            return new_X, new_F

        # Adaptive mutation scale
        def get_mutation_scale(used_frac):
            """Start with broader exploration, narrow to exploitation."""
            return 0.3 - 0.25 * used_frac

        while used < budget:
            # Check time limit
            if time.time() > deadline:
                break

            # Find non-dominated points in current population
            non_dominated_mask = np.ones(pop_size, dtype=bool)
            for i in range(pop_size):
                for j in range(pop_size):
                    if i != j and dominates(F[j], F[i]):
                        non_dominated_mask[i] = False
                        break

            # Select parents from non-dominated set
            nd_indices = np.where(non_dominated_mask)[0]
            if len(nd_indices) == 0:
                nd_indices = np.arange(pop_size)
            parent_idx = nd_indices[rng.integers(len(nd_indices))]
            parent = X[parent_idx]

            # Adaptive mutation
            scale = get_mutation_scale(used / budget)
            mutation = scale * (hi - lo) * rng.standard_normal(dim)
            child = parent + mutation
            child = np.clip(child, lo, hi)

            # Evaluate
            fc = np.array(problem(child), dtype=np.float64)
            used += 1

            # Update archive
            archive_X, archive_F = update_archive(child, fc, archive_X, archive_F)

            # Replace worst point in population (by sum of objectives)
            worst_idx = int(np.argmax(np.sum(F, axis=1)))
            X[worst_idx] = child
            F[worst_idx] = fc

            # Occasionally replace a random point for diversity
            if rng.random() < 0.3:
                rand_idx = rng.integers(pop_size)
                X[rand_idx] = child
                F[rand_idx] = fc

        return X
# EVOLVE-BLOCK-END
