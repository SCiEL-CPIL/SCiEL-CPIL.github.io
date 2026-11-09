"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-objective search combining random initialization, local refinement, and archive-based exploration."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280.0  # Safety margin before 300s limit

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        span = hi - lo

        # Archive of non-dominated solutions
        archive_X = []
        archive_F = []

        def is_dominated(f, archive_F_list):
            """Check if f is dominated by any point in archive_F_list."""
            for af in archive_F_list:
                if np.all(af <= f) and np.any(af < f):
                    return True
            return False

        def update_archive(x, f):
            """Add (x, f) to archive if non-dominated, remove dominated points."""
            non_local = True
            for i, af in enumerate(archive_F):
                if np.all(af <= f) and np.any(af < f):
                    non_local = False
                    break
            if not non_local:
                return

            # Remove points dominated by f
            new_X = []
            new_F = []
            for i in range(len(archive_X)):
                if not (np.all(f <= archive_F[i]) and np.any(f < archive_F[i])):
                    new_X.append(archive_X[i])
                    new_F.append(archive_F[i])

            new_X.append(x)
            new_F.append(f)
            archive_X[:] = new_X
            archive_F[:] = new_F

        def random_point():
            return lo + span * np.random.random(self.dim)

        def perturb(x, scale):
            x_new = x + scale * span * np.random.standard_normal(self.dim)
            return np.clip(x_new, lo, hi)

        # Phase 1: Random initialization with moderate population
        init_pop = min(30, self.budget // 5)
        remaining = self.budget

        for _ in range(init_pop):
            if remaining <= 0:
                break
            if time.time() > deadline:
                break
            x = random_point()
            f = np.array(problem(x))
            remaining -= 1
            if np.all(f < [3.0, 5.0, 7.0]):  # Better than reference
                update_archive(x, f)

        # Phase 2: Local refinement around archive points
        # Use remaining budget for local searches
        local_scale = 0.05
        while remaining > 0:
            if time.time() > deadline:
                break
            if not archive_X:
                # If archive is empty, do random search
                x = random_point()
                f = np.array(problem(x))
                remaining -= 1
                if np.all(f < [3.0, 5.0, 7.0]):
                    update_archive(x, f)
                continue

            # Pick a random archive point to refine
            idx = np.random.randint(len(archive_X))
            base_x = archive_X[idx].copy()

            # Try several local perturbations
            for _ in range(3):
                if remaining <= 0:
                    break
                if time.time() > deadline:
                    break
                x = perturb(base_x, local_scale)
                f = np.array(problem(x))
                remaining -= 1
                if np.all(f < [3.0, 5.0, 7.0]):
                    update_archive(x, f)

            # Occasionally do a larger jump for exploration
            if np.random.random() < 0.1:
                if remaining <= 0:
                    break
                x = random_point()
                f = np.array(problem(x))
                remaining -= 1
                if np.all(f < [3.0, 5.0, 7.0]):
                    update_archive(x, f)

        # Return the archive (the return value is ignored by evaluator)
        if archive_X:
            return np.array(archive_X)
        return np.array([random_point()])
# EVOLVE-BLOCK-END
