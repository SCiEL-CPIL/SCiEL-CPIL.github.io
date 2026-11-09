"""Multi-objective optimizer for RE21 using archive-based evolutionary strategy."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Archive-based multi-objective evolutionary optimizer."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280.0  # safety margin before 300s limit

        lo = np.array([b[0] for b in bounds], dtype=np.float64)
        hi = np.array([b[1] for b in bounds], dtype=np.float64)
        span = hi - lo
        dim = self.dim
        budget = self.budget

        # LHS initialization for good spread
        n_init = min(20, budget)
        # Latin Hypercube Sampling
        idx = np.arange(n_init)
        lhs = np.zeros((n_init, dim))
        for d in range(dim):
            perm = np.random.permutation(n_init)
            lhs[:, d] = (perm + np.random.random(n_init)) / n_init
        X_init = lo + span * lhs

        # Archive: list of (x, f) that are non-dominated
        archive_x = []
        archive_f = []
        used = 0

        def is_dominated(f, archive_f_list):
            """Check if f is dominated by any point in archive_f_list."""
            for af in archive_f_list:
                if np.all(af <= f) and np.any(af < f):
                    return True
            return False

        def dominates(f, af):
            """Check if f dominates af."""
            return np.all(f <= af) and np.any(f < af)

        def add_to_archive(x, f):
            """Add point to archive, removing dominated points."""
            nonlocal archive_x, archive_f
            # Check if new point is dominated
            for i, af in enumerate(archive_f):
                if np.all(af <= f) and np.any(af < f):
                    return  # dominated, don't add
            # Remove points dominated by new point
            new_archive_x = []
            new_archive_f = []
            for i, (ax, af) in enumerate(zip(archive_x, archive_f)):
                if not (np.all(f <= af) and np.any(f < af)):
                    new_archive_x.append(ax)
                    new_archive_f.append(af)
            new_archive_x.append(x)
            new_archive_f.append(f)
            archive_x = new_archive_x
            archive_f = new_archive_f

        # Initial evaluations
        for x in X_init:
            if time.time() > deadline:
                return archive_x
            f = np.array(problem(x), dtype=np.float64)
            add_to_archive(x.copy(), f)
            used += 1

        # Main loop: mutate non-dominated points
        # Use adaptive mutation step
        base_step = 0.15

        while used < budget:
            if time.time() > deadline:
                break

            # Select parent from archive
            if len(archive_x) == 0:
                # Fallback: random point
                x_parent = lo + span * np.random.random(dim)
            else:
                # Weighted selection: prefer points on the convex hull of the front
                # Simple approach: random selection from archive
                idx = np.random.randint(len(archive_x))
                x_parent = archive_x[idx].copy()

            # Gaussian mutation with adaptive step
            # Decay step size slightly as we use more budget
            progress = used / budget
            step = base_step * (1.0 - 0.3 * progress)

            # Sometimes use smaller perturbation (exploration vs exploitation)
            if np.random.random() < 0.3:
                step *= 0.5
            elif np.random.random() < 0.2:
                step *= 2.0

            noise = np.random.standard_normal(dim)
            x_child = x_parent + step * span * noise
            x_child = np.clip(x_child, lo, hi)

            f_child = np.array(problem(x_child), dtype=np.float64)
            used += 1

            add_to_archive(x_child.copy(), f_child)

            # Occasionally try a random restart to escape local regions
            if used < budget and np.random.random() < 0.05:
                x_rand = lo + span * np.random.random(dim)
                f_rand = np.array(problem(x_rand), dtype=np.float64)
                used += 1
                if used < budget:
                    add_to_archive(x_rand.copy(), f_rand)

        return archive_x
# EVOLVE-BLOCK-END
