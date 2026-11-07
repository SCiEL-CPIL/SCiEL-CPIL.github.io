"""Multi-objective optimizer for RE37: Archive-based Differential Evolution with deadline guard."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Archive-based Multi-Objective Differential Evolution for RE37."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 310.0  # Safety margin before 330s kill

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        dim = self.dim
        n_obj = self.n_obj
        budget = self.budget

        # Use fixed seed for reproducibility within the run
        rng = np.random.default_rng(42)

        # Population size for DE
        pop_size = min(60, max(30, budget // 3))
        # Archive size limit
        archive_max = 500

        # Initialize population uniformly
        X = lo + (hi - lo) * rng.random((pop_size, dim))
        F = np.array([problem(x) for x in X])
        used = pop_size

        # Initialize archive with all non-dominated points
        archive = list(zip(X, F))

        def is_dominated(f, archive_f):
            """Check if f is dominated by any point in archive_f (numpy array of shape (n, n_obj))."""
            if len(archive_f) == 0:
                return False
            # f is dominated if there exists a point that is <= in all and < in at least one
            leq = np.all(archive_f <= f, axis=1)
            lt = np.any(archive_f < f, axis=1)
            return np.any(leq & lt)

        def dominates(f, archive_f):
            """Check if f dominates any point in archive_f."""
            if len(archive_f) == 0:
                return False
            geq = np.all(archive_f >= f, axis=1)
            gt = np.any(archive_f > f, axis=1)
            return np.any(geq & gt)

        def update_archive(archive, x, f):
            """Add point (x, f) to archive, removing dominated points. Return updated archive."""
            # Check if new point is dominated
            if is_dominated(f, np.array([a[1] for a in archive]) if archive else np.empty((0, n_obj))):
                return archive

            # Remove points dominated by new point
            new_archive = []
            for ax, af in archive:
                if not (np.all(af <= f) and np.any(af < f)):
                    new_archive.append((ax, af))

            new_archive.append((x, f))

            # If archive too large, prune by removing points with smallest crowding distance
            if len(new_archive) > archive_max:
                # Compute crowding distance
                F_arr = np.array([a[1] for a in new_archive])
                cd = np.zeros(len(new_archive))
                for m in range(n_obj):
                    vals = F_arr[:, m]
                    order = np.argsort(vals)
                    cd[order[0]] = np.inf
                    cd[order[-1]] = np.inf
                    val_range = vals[order[-1]] - vals[order[0]]
                    if val_range > 1e-12:
                        for i in range(1, len(order) - 1):
                            cd[order[i]] += (vals[order[i + 1]] - vals[order[i - 1]]) / val_range

                # Remove points with smallest crowding distance
                order = np.argsort(cd)
                remove_count = len(new_archive) - archive_max
                for idx in order[:remove_count]:
                    new_archive.pop(idx)

            return new_archive

        # DE parameters
        F_cr = 0.8  # Crossover rate
        F_weight = 0.5  # Weight factor
        best_cr = 0.8
        best_weight = 0.5

        # Main DE loop
        while used < budget:
            if time.time() > deadline:
                break

            # Get archive as arrays
            if archive:
                arch_X = np.array([a[0] for a in archive])
                arch_F = np.array([a[1] for a in archive])
            else:
                arch_X = np.empty((0, dim))
                arch_F = np.empty((0, n_obj))

            # Create offspring using DE/rand/1 and DE/best/1
            new_X = []
            new_F = []

            for i in range(pop_size):
                if time.time() > deadline:
                    break

                # Choose strategy: 50% DE/rand/1, 50% DE/best/1
                if rng.random() < 0.5:
                    # DE/rand/1
                    r1, r2 = rng.choice(pop_size, size=2, replace=False)
                    base = X[r1]
                    diff = X[r2] - X[r1]
                    cr = F_cr
                    weight = F_weight
                else:
                    # DE/best/1: use best from archive
                    if len(arch_X) > 0:
                        # Pick a random point from archive as "best" proxy
                        best_idx = rng.integers(len(arch_X))
                        base = arch_X[best_idx]
                        r1, r2 = rng.choice(pop_size, size=2, replace=False)
                        diff = X[r1] - X[r2]
                        cr = best_cr
                        weight = best_weight
                    else:
                        # Fallback to DE/rand/1
                        r1, r2 = rng.choice(pop_size, size=2, replace=False)
                        base = X[r1]
                        diff = X[r2] - X[r1]
                        cr = F_cr
                        weight = F_weight

                # Generate trial vector
                v = base + weight * diff
                v = np.clip(v, lo, hi)

                # Binary crossover
                jrand = rng.integers(dim)
                for j in range(dim):
                    if rng.random() <= cr or j == jrand:
                        new_X_candidate = v
                    else:
                        new_X_candidate = X[i]

                # Evaluate
                f_new = problem(new_X_candidate)
                used += 1

                # Update archive
                archive = update_archive(archive, new_X_candidate, f_new)

                # Selection: compare with current individual
                # Use simple domination-based selection
                if is_dominated(f_new, F[i:i+1]):
                    # Current individual dominates new one, keep current
                    pass
                elif is_dominated(F[i], f_new.reshape(1, -1)):
                    # New one dominates current, replace
                    X[i] = new_X_candidate
                    F[i] = f_new
                else:
                    # Non-dominated: keep with some probability or use archive
                    # If archive has room, just keep new in archive (already done)
                    # For population, keep the one with better rank
                    # Simple heuristic: keep new with 50% probability
                    if rng.random() < 0.5:
                        X[i] = new_X_candidate
                        F[i] = f_new

                if used >= budget:
                    break

            # Every few generations, inject random points to maintain diversity
            if used < budget and rng.random() < 0.1:
                n_inject = min(5, budget - used)
                for _ in range(n_inject):
                    x_rand = lo + (hi - lo) * rng.random(dim)
                    f_rand = problem(x_rand)
                    used += 1
                    archive = update_archive(archive, x_rand, f_rand)
                    if used >= budget:
                        break

        return X
# EVOLVE-BLOCK-END
