"""Multi-objective optimizer on WFG4 using hypervolume-guided local search with front-biased initialization."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Hypervolume-guided multi-objective optimizer with front-biased initialization."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 270.0  # safety margin before 300s kill

        lo = np.array([b[0] for b in bounds], dtype=np.float64)
        hi = np.array([b[1] for b in bounds], dtype=np.float64)
        span = hi - lo
        dim = self.dim
        budget = self.budget
        ref_point = np.array([3.0, 5.0, 7.0], dtype=np.float64)

        rng = np.random

        # Archive: list of objective vectors (non-dominated)
        archive = []
        archive_X = []  # corresponding decision variables

        def dominates(f1, f2):
            """Check if f1 dominates f2."""
            return np.all(f1 <= f2) and np.any(f1 < f2)

        def update_archive(f, x):
            """Add f to archive if non-dominated, remove dominated archive points.
            Returns True if f was added."""
            # Check if f is dominated by any archive point
            for af in archive:
                if np.all(af <= f) and np.any(af < f):
                    return False
            # Remove archive points dominated by f
            new_archive = []
            new_archive_X = []
            for i, af in enumerate(archive):
                if not (np.all(f <= af) and np.any(f < af)):
                    new_archive.append(af)
                    new_archive_X.append(archive_X[i])
            new_archive.append(f.copy())
            new_archive_X.append(x.copy())
            archive[:] = new_archive
            archive_X[:] = new_archive_X
            return True

        # Phase 1: Front-biased initialization
        # WFG4 Pareto front structure: f1 = x1, f2 = (1-x1)*x2, f3 = (1-x1)*(1-x2)
        # Sample t uniformly and construct x to be near the front
        n_init = min(80, budget)
        used = 0

        for i in range(n_init):
            if used >= budget:
                break
            if time.time() > deadline - 5.0:
                break

            # Sample parameter t uniformly
            t = rng.random()

            # Construct x based on WFG4 front structure
            x = np.zeros(dim)
            x[0] = lo[0] + span[0] * t

            # For remaining variables, use a mix of front-biased and random values
            for d in range(1, dim):
                u = rng.random()
                # Mix of uniform and concentrated around center
                if rng.random() < 0.5:
                    x[d] = lo[d] + span[d] * u
                else:
                    # Concentrate around center with some noise
                    x[d] = (lo[d] + hi[d]) / 2.0 + span[d] * 0.3 * rng.standard_normal()
                    x[d] = np.clip(x[d], lo[d], hi[d])

            f = problem(x)
            used += 1
            update_archive(f, x)

        # Phase 2: Hypervolume-guided local refinement
        # Track best hypervolume to detect stagnation
        best_hv = 0.0
        stagnation_counter = 0
        stagnation_threshold = 20  # evaluations without improvement before increasing exploration

        def compute_hv_simple():
            """Compute approximate hypervolume for 3D case."""
            valid = [p for p in archive if np.all(p < ref_point)]
            if len(valid) == 0:
                return 0.0
            if len(valid) == 1:
                p = valid[0]
                return float(np.prod(ref_point - p))

            # Sort by first objective
            valid_sorted = sorted(valid, key=lambda p: p[0])
            hv = 0.0

            # For 3D, we can use a sweep line approach
            # Group by x-coordinate
            xs = sorted(set([p[0] for p in valid_sorted] + [ref_point[0]]))

            for i in range(len(xs) - 1):
                x1 = xs[i]
                x2 = xs[i + 1]
                if x2 <= x1:
                    continue

                # Find points with f1 in [x1, x2)
                pts_in_slice = [p for p in valid_sorted if x1 <= p[0] < x2]

                if len(pts_in_slice) == 0:
                    continue

                # For each point in this slice, compute the 2D hypervolume contribution
                # that is not covered by points with smaller f1
                max_y = 0.0
                max_z = 0.0
                for p in pts_in_slice:
                    y = ref_point[1] - p[1]
                    z = ref_point[2] - p[2]
                    if y > 0 and z > 0:
                        # This point contributes y*z to the 2D HV at this x-slice
                        # But we need to subtract overlap with previously seen points
                        # For simplicity, just take the max in each dimension
                        max_y = max(max_y, y)
                        max_z = max(max_z, z)

                if max_y > 0 and max_z > 0:
                    hv += (x2 - x1) * max_y * max_z

            return hv

        while used < budget:
            if time.time() > deadline - 5.0:
                break

            # Adaptive strategy based on stagnation
            if len(archive) == 0:
                # Fallback to random sampling
                x = lo + span * rng.random(dim)
            elif rng.random() < 0.15 + 0.1 * min(stagnation_counter / stagnation_threshold, 1.0):
                # Global exploration (increased probability during stagnation)
                x = lo + span * rng.random(dim)
            else:
                # Local refinement around a random archive point
                idx = rng.randint(len(archive))
                base_x = archive_X[idx]

                # Adaptive step size: start larger, decrease over time
                progress = used / budget
                base_step = 0.25 * (1.0 - 0.6 * progress)

                # Gaussian perturbation
                step = base_step * rng.standard_normal(dim)
                x = base_x + step * span
                x = np.clip(x, lo, hi)

            f = problem(x)
            used += 1

            if update_archive(f, x):
                # New point added to archive - good!
                # Check if hypervolume improved
                current_hv = compute_hv_simple()
                if current_hv > best_hv:
                    best_hv = current_hv
                    stagnation_counter = 0
                else:
                    stagnation_counter += 1
            else:
                stagnation_counter += 1

        return archive_X if archive_X else [np.zeros(dim)]

# EVOLVE-BLOCK-END
