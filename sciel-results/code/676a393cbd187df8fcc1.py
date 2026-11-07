"""Multi-objective optimizer on WFG4 using hypervolume-indicator-driven sequential search."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize_scalar


class Optimizer:
    """Hypervolume-indicator-driven sequential multi-objective optimizer."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 270.0  # safety margin before 330s kill

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

        def hypervolume_3d(points, ref):
            """Compute exact 3D hypervolume for a set of points vs reference point.
            Only points strictly better than ref in all objectives count."""
            valid = [p for p in points if np.all(p < ref)]
            if len(valid) == 0:
                return 0.0
            if len(valid) == 1:
                p = valid[0]
                return float(np.prod(ref - p))

            # Use sweep-line approach for 3D hypervolume
            # Sort points by first objective
            valid_sorted = sorted(valid, key=lambda p: p[0])

            hv = 0.0
            n = len(valid_sorted)

            for i in range(n):
                pi = valid_sorted[i]
                # Width in dimension 0
                w0 = ref[0] - pi[0]
                if w0 <= 0:
                    continue

                # For this point, find the 2D hypervolume contribution
                # in dimensions 1 and 2, considering only points with f0 >= pi[0]
                # (i.e., points that come after pi in the sorted order, plus pi itself)

                # The 2D HV of points {pi, pi+1, ..., pi+n-1} in dims 1,2
                # with reference (ref[1], ref[2])
                # This is the area in the (y,z) plane covered by rectangles
                # from each point to the reference

                # Collect all y and z coordinates
                ys = sorted(set([p[1] for p in valid_sorted[i:]] + [ref[1]]))
                zs = sorted(set([p[2] for p in valid_sorted[i:]] + [ref[2]]))

                ny = len(ys) - 1
                nz = len(zs) - 1

                if ny == 0 or nz == 0:
                    continue

                covered = np.zeros((ny, nz), dtype=bool)

                for p in valid_sorted[i:]:
                    yi = np.searchsorted(ys, p[1]) - 1
                    zi = np.searchsorted(zs, p[2]) - 1
                    if yi >= 0 and zi >= 0:
                        covered[yi:, zi:] = True

                area_2d = 0.0
                for j in range(ny):
                    dy = ys[j + 1] - ys[j]
                    if dy == 0:
                        continue
                    for k in range(nz):
                        if covered[j, k]:
                            dz = zs[k + 1] - zs[k]
                            area_2d += dy * dz

                hv += w0 * area_2d

            return hv

        def marginal_hv(f_candidate):
            """Compute the marginal hypervolume contribution of a candidate point.
            This is HV(archive ∪ {f_candidate}) - HV(archive)."""
            if np.any(f_candidate >= ref_point):
                return 0.0
            hv_before = hypervolume_3d(archive, ref_point)
            hv_after = hypervolume_3d(archive + [f_candidate], ref_point)
            return hv_after - hv_before

        # Phase 1: Front-biased initialization
        # WFG4 Pareto front: f1 = x1, f2 = (1-x1)*x2, f3 = (1-x1)*(1-x2)
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
                if rng.random() < 0.5:
                    x[d] = lo[d] + span[d] * u
                else:
                    x[d] = (lo[d] + hi[d]) / 2.0 + span[d] * 0.3 * rng.standard_normal()
                    x[d] = np.clip(x[d], lo[d], hi[d])

            f = problem(x)
            used += 1
            update_archive(f, x)

        # Phase 2: Hypervolume-indicator-driven sequential search
        # Use minimize_scalar to find optimal perturbation along a direction
        best_hv = hypervolume_3d(archive, ref_point)
        stagnation_counter = 0
        stagnation_threshold = 15

        while used < budget:
            if time.time() > deadline - 5.0:
                break

            # Adaptive strategy
            if len(archive) == 0:
                x = lo + span * rng.random(dim)
                f = problem(x)
                used += 1
                update_archive(f, x)
                continue

            # With some probability, do global exploration
            if rng.random() < 0.12 + 0.08 * min(stagnation_counter / stagnation_threshold, 1.0):
                x = lo + span * rng.random(dim)
                f = problem(x)
                used += 1
                if update_archive(f, x):
                    current_hv = hypervolume_3d(archive, ref_point)
                    if current_hv > best_hv:
                        best_hv = current_hv
                        stagnation_counter = 0
                    else:
                        stagnation_counter += 1
                else:
                    stagnation_counter += 1
                continue

            # Hypervolume-guided local search:
            # Pick a random archive point and use minimize_scalar to find
            # the best perturbation magnitude along a random direction
            idx = rng.randint(len(archive))
            base_x = archive_X[idx]

            # Generate a random direction
            direction = rng.standard_normal(dim)
            direction_norm = np.linalg.norm(direction)
            if direction_norm > 0:
                direction = direction / direction_norm
            else:
                direction = np.ones(dim) / np.sqrt(dim)

            # Adaptive step size range
            progress = used / budget
            max_step = 0.4 * (1.0 - 0.5 * progress)

            # Use minimize_scalar to find the best step size along this direction
            # that maximizes marginal hypervolume

            def neg_marginal_hv(t):
                """Negative marginal hypervolume for minimization."""
                x_candidate = base_x + t * direction * span
                x_candidate = np.clip(x_candidate, lo, hi)
                f_candidate = problem(x_candidate)
                used_ref = [0]  # track if we used an evaluation
                if used_ref[0] == 0 and time.time() > deadline - 3.0:
                    return 0.0
                return -marginal_hv(f_candidate)

            # We need to be careful with evaluations in minimize_scalar
            # Use a limited number of function evaluations
            try:
                result = minimize_scalar(
                    neg_marginal_hv,
                    bounds=(0, max_step),
                    method='bounded',
                    options={'maxiter': 5, 'xatol': 0.01}
                )
                if result.success:
                    best_t = result.x
                    x = base_x + best_t * direction * span
                    x = np.clip(x, lo, hi)
                else:
                    x = base_x + 0.1 * direction * span
                    x = np.clip(x, lo, hi)
            except Exception:
                x = base_x + 0.1 * direction * span
                x = np.clip(x, lo, hi)

            f = problem(x)
            used += 1

            if update_archive(f, x):
                current_hv = hypervolume_3d(archive, ref_point)
                if current_hv > best_hv:
                    best_hv = current_hv
                    stagnation_counter = 0
                else:
                    stagnation_counter += 1
            else:
                stagnation_counter += 1

        return archive_X if archive_X else [np.zeros(dim)]

# EVOLVE-BLOCK-END
