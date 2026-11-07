"""Multi-objective optimizer for WFG4 using coordinate-descent with hypervolume-guided scalarization."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Coordinate-descent optimizer with hypervolume contribution as scalar fitness for WFG4."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280  # Safety margin before 300s limit

        lo = np.array([b[0] for b in bounds], dtype=np.float64)
        hi = np.array([b[1] for b in bounds], dtype=np.float64)
        rng = np.random

        dim = self.dim
        n_obj = self.n_obj
        budget = self.budget
        used = [0]  # Use list for mutable counter in nested functions

        # Reference point for hypervolume calculation
        ref = np.array([3.0, 5.0, 7.0], dtype=np.float64)

        # Archive of non-dominated solutions
        archive_X = []
        archive_F = []

        def is_valid(f):
            """Check if all objectives are strictly better than reference point."""
            return np.all(f < ref)

        def dominates(f_a, f_b):
            """Check if f_a dominates f_b (a <= b and a < b in at least one)."""
            return np.all(f_a <= f_b) and np.any(f_a < f_b)

        def update_archive(x_new, f_new):
            """Update archive with new point if non-dominated."""
            nonlocal archive_X, archive_F
            if not is_valid(f_new):
                return

            # Check if new point is dominated by any existing
            for f_old in archive_F:
                if dominates(f_old, f_new):
                    return

            # Remove points dominated by new point
            new_archive_X = []
            new_archive_F = []
            for x_old, f_old in zip(archive_X, archive_F):
                if dominates(f_new, f_old):
                    continue
                new_archive_X.append(x_old)
                new_archive_F.append(f_old)

            new_archive_X.append(x_new)
            new_archive_F.append(f_new)
            archive_X = new_archive_X
            archive_F = new_archive_F

        def hypervolume_contribution(f_new, archive_F):
            """
            Compute the hypervolume contribution of f_new given the current archive.
            This is the volume that would be added to the hypervolume if f_new were added
            to the archive. Points not better than ref contribute 0.
            """
            if not is_valid(f_new):
                return 0.0

            # Hypervolume of archive before adding f_new
            hv_before = _compute_hypervolume(archive_F, ref)

            # Hypervolume of archive after adding f_new
            archive_F_new = list(archive_F) + [f_new]
            hv_after = _compute_hypervolume(archive_F_new, ref)

            return hv_after - hv_before

        def _compute_hypervolume(F, ref):
            """
            Compute exact hypervolume for 3 objectives using inclusion-exclusion.
            F is a list of arrays, ref is the reference point.
            Only points strictly better than ref in all objectives count.
            """
            # Filter points strictly better than ref
            valid_F = [f for f in F if np.all(f < ref)]
            if not valid_F:
                return 0.0

            # For 3 objectives, the hypervolume is the volume of the union of boxes
            # from each point to the reference point.
            # Use the inclusion-exclusion principle for small sets, or a sweep-line approach.
            # For small archives, we can use a recursive decomposition.

            # Convert to numpy array
            F_arr = np.array(valid_F)
            n = len(F_arr)

            if n == 0:
                return 0.0
            if n == 1:
                f = F_arr[0]
                return float(np.prod(ref - f))

            # For small n, use the standard 3D hypervolume algorithm
            # Sort by first objective
            order = np.argsort(F_arr[:, 0])
            F_sorted = F_arr[order]

            # Recursive decomposition: 
            # The hypervolume can be computed by decomposing along the first objective.
            # For each point i (sorted by f1), the contribution is:
            # (f1[i+1] - f1[i]) * HV_2D(F_i, ref_2D)
            # where F_i are points with f1 <= f1[i+1] (i.e., points 0..i)
            # and ref_2D is the reference in the remaining 2 dimensions.

            # Actually, let's use a simpler approach for small n:
            # The hypervolume is the volume of the union of boxes [f_i, ref].
            # For 3D, we can use the inclusion-exclusion over all subsets, but that's 2^n.
            # For n up to ~20, this is too slow.
            # Instead, use a sweep-line approach.

            # Sweep-line in 3D:
            # Project all points and ref onto the 2D plane (obj2, obj3).
            # The hypervolume is the integral over obj1 of the 2D hypervolume of the
            # projection of points with f1 <= current_obj1.

            # Simpler: for small archives (n < 30), use the standard algorithm.
            # Let's implement a clean 3D hypervolume calculation.

            # Sort by first dimension
            F_sorted = F_arr[np.argsort(F_arr[:, 0])]

            hv = 0.0
            prev_f1 = F_sorted[0, 0]

            # For each "layer" in the first dimension
            for i in range(n):
                f1_i = F_sorted[i, 0]
                f1_next = F_sorted[i + 1, 0] if i + 1 < n else ref[0]

                # The 2D hypervolume of points 0..i in the (obj2, obj3) plane
                # with reference (ref[1], ref[2])
                F_2d = F_sorted[:i + 1, 1:]  # Points 0..i, dimensions 1 and 2
                hv_2d = _hypervolume_2d(F_2d, ref[1:])

                hv += (f1_next - f1_i) * hv_2d

            return float(hv)

        def _hypervolume_2d(F, ref):
            """Compute 2D hypervolume for points F (n x 2) with reference ref (2,)."""
            if len(F) == 0:
                return 0.0

            # Sort by first dimension
            F_sorted = F[np.argsort(F[:, 0])]
            n = len(F_sorted)

            hv = 0.0
            for i in range(n):
                f1_i = F_sorted[i, 0]
                f1_next = F_sorted[i + 1, 0] if i + 1 < n else ref[0]
                f2_max = min(F_sorted[:i + 1, 1].min(), ref[1])
                # The 2D hypervolume contribution is (f1_next - f1_i) * (ref[1] - min_f2)
                # where min_f2 is the minimum f2 among points 0..i
                min_f2 = F_sorted[:i + 1, 1].min()
                if min_f2 < ref[1]:
                    hv += (f1_next - f1_i) * (ref[1] - min_f2)

            return float(hv)

        def scalar_fitness(x):
            """
            Scalar fitness function using hypervolume contribution.
            We want to maximize this, so we minimize its negative.
            """
            if used[0] >= budget or time.time() > deadline:
                return np.inf

            x = np.clip(x, lo, hi)
            f = np.array(problem(x), dtype=np.float64)
            used[0] += 1

            if not is_valid(f):
                return np.inf

            # Compute hypervolume contribution
            hv_contrib = hypervolume_contribution(f, archive_F)
            return -hv_contrib  # Minimize negative contribution

        # Phase 1: Random initialization with local coordinate descent
        n_starts = 15  # Number of random starting points

        for s in range(n_starts):
            if used[0] >= budget or time.time() > deadline:
                break

            # Random start
            x0 = lo + (hi - lo) * rng.random_sample(dim)
            f0 = np.array(problem(x0), dtype=np.float64)
            used[0] += 1
            update_archive(x0, f0)

            # Coordinate descent: iterate over each dimension, do ternary search
            n_cd_iterations = 3  # Number of full coordinate descent passes
            for it in range(n_cd_iterations):
                if used[0] >= budget - 10 or time.time() > deadline:
                    break

                for d in range(dim):
                    if used[0] >= budget - 5 or time.time() > deadline:
                        break

                    # Ternary search on dimension d
                    a = lo[d]
                    b = hi[d]

                    # Perform ternary search with limited evaluations
                    n_ternary = 4  # Number of ternary search steps (each uses 2 evals)
                    for _ in range(n_ternary):
                        if used[0] >= budget - 2 or time.time() > deadline:
                            break

                        m1 = a + (b - a) / 3.0
                        m2 = b - (b - a) / 3.0

                        # Evaluate at m1
                        x1 = x0.copy()
                        x1[d] = m1
                        x1 = np.clip(x1, lo, hi)
                        f1 = np.array(problem(x1), dtype=np.float64)
                        used[0] += 1

                        # Evaluate at m2
                        x2 = x0.copy()
                        x2[d] = m2
                        x2 = np.clip(x2, lo, hi)
                        f2 = np.array(problem(x2), dtype=np.float64)
                        used[0] += 1

                        # Compare hypervolume contributions
                        hv1 = hypervolume_contribution(f1, archive_F) if is_valid(f1) else -np.inf
                        hv2 = hypervolume_contribution(f2, archive_F) if is_valid(f2) else -np.inf

                        if hv1 > hv2:
                            b = m2
                        else:
                            a = m1

                        # Update archive with the better point
                        if hv1 >= hv2:
                            if is_valid(f1):
                                update_archive(x1, f1)
                                x0 = x1
                        else:
                            if is_valid(f2):
                                update_archive(x2, f2)
                                x0 = x2

                    # After ternary search, evaluate at the best point
                    if used[0] < budget and time.time() < deadline:
                        x_best = x0.copy()
                        x_best[d] = (a + b) / 2.0
                        x_best = np.clip(x_best, lo, hi)
                        f_best = np.array(problem(x_best), dtype=np.float64)
                        used[0] += 1
                        if is_valid(f_best):
                            update_archive(x_best, f_best)
                            x0 = x_best

        # Phase 2: Use remaining budget for more random starts with lighter local search
        while used[0] < budget and time.time() < deadline:
            x_rand = lo + (hi - lo) * rng.random_sample(dim)
            f_rand = np.array(problem(x_rand), dtype=np.float64)
            used[0] += 1
            update_archive(x_rand, f_rand)

            # Light local search: just one pass over a few random dimensions
            if used[0] < budget - 5 and time.time() < deadline:
                dims_to_try = rng.choice(dim, size=min(3, dim), replace=False)
                for d in dims_to_try:
                    if used[0] >= budget - 2 or time.time() > deadline:
                        break

                    # Simple 3-point evaluation around current
                    step = (hi[d] - lo[d]) * 0.1
                    candidates = [x_rand[d] - step, x_rand[d], x_rand[d] + step]
                    best_f = f_rand
                    best_x = x_rand

                    for cand in candidates:
                        if used[0] >= budget or time.time() > deadline:
                            break
                        x_cand = x_rand.copy()
                        x_cand[d] = np.clip(cand, lo[d], hi[d])
                        f_cand = np.array(problem(x_cand), dtype=np.float64)
                        used[0] += 1

                        if is_valid(f_cand):
                            # Check if it's better (non-dominated by current)
                            if not dominates(best_f, f_cand):
                                best_f = f_cand
                                best_x = x_cand

                    if is_valid(best_f):
                        update_archive(best_x, best_f)
                        x_rand = best_x

        # Return the archive
        if not archive_X:
            # Fallback: return one random point
            x_ret = lo + (hi - lo) * rng.random_sample(dim)
            return np.array([x_ret])

        # Filter to non-dominated
        non_dom_X = []
        non_dom_F = []
        for x, f in zip(archive_X, archive_F):
            is_dom = False
            for f2 in archive_F:
                if f2 is not f and dominates(f2, f):
                    is_dom = True
                    break
            if not is_dom:
                non_dom_X.append(x)
                non_dom_F.append(f)

        if not non_dom_X:
            non_dom_X = archive_X[:1]

        return np.array(non_dom_X)
# EVOLVE-BLOCK-END
