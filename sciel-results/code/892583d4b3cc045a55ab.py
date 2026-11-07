"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Archive-based multi-objective optimizer for RE21."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 290.0  # safety margin before 300s limit

        lo = np.array([b[0] for b in bounds], dtype=float)
        hi = np.array([b[1] for b in bounds], dtype=float)
        ref = np.array([3000.0, 0.06])
        rng = np.random

        used = 0
        all_X = []
        all_F = []

        def evaluate(x):
            nonlocal used
            x_clipped = np.clip(x, lo, hi)
            fc = problem(x_clipped)
            used += 1
            all_X.append(x_clipped.copy())
            all_F.append(np.array(fc, dtype=float))
            return np.array(fc, dtype=float)

        def is_dominated(f, archive_F):
            """Check if f is dominated by any point in archive_F."""
            for af in archive_F:
                if np.all(af <= f) and np.any(af < f):
                    return True
            return False

        def update_archive(x, f, archive_X, archive_F):
            """Update archive with new point if it's non-dominated. Returns True if added."""
            # Check if new point is dominated
            if is_dominated(f, archive_F):
                return False
            # Remove dominated points from archive
            new_archive_X = []
            new_archive_F = []
            for i in range(len(archive_X)):
                if not (np.all(f <= archive_F[i]) and np.any(f < archive_F[i])):
                    new_archive_X.append(archive_X[i])
                    new_archive_F.append(archive_F[i])
            # Add new point
            new_archive_X.append(x.copy())
            new_archive_F.append(f.copy())
            archive_X[:] = new_archive_X
            archive_F[:] = new_archive_F
            return True

        def compute_hv_2d(points_F, ref_point):
            """Compute hypervolume for 2D points relative to reference point."""
            # Filter points that are better than reference in all objectives
            valid = [f for f in points_F if np.all(f < ref_point)]
            if len(valid) == 0:
                return 0.0
            # Sort by first objective
            valid.sort(key=lambda x: x[0])
            # Remove dominated points
            non_dominated = []
            for f in valid:
                if not non_dominated or f[1] < non_dominated[-1][1]:
                    non_dominated.append(f)
            # Compute HV
            hv = 0.0
            for i in range(len(non_dominated)):
                x1 = non_dominated[i][0]
                y1 = non_dominated[i][1]
                if i + 1 < len(non_dominated):
                    x2 = non_dominated[i + 1][0]
                else:
                    x2 = ref_point[0]
                hv += (x2 - x1) * (ref_point[1] - y1)
            return hv

        # Archive for Pareto front
        archive_X = []
        archive_F = []

        def hv_guided_select(archive_F, n_select=2):
            """Select parents based on hypervolume contribution."""
            if len(archive_F) <= n_select:
                return list(range(len(archive_F)))
            
            # Compute HV contribution of each point
            contributions = []
            for i in range(len(archive_F)):
                # HV with point i removed
                others = [archive_F[j] for j in range(len(archive_F)) if j != i]
                hv_without = compute_hv_2d(others, ref)
                hv_with = compute_hv_2d(archive_F, ref)
                contributions.append(hv_with - hv_without)
            
            # Select top contributors
            order = np.argsort(contributions)[::-1]
            return list(order[:n_select])

        def nelder_mead_polish(start_x, max_iter=20, tol=1e-6):
            """Polish a solution using Nelder-Mead on a scalarized objective."""
            nonlocal used
            n = self.dim
            
            # Use a weighted sum based on current front
            if len(archive_F) == 0:
                weights = np.array([0.5, 0.5])
            else:
                # Pick weights based on diversity
                weights = np.array([0.5, 0.5])
            
            # Build initial simplex
            simplex = [start_x.copy()]
            for i in range(n):
                dx = start_x.copy()
                step = 0.05 * (hi[i] - lo[i])
                dx[i] += step
                simplex.append(np.clip(dx, lo, hi))

            # Evaluate initial simplex
            fvals = []
            for i in range(n + 1):
                if used >= self.budget or time.time() > deadline:
                    return
                f = evaluate(simplex[i])
                fvals.append(np.dot(weights, f))

            # Nelder-Mead iterations
            for it in range(max_iter):
                if used >= self.budget or time.time() > deadline:
                    break

                # Sort simplex
                order = np.argsort(fvals)
                simplex = [simplex[i] for i in order]
                fvals = [fvals[i] for i in order]

                # Check convergence
                if abs(fvals[-1] - fvals[0]) < tol:
                    break

                centroid = np.mean(simplex[:-1], axis=0)

                # Reflection
                alpha = 1.0
                xr = centroid + alpha * (centroid - simplex[-1])
                xr = np.clip(xr, lo, hi)
                if used >= self.budget or time.time() > deadline:
                    break
                fr = np.dot(weights, evaluate(xr))

                if fr < fvals[0]:
                    # Expansion
                    gamma = 2.0
                    xe = centroid + gamma * (centroid - simplex[-1])
                    xe = np.clip(xe, lo, hi)
                    if used >= self.budget or time.time() > deadline:
                        break
                    fe = np.dot(weights, evaluate(xe))
                    if fe < fr:
                        simplex[-1] = xe
                        fvals[-1] = fe
                    else:
                        simplex[-1] = xr
                        fvals[-1] = fr
                elif fr < fvals[-2]:
                    simplex[-1] = xr
                    fvals[-1] = fr
                else:
                    # Contraction
                    rho = 0.5
                    xc = centroid + rho * (simplex[-1] - centroid)
                    xc = np.clip(xc, lo, hi)
                    if used >= self.budget or time.time() > deadline:
                        break
                    fc = np.dot(weights, evaluate(xc))
                    if fc < fvals[-1]:
                        simplex[-1] = xc
                        fvals[-1] = fc
                    else:
                        # Shrink
                        for i in range(1, n + 1):
                            simplex[i] = simplex[0] + 0.5 * (simplex[i] - simplex[0])
                            simplex[i] = np.clip(simplex[i], lo, hi)
                            if used >= self.budget or time.time() > deadline:
                                return
                            fvals[i] = np.dot(weights, evaluate(simplex[i]))

        # Phase 1: Initial sampling to build archive
        n_init = 30
        for i in range(n_init):
            if time.time() > deadline or used >= self.budget:
                break
            x = lo + (hi - lo) * rng.random(self.dim)
            f = evaluate(x)
            update_archive(x, f, archive_X, archive_F)

        # Phase 2: Main loop with HV-guided selection and local search
        while used < self.budget and time.time() < deadline:
            if len(archive_F) == 0:
                # Fallback: random sampling
                x = lo + (hi - lo) * rng.random(self.dim)
                f = evaluate(x)
                update_archive(x, f, archive_X, archive_F)
                continue

            # Select parent based on HV contribution
            parent_idx = hv_guided_select(archive_F, n_select=1)[0]
            parent_x = archive_X[parent_idx]

            # Adaptive mutation scale based on front spread
            if len(archive_F) > 1:
                spread = np.std(archive_F, axis=0)
                mutation_scale = 0.3 * np.maximum(spread, 0.1 * (hi - lo))
            else:
                mutation_scale = 0.3 * (hi - lo)

            # Generate offspring
            n_offspring = 3
            for j in range(n_offspring):
                if used >= self.budget or time.time() > deadline:
                    break

                # Mutation
                noise = rng.normal(0, 1, self.dim)
                child_x = parent_x + noise * mutation_scale
                child_x = np.clip(child_x, lo, hi)

                # Evaluate and update archive
                child_f = evaluate(child_x)
                if update_archive(child_x, child_f, archive_X, archive_F):
                    # New point added to archive - polish it
                    if used < self.budget - 10 and time.time() < deadline - 5:
                        nelder_mead_polish(child_x, max_iter=15)

            # Periodic random exploration
            if used % 20 == 0 and used < self.budget:
                if time.time() < deadline:
                    x = lo + (hi - lo) * rng.random(self.dim)
                    f = evaluate(x)
                    update_archive(x, f, archive_X, archive_F)

        # Phase 3: Final polish of best archive points
        if len(archive_F) > 0 and time.time() < deadline - 10:
            # Polish top HV contributors
            top_indices = hv_guided_select(archive_F, n_select=min(5, len(archive_F)))
            for idx in top_indices:
                if used >= self.budget or time.time() > deadline - 5:
                    break
                nelder_mead_polish(archive_X[idx], max_iter=20)

        return all_X
# EVOLVE-BLOCK-END
