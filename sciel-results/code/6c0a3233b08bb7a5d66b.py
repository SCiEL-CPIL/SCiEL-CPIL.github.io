"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-objective optimizer for RE21 using multi-start local search."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        t0 = time.time()
        deadline = t0 + 280.0

        lo = np.array([b[0] for b in bounds], dtype=np.float64)
        hi = np.array([b[1] for b in bounds], dtype=np.float64)
        span = hi - lo
        dim = self.dim
        n_obj = self.n_obj
        budget = self.budget

        # Archive of all non-dominated points seen
        archive_F = []
        archive_X = []

        def is_dominated(f, archive_F_list):
            """Check if f is dominated by any point in archive."""
            for af in archive_F_list:
                if np.all(af <= f) and np.any(af < f):
                    return True
            return False

        def add_to_archive(x, f):
            """Add point to archive if not dominated."""
            if is_dominated(f, archive_F):
                return
            # Remove archive points dominated by new point
            keep = []
            for j in range(len(archive_F)):
                if not (np.all(f <= archive_F[j]) and np.any(f < archive_F[j])):
                    keep.append(j)
            for j in reversed(range(len(archive_F))):
                if j not in keep:
                    archive_F.pop(j)
                    archive_X.pop(j)
            archive_F.append(f.copy())
            archive_X.append(x.copy())

        # Nelder-Mead local search with adaptive step
        def nelder_mead(x0, f0, max_iter=50, tol=1e-8):
            """Local search using Nelder-Mead method."""
            n = dim
            # Create initial simplex
            simplex = [x0.copy() + np.zeros(n)]
            for i in range(n):
                perturbation = x0.copy()
                if abs(perturbation[i]) > 1e-10:
                    perturbation[i] *= 1.05
                else:
                    perturbation[i] += 0.05 * span[i]
                simplex.append(perturbation)

            # Clip to bounds
            for i in range(len(simplex)):
                simplex[i] = np.clip(simplex[i], lo, hi)

            # Evaluate simplex
            fvals = [f0.copy()]
            for i in range(1, len(simplex)):
                fvals.append(problem(simplex[i]))

            # Track budget usage
            nonlocal used
            used += n  # n additional evaluations for simplex

            # Nelder-Mead iterations
            alpha, gamma, rho, sigma = 1.0, 2.0, -0.5, 0.5
            for _ in range(max_iter):
                if time.time() > deadline:
                    break
                if used >= budget:
                    break

                # Sort by sum of objectives
                order = np.argsort([np.sum(fv) for fv in fvals])
                simplex = [simplex[i] for i in order]
                fvals = [fvals[i] for i in order]

                # Check convergence
                centroid = np.mean(simplex[:-1], axis=0)
                if np.max(np.abs(simplex[-1] - centroid)) < tol:
                    break

                # Reflection
                x_r = centroid + alpha * (centroid - simplex[-1])
                x_r = np.clip(x_r, lo, hi)
                f_r = problem(x_r)
                used += 1
                if used >= budget or time.time() > deadline:
                    break

                if f_r < fvals[0]:
                    # Expansion
                    x_e = centroid + gamma * (x_r - centroid)
                    x_e = np.clip(x_e, lo, hi)
                    f_e = problem(x_e)
                    used += 1
                    if used >= budget or time.time() > deadline:
                        break
                    if f_e < f_r:
                        simplex[-1] = x_e
                        fvals[-1] = f_e
                    else:
                        simplex[-1] = x_r
                        fvals[-1] = f_r
                elif f_r < fvals[-2]:
                    simplex[-1] = x_r
                    fvals[-1] = f_r
                else:
                    # Contraction
                    x_c = centroid + rho * (simplex[-1] - centroid)
                    x_c = np.clip(x_c, lo, hi)
                    f_c = problem(x_c)
                    used += 1
                    if used >= budget or time.time() > deadline:
                        break
                    if f_c < fvals[-1]:
                        simplex[-1] = x_c
                        fvals[-1] = f_c
                    else:
                        # Shrink
                        for i in range(1, len(simplex)):
                            simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                            simplex[i] = np.clip(simplex[i], lo, hi)
                            fvals[i] = problem(simplex[i])
                            used += 1
                            if used >= budget or time.time() > deadline:
                                break
                        if used >= budget or time.time() > deadline:
                            break

                # Add best point to archive
                best_idx = np.argmin([np.sum(fv) for fv in fvals])
                add_to_archive(simplex[best_idx], fvals[best_idx])

            # Return best point found
            best_idx = np.argmin([np.sum(fv) for fv in fvals])
            return simplex[best_idx], fvals[best_idx]

        # Multi-start strategy
        # Phase 1: Initial diverse sampling
        n_initial = 20
        X_init = np.zeros((n_initial, dim))
        # Some near origin (Rastrigin-like optimum)
        for i in range(n_initial // 2):
            X_init[i] = 0.05 * span * np.random.standard_normal(dim)
        # Some random
        for i in range(n_initial // 2, n_initial):
            X_init[i] = lo + span * np.random.random(dim)

        F_init = []
        used = 0
        for i in range(n_initial):
            f = problem(X_init[i])
            F_init.append(f)
            used += 1
            add_to_archive(X_init[i], f)
            if used >= budget or time.time() > deadline:
                break

        # Phase 2: Multi-start local search from best archive points
        # Select diverse starting points from archive
        start_points = []
        if len(archive_F) > 0:
            # Take non-dominated points as starting points
            nd_mask = np.zeros(len(archive_F), dtype=bool)
            for i in range(len(archive_F)):
                for j in range(len(archive_F)):
                    if i != j and np.all(archive_F[j] <= archive_F[i]) and np.any(archive_F[j] < archive_F[i]):
                        nd_mask[i] = True
                        break
            nd_idx = np.where(~nd_mask)[0]
            if len(nd_idx) > 0:
                # Take up to 8 diverse starting points
                max_starts = min(8, len(nd_idx), (budget - used) // 10)
                if max_starts > 0:
                    # Randomly select from non-dominated
                    selected = np.random.choice(nd_idx, size=min(max_starts, len(nd_idx)), replace=False)
                    for idx in selected:
                        start_points.append(archive_X[idx])

        # If not enough starts, add some random points
        while len(start_points) < 4 and used < budget - 10:
            x = lo + span * np.random.random(dim)
            f = problem(x)
            used += 1
            start_points.append(x)
            add_to_archive(x, f)
            if time.time() > deadline:
                break

        # Run local search from each starting point
        for x_start in start_points:
            if used >= budget or time.time() > deadline:
                break
            x_best, f_best = nelder_mead(x_start, problem(x_start))
            used += 1  # for the initial evaluation in nelder_mead
            add_to_archive(x_best, f_best)

        # Phase 3: Final refinement - try to improve with smaller steps
        if used < budget and len(archive_F) > 0:
            # Take the best few points and do fine-tuning
            # Sort archive by sum of objectives
            sums = [np.sum(f) for f in archive_F]
            order = np.argsort(sums)
            n_refine = min(5, len(order), (budget - used) // 5)
            for idx in order[:n_refine]:
                if used >= budget or time.time() > deadline:
                    break
                x_start = archive_X[idx].copy()
                f_start = archive_F[idx].copy()
                # Small perturbations
                for _ in range(3):
                    if used >= budget or time.time() > deadline:
                        break
                    x_try = x_start + 0.01 * span * np.random.standard_normal(dim)
                    x_try = np.clip(x_try, lo, hi)
                    f_try = problem(x_try)
                    used += 1
                    add_to_archive(x_try, f_try)
                    if np.sum(f_try) < np.sum(f_start):
                        x_start = x_try
                        f_start = f_try

        # Return the archive points
        if len(archive_X) > 0:
            return np.array(archive_X)
        else:
            return np.array([X_init[0]])
# EVOLVE-BLOCK-END
