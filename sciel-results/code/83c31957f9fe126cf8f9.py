# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Grid-based decomposition with local simplex refinement for multi-objective optimization."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        t0 = time.time()
        deadline = t0 + 280.0  # Safety margin before 330s kill

        lo = np.array([b[0] for b in bounds], dtype=np.float64)
        hi = np.array([b[1] for b in bounds], dtype=np.float64)
        dim = self.dim
        n_obj = self.n_obj
        budget = self.budget

        # Grid-based decomposition: partition into hypercubes
        # For 4D, use a small grid to ensure coverage
        grid_per_dim = 3  # 3^4 = 81 cells
        # But we need to be careful with budget: 81 cells + local search
        # Let's use a coarser grid initially and refine promising cells

        # Strategy:
        # 1. Evaluate grid points (vertices of hypercubes)
        # 2. Identify promising regions based on objective values
        # 3. Perform local simplex search in promising regions
        # 4. Use remaining budget for global refinement

        # Step 1: Create grid points
        # For each dimension, create grid points
        grid_points = []
        for i in range(grid_per_dim):
            for j in range(grid_per_dim):
                for k in range(grid_per_dim):
                    for l in range(grid_per_dim):
                        x = np.array([
                            lo[0] + (hi[0] - lo[0]) * i / (grid_per_dim - 1),
                            lo[1] + (hi[1] - lo[1]) * j / (grid_per_dim - 1),
                            lo[2] + (hi[2] - lo[2]) * k / (grid_per_dim - 1),
                            lo[3] + (hi[3] - lo[3]) * l / (grid_per_dim - 1),
                        ])
                        grid_points.append(x)

        grid_points = np.array(grid_points)
        n_grid = len(grid_points)

        # Evaluate grid points
        F_grid = np.array([problem(x) for x in grid_points])
        used = n_grid

        # Step 2: Identify non-dominated points from grid
        # A point is non-dominated if no other point is better in all objectives
        def is_nondominated(F):
            n = len(F)
            nondom = np.ones(n, dtype=bool)
            for i in range(n):
                if not nondom[i]:
                    continue
                for j in range(n):
                    if i == j:
                        continue
                    if not nondom[j]:
                        continue
                    # Check if j dominates i
                    if np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                        nondom[i] = False
                        break
            return nondom

        nondom_mask = is_nondominated(F_grid)
        nondom_X = grid_points[nondom_mask]
        nondom_F = F_grid[nondom_mask]

        # Step 3: Local simplex search around promising (non-dominated) points
        # Use Nelder-Mead style simplex for each promising point
        # Budget allocation: spend remaining budget on local searches

        remaining_budget = budget - used
        if remaining_budget <= 0:
            # No budget left, return what we have
            return np.vstack([grid_points])

        # Select top non-dominated points for local search
        # Sort by sum of objectives (lower is better)
        if len(nondom_X) > 0:
            sums = np.sum(nondom_F, axis=1)
            order = np.argsort(sums)
            # Take top k for local search
            k_local = min(len(nondom_X), max(1, remaining_budget // 20))  # ~20 evals per local search
            local_starts = nondom_X[order[:k_local]]
        else:
            # No non-dominated points, use grid points
            sums = np.sum(F_grid, axis=1)
            order = np.argsort(sums)
            k_local = min(n_grid, max(1, remaining_budget // 20))
            local_starts = grid_points[order[:k_local]]

        # Perform local simplex search
        best_X = list(nondom_X)
        best_F = list(nondom_F)

        def simplex_search(start, problem, lo, hi, budget, deadline, t0):
            """Nelder-Mead simplex search for multi-objective (minimize sum as surrogate)."""
            dim = len(start)
            # Initialize simplex
            simplex = [start.copy()]
            f0 = problem(start)
            used = 1
            # Add perturbation points
            for i in range(dim):
                x = start.copy()
                step = 0.1 * (hi[i] - lo[i])
                x[i] += step
                x = np.clip(x, lo, hi)
                simplex.append(x)
                used += 1

            # Evaluate simplex
            F_simplex = [f0] + [problem(x) for x in simplex[1:]]
            used += dim

            # Nelder-Mead iterations
            alpha = 1.0
            gamma = 2.0
            rho = 0.5
            sigma = 0.5

            max_iter = budget - used
            for _ in range(max_iter):
                if time.time() > deadline:
                    break

                # Sort by sum of objectives
                sums = np.array([np.sum(f) for f in F_simplex])
                order = np.argsort(sums)
                simplex = [simplex[i] for i in order]
                F_simplex = [F_simplex[i] for i in order]

                # Centroid of all but worst
                centroid = np.mean(simplex[:-1], axis=0)
                worst = simplex[-1]
                worst_f = F_simplex[-1]

                # Reflection
                x_ref = centroid + alpha * (centroid - worst)
                x_ref = np.clip(x_ref, lo, hi)
                f_ref = problem(x_ref)
                used += 1

                if np.sum(f_ref) < np.sum(F_simplex[0]):
                    # Expansion
                    x_exp = centroid + gamma * (x_ref - centroid)
                    x_exp = np.clip(x_exp, lo, hi)
                    f_exp = problem(x_exp)
                    used += 1
                    if np.sum(f_exp) < np.sum(f_ref):
                        simplex[-1] = x_exp
                        F_simplex[-1] = f_exp
                    else:
                        simplex[-1] = x_ref
                        F_simplex[-1] = f_ref
                elif np.sum(f_ref) < np.sum(F_simplex[-2]):
                    simplex[-1] = x_ref
                    F_simplex[-1] = f_ref
                else:
                    # Contraction
                    x_con = centroid + rho * (worst - centroid)
                    x_con = np.clip(x_con, lo, hi)
                    f_con = problem(x_con)
                    used += 1
                    if np.sum(f_con) < np.sum(worst_f):
                        simplex[-1] = x_con
                        F_simplex[-1] = f_con
                    else:
                        # Shrink
                        for i in range(1, len(simplex)):
                            simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                            simplex[i] = np.clip(simplex[i], lo, hi)
                            F_simplex[i] = problem(simplex[i])
                            used += 1
                            if used > budget:
                                break

            # Return best point from simplex
            sums = np.array([np.sum(f) for f in F_simplex])
            best_idx = np.argmin(sums)
            return simplex[best_idx], F_simplex[best_idx], used

        for start in local_starts:
            if time.time() > deadline:
                break
            if used >= budget:
                break

            # Allocate budget for this local search
            alloc = min(remaining_budget - (used - n_grid), 30)
            if alloc < 5:
                continue

            x_best, f_best, local_used = simplex_search(
                start, problem, lo, hi, alloc, deadline, t0
            )
            used += local_used
            best_X.append(x_best)
            best_F.append(f_best)

        # Step 4: Use remaining budget for global random refinement
        remaining = budget - used
        if remaining > 0 and time.time() < deadline:
            rng = np.random.default_rng(42)
            for _ in range(min(remaining, 50)):
                if time.time() > deadline:
                    break
                x = lo + (hi - lo) * rng.random(dim)
                f = problem(x)
                best_X.append(x)
                best_F.append(f)
                used += 1

        # Final non-dominated filtering
        if len(best_X) == 0:
            return np.array([lo + (hi - lo) * 0.5])

        best_X = np.array(best_X)
        best_F = np.array(best_F)

        # Filter to only points strictly better than reference [3000, 0.06]
        ref = np.array([3000.0, 0.06])
        valid_mask = np.all(best_F < ref, axis=1)
        if np.any(valid_mask):
            best_X = best_X[valid_mask]
            best_F = best_F[valid_mask]

        # Final non-dominated selection
        if len(best_X) > 0:
            final_mask = is_nondominated(best_F)
            if np.any(final_mask):
                best_X = best_X[final_mask]
                best_F = best_F[final_mask]

        return best_X
# EVOLVE-BLOCK-END
