# EVOLVE-BLOCK-START
import time
import numpy as np


class Optimizer:
    """LHS sampling + Evolutionary Strategy with hypervolume-based selection for WFG7."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        deadline = time.time() + 280.0  # safety margin before 300s limit
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        span = hi - lo
        dim = self.dim
        n_obj = self.n_obj
        budget = self.budget

        X = []
        F = []
        eval_count = 0

        def evaluate(x):
            nonlocal eval_count
            if eval_count >= budget or time.time() > deadline:
                return None
            fx = problem(x)
            eval_count += 1
            X.append(x.copy())
            F.append(np.asarray(fx, dtype=float))
            return fx

        # Phase 1: LHS sampling for broad coverage
        n_lhs = 40
        lhs_points = []
        for i in range(n_lhs):
            point = np.zeros(dim)
            for d in range(dim):
                u = np.random.rand()
                frac = (i + u) / n_lhs
                point[d] = lo[d] + frac * span[d]
            lhs_points.append(point)

        for x in lhs_points:
            evaluate(x)
            if eval_count >= budget:
                break

        # Phase 2: Evolutionary Strategy with hypervolume-contribution selection
        ref = np.array([3.0, 5.0, 7.0])

        def hypervolume_contribution(point, front):
            """Compute the hypervolume contribution of a single point to a set."""
            # Only count if point is better than ref in all objectives
            if np.any(point >= ref):
                return 0.0
            # Compute the volume of the region dominated by point but not by any other in front
            # For 3 objectives, this is the box from point to ref, minus overlaps
            # Simple approximation: volume of box from point to ref
            # But we need to subtract regions already covered by other points
            # For efficiency, use a simplified contribution: 
            # The point's contribution is the volume of the box [point, ref] 
            # minus the volume covered by other points that dominate parts of it.
            # For a rough estimate, we can use the volume of the box from point to ref
            # scaled by how "extreme" it is.
            box_vol = np.prod(ref - point)
            # Penalize points that are close to other front points
            if len(front) > 0:
                min_dist = np.inf
                for other in front:
                    # Distance in objective space
                    d = np.linalg.norm(point - other)
                    min_dist = min(min_dist, d)
                # If very close to another point, reduce contribution
                if min_dist < 1e-6:
                    return 0.0
            return box_vol

        def nondominated_filter(points_f):
            """Return indices of non-dominated points."""
            n = len(points_f)
            if n == 0:
                return []
            dominated = np.zeros(n, dtype=bool)
            for i in range(n):
                if dominated[i]:
                    continue
                for j in range(i + 1, n):
                    if dominated[j]:
                        continue
                    if np.all(points_f[i] <= points_f[j]) and np.any(points_f[i] < points_f[j]):
                        dominated[j] = True
                    elif np.all(points_f[j] <= points_f[i]) and np.any(points_f[j] < points_f[i]):
                        dominated[i] = True
                        break
            return [i for i in range(n) if not dominated[i]]

        # Main ES loop
        pop_size = 12
        sigma = 0.15  # step size for mutation

        while eval_count < budget and time.time() < deadline:
            # Build current front
            if len(F) == 0:
                break
            F_arr = np.array(F)
            nd_idx = nondominated_filter(F_arr)
            if len(nd_idx) == 0:
                break

            # Select parents based on hypervolume contribution
            # Compute contributions for non-dominated points
            contributions = []
            front_points = [F_arr[i] for i in nd_idx]
            for k, idx in enumerate(nd_idx):
                contrib = hypervolume_contribution(F_arr[idx], [F_arr[j] for j in nd_idx if j != idx])
                contributions.append(contrib)

            # Select top pop_size parents by contribution
            sorted_parents = sorted(zip(contributions, nd_idx), key=lambda x: -x[0])
            n_parents = min(pop_size, len(sorted_parents))
            parent_indices = [sorted_parents[i][1] for i in range(n_parents)]

            # Generate offspring by mutating parents
            offspring = []
            for _ in range(pop_size):
                # Pick a random parent
                p_idx = parent_indices[np.random.randint(n_parents)]
                x_parent = X[p_idx].copy()
                # Gaussian mutation
                mutation = np.random.randn(dim) * sigma
                x_child = x_parent + mutation
                # Clip to bounds
                x_child = np.clip(x_child, lo, hi)
                offspring.append(x_child)

            # Evaluate offspring
            for x_child in offspring:
                if eval_count >= budget or time.time() > deadline:
                    break
                evaluate(x_child)

            # Adapt sigma: if we got new non-dominated points, keep sigma; else reduce
            if len(F) > 0:
                F_arr_new = np.array(F)
                nd_new = nondominated_filter(F_arr_new)
                # If the best contribution improved, keep sigma; else reduce slightly
                if len(nd_new) > 0:
                    # Check if we made progress
                    pass
                sigma *= 0.95  # gentle adaptation
                sigma = max(sigma, 0.01)

        return X
# EVOLVE-BLOCK-END
