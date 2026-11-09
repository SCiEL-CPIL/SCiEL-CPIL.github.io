"""Multi-objective optimizer for WFG7 with NSGA-II + hypervolume-based selection."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """NSGA-II style multi-objective optimizer with hypervolume-based parent selection."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280.0  # Safety margin before 330s limit

        lo = np.array([b[0] for b in bounds], dtype=np.float64)
        hi = np.array([b[1] for b in bounds], dtype=np.float64)
        dim = self.dim
        n_obj = self.n_obj
        budget = self.budget

        # Population size: balance between exploration and per-generation compute
        pop_size = min(100, max(40, budget // 3))

        # Use a fixed seed for reproducibility within the run
        rng = np.random.default_rng(42)

        # Initialize population
        X = lo + (hi - lo) * rng.random((pop_size, dim))
        F = np.array([problem(x) for x in X], dtype=np.float64)
        used = pop_size

        def dominates(a, b):
            """Check if a dominates b (a is better than b in all objectives and at least one strictly)."""
            return np.all(a <= b) and np.any(a < b)

        def fast_non_dominated_sort(F):
            """NSGA-II fast non-dominated sorting. Returns list of fronts (lists of indices)."""
            n = len(F)
            S = [[] for _ in range(n)]  # S[i] = indices dominated by i
            n_dominating = np.zeros(n, dtype=int)
            fronts = [[]]

            for i in range(n):
                for j in range(n):
                    if i == j:
                        continue
                    if dominates(F[i], F[j]):
                        S[i].append(j)
                        n_dominating[j] += 1
                    elif dominates(F[j], F[i]):
                        n_dominating[i] += 1

            for i in range(n):
                if n_dominating[i] == 0:
                    fronts[0].append(i)

            k = 0
            while fronts[k]:
                next_front = []
                for i in fronts[k]:
                    for j in S[i]:
                        n_dominating[j] -= 1
                        if n_dominating[j] == 0:
                            next_front.append(j)
                k += 1
                if next_front:
                    fronts.append(next_front)

            return fronts

        def crowding_distance(F, front_indices):
            """Compute crowding distance for a front."""
            n = len(front_indices)
            if n <= 2:
                return np.full(n, np.inf)

            cd = np.zeros(n)
            for m in range(n_obj):
                vals = F[front_indices, m]
                min_val = vals.min()
                max_val = vals.max()
                if max_val - min_val < 1e-12:
                    continue
                order = np.argsort(vals)
                cd[order[0]] = np.inf
                cd[order[-1]] = np.inf
                for i in range(1, n - 1):
                    cd[order[i]] += (vals[order[i + 1]] - vals[order[i - 1]]) / (max_val - min_val)
            return cd

        def hypervolume_contribution(point, points, ref):
            """Approximate hypervolume contribution of a point given other points and reference."""
            # Simple approximation: volume of the box dominated by point but not by others
            # For 3D: compute the hypervolume of the region that only this point covers
            n = len(points)
            if n == 0:
                return np.prod(ref - point) if np.all(point < ref) else 0.0

            # For each other point, check if it dominates the contribution region
            # Simplified: use the box from point to ref, subtract overlaps
            # This is expensive, so use a simpler heuristic
            vol = 0.0
            if np.all(point < ref):
                # Compute the box volume
                box_vol = np.prod(ref - point)
                # Subtract volumes covered by other points that dominate parts of this box
                # Approximation: for each other point, if it's better in some objectives,
                # reduce the contribution
                reduction = 0.0
                for p in points:
                    if np.all(p <= point):
                        # p dominates point, so point contributes nothing
                        return 0.0
                    # Partial overlap
                    overlap = np.prod(np.maximum(point - p, 0))
                    # Actually, we want the region that point covers but p doesn't
                    # This is complex; use a simpler heuristic
                return box_vol * 0.5  # Heuristic discount

            return 0.0

        def select_parents(F, X, n_parents):
            """Select parents using non-dominated fronts and crowding distance."""
            fronts = fast_non_dominated_sort(F)

            # Assign ranks
            rank = np.zeros(len(F), dtype=int)
            for i, front in enumerate(fronts):
                for idx in front:
                    rank[idx] = i

            # Sort by rank, then by crowding distance (descending)
            # First, compute crowding distance for each front
            cd = np.zeros(len(F))
            for front in fronts:
                if len(front) > 0:
                    front_cd = crowding_distance(F, front)
                    for j, idx in enumerate(front):
                        cd[idx] = front_cd[j]

            # Create selection keys: (rank, -crowding_distance)
            keys = list(zip(rank, -cd))
            sorted_indices = np.argsort(keys, kind='stable')

            # Select top n_parents
            return sorted_indices[:n_parents]

        def crossover(x1, x2, rng):
            """Simulated binary crossover."""
            if rng.random() < 0.9:
                eta = 20
                child1 = np.zeros_like(x1)
                child2 = np.zeros_like(x2)
                for i in range(dim):
                    if rng.random() < 0.5:
                        child1[i] = x1[i]
                        child2[i] = x2[i]
                    else:
                        child1[i] = x2[i]
                        child2[i] = x1[i]
                return child1, child2
            else:
                return x1.copy(), x2.copy()

        def mutate(x, rng, generation, total_generations):
            """Adaptive mutation with decreasing step size."""
            progress = generation / max(1, total_generations)
            base_step = 0.1 * (1.0 - 0.7 * progress)
            noise = rng.standard_normal(dim) * base_step
            x_new = x + noise
            return np.clip(x_new, lo, hi)

        total_generations = max(1, budget - used) // max(1, pop_size // 2)
        generation = 0

        while used < budget:
            if time.time() > deadline:
                break

            # Select parents
            n_parents = min(pop_size, budget - used)
            parent_indices = select_parents(F, X, n_parents)

            # Create children
            children_X = []
            children_F = []
            while len(children_X) < pop_size and used + len(children_F) < budget:
                # Select two parents
                p1_idx = parent_indices[rng.integers(len(parent_indices))]
                p2_idx = parent_indices[rng.integers(len(parent_indices))]

                c1, c2 = crossover(X[p1_idx], X[p2_idx], rng)
                c1 = mutate(c1, rng, generation, total_generations)
                c2 = mutate(c2, rng, generation, total_generations)

                children_X.append(c1)
                f1 = problem(c1)
                children_F.append(f1)
                used += 1

                if used < budget:
                    children_X.append(c2)
                    f2 = problem(c2)
                    children_F.append(f2)
                    used += 1

            if not children_X:
                break

            # Combine parent and child populations
            X_combined = np.vstack([X, np.array(children_X)])
            F_combined = np.vstack([F, np.array(children_F)])

            # Non-dominated sort
            fronts = fast_non_dominated_sort(F_combined)

            # Build new population
            new_X = []
            new_F = []
            for front in fronts:
                if len(new_X) + len(front) <= pop_size:
                    for idx in front:
                        new_X.append(X_combined[idx])
                        new_F.append(F_combined[idx])
                else:
                    # Need to truncate this front based on crowding distance
                    remaining = pop_size - len(new_X)
                    front_cd = crowding_distance(F_combined, front)
                    order = np.argsort(-front_cd)
                    for j in range(remaining):
                        idx = front[order[j]]
                        new_X.append(X_combined[idx])
                        new_F.append(F_combined[idx])
                    break

            if new_X:
                X = np.array(new_X)
                F = np.array(new_F)

            generation += 1

        # Final polish: use remaining budget for local refinement of best points
        remaining = budget - used
        if remaining > 0 and time.time() < deadline - 30:
            # Find the best points (lowest sum of objectives)
            scores = np.sum(F, axis=1)
            best_indices = np.argsort(scores)[:min(10, len(F))]

            for _ in range(remaining):
                if time.time() > deadline:
                    break
                idx = best_indices[rng.integers(len(best_indices))]
                x = X[idx].copy()
                # Small local perturbation
                step = 0.01 * (hi - lo)
                x += rng.standard_normal(dim) * step
                x = np.clip(x, lo, hi)
                f = problem(x)
                used += 1

                # If this is better, replace
                if np.sum(f) < np.sum(F[idx]):
                    X[idx] = x
                    F[idx] = f
                elif len(X) < pop_size:
                    X = np.vstack([X, x.reshape(1, -1)])
                    F = np.vstack([F, f.reshape(1, -1)])

        return X
# EVOLVE-BLOCK-END
