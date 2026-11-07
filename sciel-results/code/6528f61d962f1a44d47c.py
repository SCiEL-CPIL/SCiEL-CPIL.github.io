"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-objective optimizer for RE37: Grid init + NSGA-II + weighted local refinement."""

    def __init__(self, budget, dim, n_obj):
        self.budget, self.dim, self.n_obj = budget, dim, n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280  # safety margin before 300s limit

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        dim = self.dim
        n_obj = self.n_obj
        budget = self.budget

        # Use the global numpy RNG (seeded by evaluator via np.random.seed)
        # for diversity across seeds; wrap in a helper for consistent API.
        class _RNG:
            def __init__(self):
                pass
            def permutation(self, n):
                return np.random.permutation(n)
            def random(self, size=None):
                return np.random.random(size)
            def random_integers(self, high, low=None):
                if low is None:
                    low = 0
                return np.random.randint(low, high)
            def standard_normal(self, size=None):
                return np.random.standard_normal(size)
            def choice(self, a, size=None, replace=True):
                return np.random.choice(a, size=size, replace=replace)

        rng = _RNG()

        # Track all evaluated points
        X_all = []
        F_all = []
        eval_count = 0

        def evaluate(x):
            nonlocal eval_count
            x = np.clip(x, lo, hi)
            f = np.array(problem(x))
            X_all.append(x.copy())
            F_all.append(f.copy())
            eval_count += 1
            return f

        def time_ok():
            return (time.time() - start_time) < deadline and eval_count < budget

        # Phase 1: Systematic initialization
        # Combine a coarse grid with LHS for good coverage
        # Grid: 3^4 = 81 points
        grid_size = 3
        grid_points = []
        for i in range(grid_size):
            for j in range(grid_size):
                for k in range(grid_size):
                    for l in range(grid_size):
                        x = np.array([
                            lo[0] + (hi[0] - lo[0]) * i / (grid_size - 1),
                            lo[1] + (hi[1] - lo[1]) * j / (grid_size - 1),
                            lo[2] + (hi[2] - lo[2]) * k / (grid_size - 1),
                            lo[3] + (hi[3] - lo[3]) * l / (grid_size - 1),
                        ])
                        grid_points.append(x)

        # LHS points: 24 points
        lhs_size = 24
        lhs_points = np.zeros((lhs_size, dim))
        for d in range(dim):
            perm = rng.permutation(lhs_size)
            lhs_points[:, d] = (perm + rng.random(lhs_size)) / lhs_size
        lhs_points = lo + (hi - lo) * lhs_points

        # Evaluate all initial points
        for x in grid_points:
            if not time_ok():
                break
            evaluate(x)

        for x in lhs_points:
            if not time_ok():
                break
            evaluate(x)

        if not time_ok():
            return np.array(X_all)

        # Phase 2: NSGA-II style evolutionary loop
        # Maintain a population of size pop_size
        pop_size = 40

        # Select initial population from evaluated points
        # Use non-dominated sort to pick diverse points
        F_current = np.array(F_all)

        def fast_non_dominated_sort(F):
            n = len(F)
            S = [[] for _ in range(n)]
            n_dom = np.zeros(n, dtype=int)
            rank = np.zeros(n, dtype=int)

            for i in range(n):
                for j in range(i + 1, n):
                    if np.all(F[i] <= F[j]) and np.any(F[i] < F[j]):
                        S[i].append(j)
                        n_dom[j] += 1
                    elif np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                        S[j].append(i)
                        n_dom[i] += 1

            fronts = []
            current_front = [i for i in range(n) if n_dom[i] == 0]
            k = 0
            while current_front:
                fronts.append(current_front)
                next_front = []
                for i in current_front:
                    rank[i] = k
                    for j in S[i]:
                        n_dom[j] -= 1
                        if n_dom[j] == 0:
                            next_front.append(j)
                current_front = next_front
                k += 1

            return fronts, rank

        def crowding_distance(F, front):
            if len(front) <= 2:
                return {i: float('inf') for i in front}

            dist = {i: 0.0 for i in front}
            for m in range(n_obj):
                vals = sorted([(F[i][m], i) for i in front])
                dist[vals[0][1]] = float('inf')
                dist[vals[-1][1]] = float('inf')
                fmin = vals[0][0]
                fmax = vals[-1][0]
                if fmax - fmin < 1e-12:
                    continue
                for k in range(1, len(vals) - 1):
                    dist[vals[k][1]] += (vals[k + 1][0] - vals[k - 1][0]) / (fmax - fmin)
            return dist

        # Initialize population from non-dominated points
        fronts, rank = fast_non_dominated_sort(F_current)
        pop_indices = []
        for front in fronts:
            if len(pop_indices) + len(front) <= pop_size:
                pop_indices.extend(front)
            else:
                cd = crowding_distance(F_current, front)
                sorted_front = sorted(front, key=lambda i: cd.get(i, 0), reverse=True)
                remaining = pop_size - len(pop_indices)
                pop_indices.extend(sorted_front[:remaining])
                break

        if len(pop_indices) < pop_size:
            # Fill with random points
            for i in range(len(pop_indices), min(pop_size, len(F_all))):
                pop_indices.append(i)

        X = np.array([X_all[i] for i in pop_indices])
        F = np.array([F_all[i] for i in pop_indices])

        # Evolutionary loop
        while time_ok():
            # Check if we have budget for at least one child
            if eval_count >= budget - 1:
                break

            # Compute fronts and crowding for selection
            fronts, rank = fast_non_dominated_sort(F)
            cd = {}
            for front in fronts:
                cd.update(crowding_distance(F, front))

            def tournament():
                i = rng.random_integers(len(F))
                j = rng.random_integers(len(F))
                if rank[i] != rank[j]:
                    return i if rank[i] < rank[j] else j
                return i if cd.get(i, 0) > cd.get(j, 0) else j

            # Generate children
            n_children = min(pop_size, budget - eval_count)
            children_X = []
            children_F = []

            for _ in range(n_children):
                if not time_ok():
                    break

                p1 = tournament()
                p2 = tournament()

                # SBX crossover
                if rng.random() < 0.9:
                    child = np.zeros(dim)
                    for d in range(dim):
                        if rng.random() < 0.5:
                            u = rng.random()
                            if u <= 0.5:
                                beta = (2 * u) ** (1 / (2 + 15))
                            else:
                                beta = (1 / (2 * (1 - u))) ** (1 / (2 + 15))
                            if X[p1][d] <= X[p2][d]:
                                child[d] = 0.5 * ((1 + beta) * X[p1][d] + (1 - beta) * X[p2][d])
                            else:
                                child[d] = 0.5 * ((1 + beta) * X[p2][d] + (1 - beta) * X[p1][d])
                        else:
                            child[d] = X[p1][d] if rng.random() < 0.5 else X[p2][d]
                else:
                    child = X[p1].copy()

                # Polynomial mutation
                for d in range(dim):
                    if rng.random() < 0.9:
                        u = rng.random()
                        if u < 0.5:
                            delta = (2 * u) ** (1 / (2 + 20)) - 1
                        else:
                            delta = 1 - (2 * (1 - u)) ** (1 / (2 + 20))
                        child[d] += delta * (hi[d] - lo[d])

                child = np.clip(child, lo, hi)
                fc = evaluate(child)
                children_X.append(child)
                children_F.append(fc)

            if not children_X:
                break

            children_X = np.array(children_X)
            children_F = np.array(children_F)

            # Combine parent and child populations
            X_all_gen = np.vstack([X, children_X])
            F_all_gen = np.vstack([F, children_F])

            # Non-dominated sort on combined population
            fronts_all, rank_all = fast_non_dominated_sort(F_all_gen)

            # Select next generation
            new_X = []
            new_F = []
            for front in fronts_all:
                if len(new_X) + len(front) <= pop_size:
                    for idx in front:
                        new_X.append(X_all_gen[idx])
                        new_F.append(F_all_gen[idx])
                else:
                    remaining = pop_size - len(new_X)
                    cd_front = crowding_distance(F_all_gen, front)
                    sorted_front = sorted(front, key=lambda i: cd_front.get(i, 0), reverse=True)
                    for idx in sorted_front[:remaining]:
                        new_X.append(X_all_gen[idx])
                        new_F.append(F_all_gen[idx])
                    break

            if len(new_X) < pop_size:
                for front in fronts_all:
                    for idx in front:
                        if len(new_X) >= pop_size:
                            break
                        if not any(np.allclose(X_all_gen[idx], nx) for nx in new_X):
                            new_X.append(X_all_gen[idx])
                            new_F.append(F_all_gen[idx])
                    if len(new_X) >= pop_size:
                        break

            X = np.array(new_X[:pop_size])
            F = np.array(new_F[:pop_size])

        # Phase 3: Local refinement using weighted scalarization
        remaining = budget - eval_count
        if remaining > 5 and time_ok():
            # Get non-dominated points from all evaluations
            F_all_arr = np.array(F_all)
            fronts_all, _ = fast_non_dominated_sort(F_all_arr)
            nd_indices = fronts_all[0] if fronts_all else list(range(len(F_all)))

            # Generate weight vectors for different trade-offs
            weights = []
            # Simplex lattice
            for w1 in [0.1, 0.3, 0.5, 0.7]:
                for w2 in [0.1, 0.3, 0.5]:
                    w3 = 1.0 - w1 - w2
                    if w3 >= 0.1:
                        weights.append([w1, w2, w3])

            # Add random weights
            for _ in range(10):
                w = rng.random(n_obj)
                w /= w.sum()
                weights.append(w)

            weights = np.array(weights)

            # For each weight, find best point and refine
            for w in weights:
                if not time_ok():
                    break
                if remaining < 5:
                    break

                # Find the point that minimizes the weighted sum
                weighted_sums = F_all_arr @ w
                best_idx = np.argmin(weighted_sums)
                x0 = X_all[best_idx].copy()
                f0 = F_all[best_idx].copy()

                # Local search: try perturbations in random directions
                step = 0.1 * (hi - lo)
                best_x = x0.copy()
                best_f = f0.copy()

                for ls_iter in range(min(10, remaining)):
                    if not time_ok():
                        break

                    improved = False
                    for _ in range(3):
                        perturbation = rng.standard_normal(dim)
                        new_x = best_x + step * perturbation
                        new_x = np.clip(new_x, lo, hi)
                        new_f = evaluate(new_x)
                        remaining -= 1

                        # Check dominance
                        if np.all(new_f <= best_f) and np.any(new_f < best_f):
                            best_x = new_x
                            best_f = new_f
                            improved = True
                            break
                        # Check weighted sum improvement
                        elif np.dot(new_f, w) < np.dot(best_f, w) - 1e-8:
                            # Check if it's non-dominated w.r.t. existing archive
                            is_dominated = False
                            for f_arch in F_all:
                                if np.all(f_arch <= new_f) and np.any(f_arch < new_f):
                                    is_dominated = True
                                    break
                            if not is_dominated:
                                best_x = new_x
                                best_f = new_f
                                improved = True
                                break

                    if improved:
                        step *= 0.8
                    else:
                        step *= 0.5
                        if step < 0.005 * (hi - lo).min():
                            break

        return np.array(X_all)
# EVOLVE-BLOCK-END
