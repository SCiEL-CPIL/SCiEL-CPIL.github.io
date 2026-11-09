"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-objective optimizer for RE37 using NSGA-II with adaptive mutation."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 290  # 300s limit, 10s safety margin

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        rng = np.random.default_rng(0)

        # Population size: use a reasonable fraction of budget
        pop_size = min(50, max(20, self.budget // 4))
        # Ensure pop_size is even for NSGA-II
        if pop_size % 2 != 0:
            pop_size += 1

        # Initialize population
        X = lo + (hi - lo) * rng.random((pop_size, self.dim))
        F = np.array([problem(x) for x in X])
        used = pop_size

        # NSGA-II helpers
        def fast_non_dominated_sort(F):
            """Fast non-dominated sorting. Returns list of fronts (lists of indices)."""
            n = len(F)
            S = [[] for _ in range(n)]  # domination count
            n_dom = np.zeros(n, dtype=int)  # number of solutions dominated
            fronts = []
            front_idx = np.zeros(n, dtype=int)

            for i in range(n):
                for j in range(n):
                    if i == j:
                        continue
                    # Check if i dominates j
                    if np.all(F[i] <= F[j]) and np.any(F[i] < F[j]):
                        S[i].append(j)
                    elif np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                        n_dom[i] += 1

            # First front
            front = [i for i in range(n) if n_dom[i] == 0]
            if front:
                fronts.append(front)
                for i in front:
                    front_idx[i] = 0

            i = 0
            while front and i < n:
                next_front = []
                for p in front:
                    for q in S[p]:
                        n_dom[q] -= 1
                        if n_dom[q] == 0:
                            front_idx[q] = i + 1
                            next_front.append(q)
                i += 1
                if next_front:
                    fronts.append(next_front)
                front = next_front

            return fronts

        def crowding_distance(F, front_indices):
            """Compute crowding distance for a front."""
            n = len(front_indices)
            if n <= 2:
                return np.full(n, np.inf)

            cd = np.zeros(n)
            m = F.shape[1]  # number of objectives

            for k in range(m):
                # Sort by objective k
                sorted_idx = np.argsort(F[front_indices, k])
                cd[sorted_idx[0]] = np.inf
                cd[sorted_idx[-1]] = np.inf

                f_min = F[front_indices, k][0]
                f_max = F[front_indices, k][-1]
                if f_max == f_min:
                    continue

                for i in range(1, n - 1):
                    cd[sorted_idx[i]] += (
                        F[front_indices, k][sorted_idx[i + 1]] -
                        F[front_indices, k][sorted_idx[i - 1]]
                    ) / (f_max - f_min)

            return cd

        def adaptive_mutation(parent, generation, total_generations):
            """Adaptive mutation with decreasing step size."""
            # Step size decreases over generations
            progress = generation / max(1, total_generations)
            base_step = 0.2 * (1 - 0.5 * progress)
            # Add some randomness to step size
            step = base_step * (0.5 + rng.random())
            child = parent + step * (hi - lo) * rng.standard_normal(self.dim)
            return np.clip(child, lo, hi)

        # Determine number of generations
        # Each generation: select parents, create children, evaluate
        # Budget remaining after init
        remaining = self.budget - used
        # Each generation creates pop_size/2 children (tournament selection from 2 parents)
        # Actually, in NSGA-II, we create a population of same size as parent pop
        # Let's create pop_size children per generation
        children_per_gen = pop_size
        max_generations = remaining // children_per_gen
        if max_generations < 1:
            max_generations = 1

        # Track best hypervolume proxy (sum of objective values, lower is better)
        # We'll track the entire archive of non-dominated points

        archive = []  # list of (x, f) pairs that are non-dominated

        # Add initial population to archive
        for i in range(pop_size):
            archive.append((X[i].copy(), F[i].copy()))

        def update_archive(x, f):
            """Add point to archive if non-dominated."""
            non_dominated = True
            for _, af in archive:
                if np.all(af <= f) and np.any(af < f):
                    non_dominated = False
                    break
            if not non_dominated:
                return
            # Remove dominated points from archive
            new_archive = []
            for ax, af in archive:
                if not (np.all(f <= af) and np.any(f < af)):
                    new_archive.append((ax, af))
            new_archive.append((x.copy(), f.copy()))
            archive.clear()
            archive.extend(new_archive)

        # Initial archive update
        archive.clear()
        for i in range(pop_size):
            update_archive(X[i], F[i])

        # NSGA-II main loop
        for gen in range(max_generations):
            if time.time() > deadline:
                break
            if used >= self.budget:
                break

            # Fast non-dominated sorting
            fronts = fast_non_dominated_sort(F)

            # Build next generation
            next_X = []
            next_F = []
            next_used = used

            # Select parents using tournament selection
            # Create offspring population of size pop_size
            for _ in range(children_per_gen):
                if next_used >= self.budget:
                    break
                if time.time() > deadline:
                    break

                # Tournament selection: pick 2 random parents, choose the one with better rank
                i1, i2 = rng.integers(0, pop_size, size=2)

                # Determine front index for each
                front_i1 = -1
                front_i2 = -1
                for fi, front in enumerate(fronts):
                    if i1 in front:
                        front_i1 = fi
                        break
                    if i2 in front:
                        front_i2 = fi
                        break

                if front_i1 < front_i2:
                    parent = X[i1]
                elif front_i2 < front_i1:
                    parent = X[i2]
                else:
                    # Same front, use crowding distance
                    # Compute crowding distance for this front
                    front_idx = front_i1
                    if front_idx >= 0:
                        cd = crowding_distance(F, fronts[front_idx])
                        # Map indices
                        cd_map = {}
                        for ci, idx in enumerate(fronts[front_idx]):
                            cd_map[idx] = cd[ci]
                        cd_i1 = cd_map.get(i1, 0)
                        cd_i2 = cd_map.get(i2, 0)
                        parent = X[i1] if cd_i1 >= cd_i2 else X[i2]
                    else:
                        parent = X[i1]

                # Mutation
                child = adaptive_mutation(parent, gen, max_generations)

                # Evaluate
                fc = problem(child)
                next_used += 1
                next_X.append(child)
                next_F.append(fc)

            if not next_X:
                break

            next_X = np.array(next_X)
            next_F = np.array(next_F)

            # Combine parent and offspring populations
            combined_X = np.vstack([X, next_X])
            combined_F = np.vstack([F, next_F])

            # Sort combined population
            combined_fronts = fast_non_dominated_sort(combined_F)

            # Select next population
            new_X = []
            new_F = []
            for front in combined_fronts:
                if len(new_X) + len(front) <= pop_size:
                    for idx in front:
                        new_X.append(combined_X[idx])
                        new_F.append(combined_F[idx])
                else:
                    # Need to select from this front based on crowding distance
                    need = pop_size - len(new_X)
                    cd = crowding_distance(combined_F, front)
                    sorted_by_cd = np.argsort(-cd)
                    for si in range(need):
                        idx = front[sorted_by_cd[si]]
                        new_X.append(combined_X[idx])
                        new_F.append(combined_F[idx])
                    break

            X = np.array(new_X)
            F = np.array(new_F)
            used = next_used

            # Update archive
            for i in range(len(X)):
                update_archive(X[i], F[i])

        return X
# EVOLVE-BLOCK-END
