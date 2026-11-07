# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """
    Multi-objective optimizer for WFG9 using NSGA-II with proper
    non-dominated sorting and crowding distance for diversity.
    """

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 290  # safety margin before 330s kill

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])

        # Population size: use a reasonable fraction of budget
        pop_size = 30
        # Ensure we don't exceed budget
        max_generations = self.budget // pop_size
        if max_generations < 1:
            pop_size = self.budget
            max_generations = 1

        rng = np.random.RandomState(42)

        # Initialize population with LHS-like spread
        X = np.zeros((pop_size, self.dim))
        for j in range(self.dim):
            perm = rng.permutation(pop_size)
            X[np.arange(pop_size), j] = lo[j] + (hi[j] - lo[j]) * (perm + rng.uniform(size=pop_size)) / pop_size
        X = np.clip(X, lo, hi)

        used = 0
        F = np.zeros((pop_size, self.n_obj))
        for i in range(pop_size):
            F[i] = problem(X[i])
            used += 1

        # Track all evaluated points for hypervolume
        all_points = [F[i].copy() for i in range(pop_size)]

        def fast_non_dominated_sort(F):
            """NSGA-II non-dominated sorting. Returns list of fronts (each a list of indices)."""
            n = len(F)
            S = [[] for _ in range(n)]  # S[i] = set of indices dominated by i
            n_dom = np.zeros(n, dtype=int)  # number of solutions that dominate i
            rank = np.zeros(n, dtype=int)

            for i in range(n):
                for j in range(n):
                    if i == j:
                        continue
                    # Check if i dominates j
                    if np.all(F[i] <= F[j]) and np.any(F[i] < F[j]):
                        S[i].append(j)
                    elif np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                        n_dom[i] += 1

            front = []
            for i in range(n):
                if n_dom[i] == 0:
                    rank[i] = 0
                    front.append(i)

            current_front = 0
            fronts = []
            while front:
                fronts.append(front)
                next_front = []
                for i in front:
                    for j in S[i]:
                        n_dom[j] -= 1
                        if n_dom[j] == 0:
                            rank[j] = current_front + 1
                            next_front.append(j)
                front = next_front
                current_front += 1

            return fronts, rank

        def crowding_distance(F, indices):
            """Compute crowding distance for a set of indices within F."""
            n = len(indices)
            if n <= 2:
                return np.full(n, np.inf)

            dist = np.zeros(n)
            F_sub = F[indices]

            for m in range(self.n_obj):
                # Sort by objective m
                sorted_idx = np.argsort(F_sub[:, m])
                dist[sorted_idx[0]] = np.inf
                dist[sorted_idx[-1]] = np.inf

                f_min = F_sub[sorted_idx[0], m]
                f_max = F_sub[sorted_idx[-1], m]
                if f_max - f_min < 1e-15:
                    continue

                for i in range(1, n - 1):
                    dist[sorted_idx[i]] += (F_sub[sorted_idx[i + 1], m] - F_sub[sorted_idx[i - 1], m]) / (f_max - f_min)

            return dist

        def tournament_select(pop_indices, ranks, F, n_select, rng):
            """Tournament selection based on rank and crowding distance."""
            selected = []
            for _ in range(n_select):
                idx1 = rng.choice(pop_indices)
                idx2 = rng.choice(pop_indices)

                if ranks[idx1] != ranks[idx2]:
                    winner = idx1 if ranks[idx1] < ranks[idx2] else idx2
                else:
                    # Compute crowding distance for the front containing these two
                    front_members = [i for i in pop_indices if ranks[i] == ranks[idx1]]
                    cd = crowding_distance(F, front_members)
                    pos1 = list(front_members).index(idx1)
                    pos2 = list(front_members).index(idx2)
                    winner = idx1 if cd[pos1] > cd[pos2] else idx2

                selected.append(winner)
            return selected

        # Main evolutionary loop
        for gen in range(max_generations):
            if used >= self.budget:
                break
            if time.time() > deadline - 10:
                break

            # Evaluate current population
            fronts, ranks = fast_non_dominated_sort(F)

            # Create offspring population
            offspring_X = np.zeros((pop_size, self.dim))
            offspring_F = np.zeros((pop_size, self.n_obj))

            for i in range(pop_size):
                if used >= self.budget:
                    break
                if time.time() > deadline - 10:
                    break

                # Select two parents
                p1, p2 = tournament_select(np.arange(pop_size), ranks, F, 2, rng)

                # SBX crossover
                child = np.zeros(self.dim)
                for j in range(self.dim):
                    if rng.random() < 0.9:  # crossover probability
                        x1, x2 = X[p1, j], X[p2, j]
                        if x1 == x2:
                            child[j] = x1
                        else:
                            u = rng.random()
                            beta = 1 + 2 * min(x1, x2) / max(abs(x1 - x2), 1e-15)
                            if u <= 1 / (beta + 1):
                                alpha = (2 * u) ** (1 / (beta + 1))
                            else:
                                alpha = (1 / (2 * (1 - u))) ** (1 / (beta + 1))
                            child[j] = 0.5 * ((1 + alpha) * x1 + (1 - alpha) * x2)
                    else:
                        child[j] = X[p1, j] if rng.random() < 0.5 else X[p2, j]

                # Polynomial mutation
                for j in range(self.dim):
                    if rng.random() < 1.0 / self.dim:  # mutation probability
                        delta = 1.0
                        if rng.random() < 0.5:
                            delta = -1.0
                        u = rng.random()
                        child[j] += delta * (hi[j] - lo[j]) * ((2 * u) ** (1 / (20 + 1)) - 1)

                child = np.clip(child, lo, hi)
                offspring_X[i] = child
                offspring_F[i] = problem(child)
                used += 1
                all_points.append(offspring_F[i].copy())

            # Combine parent and offspring
            combined_X = np.vstack([X, offspring_X])
            combined_F = np.vstack([F, offspring_F])
            combined_size = len(combined_F)

            # Non-dominated sort on combined population
            combined_fronts, combined_ranks = fast_non_dominated_sort(combined_F)

            # Select next generation
            next_X = np.zeros((pop_size, self.dim))
            next_F = np.zeros((pop_size, self.n_obj))
            next_indices = []

            for front in combined_fronts:
                if len(next_indices) + len(front) <= pop_size:
                    next_indices.extend(front)
                else:
                    # Need to trim this front by crowding distance
                    remaining = pop_size - len(next_indices)
                    cd = crowding_distance(combined_F, front)
                    sorted_by_cd = np.argsort(cd)[::-1]  # descending
                    for idx in sorted_by_cd[:remaining]:
                        next_indices.append(front[idx])
                    break

            for i, idx in enumerate(next_indices):
                next_X[i] = combined_X[idx]
                next_F[i] = combined_F[idx]

            X = next_X
            F = next_F

        # Return all evaluated points (for hypervolume calculation)
        return np.array(all_points)
# EVOLVE-BLOCK-END
