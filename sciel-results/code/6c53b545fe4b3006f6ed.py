"""Multi-objective optimizer on RE21 using NSGA-II."""
# EVOLVE-BLOCK-START
import numpy as np


class Optimizer:
    """NSGA-II multi-objective evolutionary optimizer."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        rng = np.random.default_rng(0)
        lo = np.array([b[0] for b in bounds], dtype=float)
        hi = np.array([b[1] for b in bounds], dtype=float)
        span = hi - lo

        # Population size: use a good fraction of budget
        pop_size = int(min(40, max(10, self.budget // 5)))
        max_iter = max(1, (self.budget - pop_size) // (2 * pop_size))

        # Initialize population
        X = lo + span * rng.random((pop_size, self.dim))
        F = np.array([problem(x) for x in X], dtype=float)
        used = pop_size

        # NSGA-II helper functions
        def non_dominated_sort(F):
            """Return list of fronts (each a list of indices)."""
            n = len(F)
            S = [set() for _ in range(n)]
            n_count = [0] * n
            rank = [0] * n
            fronts = []
            for i in range(n):
                for j in range(n):
                    if i == j:
                        continue
                    # i dominates j?
                    if np.all(F[i] <= F[j]) and np.any(F[i] < F[j]):
                        S[i].add(j)
                    elif np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                        n_count[i] += 1
                if n_count[i] == 0:
                    rank[i] = 0
                    fronts.append([i])
            k = 0
            while k < len(fronts) and fronts[k]:
                current_front = fronts[k]
                next_front = []
                for i in current_front:
                    for j in S[i]:
                        n_count[j] -= 1
                        if n_count[j] == 0:
                            rank[j] = k + 1
                            next_front.append(j)
                k += 1
                if next_front:
                    fronts.append(next_front)
            return fronts

        def crowding_distance(F, front_indices):
            """Compute crowding distance for given front indices."""
            n = len(front_indices)
            if n <= 2:
                return [float('inf')] * n
            cd = [0.0] * n
            for m in range(self.n_obj):
                vals = [F[idx, m] for idx in front_indices]
                sorted_idx = sorted(range(n), key=lambda i: vals[i])
                cd[sorted_idx[0]] = float('inf')
                cd[sorted_idx[-1]] = float('inf')
                val_range = vals[sorted_idx[-1]] - vals[sorted_idx[0]]
                if val_range == 0:
                    continue
                for i in range(1, n - 1):
                    cd[sorted_idx[i]] += (vals[sorted_idx[i + 1]] - vals[sorted_idx[i - 1]]) / val_range
            return cd

        # Main evolution loop
        while used < self.budget:
            # Evaluate current population fitness (already done for initial)
            fronts = non_dominated_sort(F)
            rank = np.zeros(len(F), dtype=int)
            for k, front in enumerate(fronts):
                for idx in front:
                    rank[idx] = k

            # Crowding distance per front
            cd = np.zeros(len(F))
            for front in fronts:
                if front:
                    cds = crowding_distance(F, front)
                    for i, idx in enumerate(front):
                        cd[idx] = cds[i]

            # Create offspring population
            offspring_X = []
            offspring_F = []
            n_offspring = min(pop_size, self.budget - used)

            for _ in range(n_offspring):
                if used >= self.budget:
                    break

                # Tournament selection (size 2) based on rank then crowding
                def better(a, b):
                    if rank[a] < rank[b]:
                        return True
                    if rank[a] > rank[b]:
                        return False
                    return cd[a] > cd[b]

                p1 = rng.integers(len(F))
                p2 = rng.integers(len(F))
                parent1 = p1 if better(p1, p2) else p2
                p3 = rng.integers(len(F))
                p4 = rng.integers(len(F))
                parent2 = p3 if better(p3, p4) else p4

                # SBX-style crossover
                child1 = X[parent1].copy()
                child2 = X[parent2].copy()
                if rng.random() < 0.9:  # crossover probability
                    for d in range(self.dim):
                        if rng.random() < 0.5:
                            # Blend
                            alpha = rng.random()
                            child1[d] = alpha * X[parent1, d] + (1 - alpha) * X[parent2, d]
                            child2[d] = (1 - alpha) * X[parent1, d] + alpha * X[parent2, d]

                # Mutation: Gaussian perturbation
                for d in range(self.dim):
                    if rng.random() < 0.2:
                        child1[d] += 0.1 * span[d] * rng.standard_normal()
                    if rng.random() < 0.2:
                        child2[d] += 0.1 * span[d] * rng.standard_normal()

                child1 = np.clip(child1, lo, hi)
                child2 = np.clip(child2, lo, hi)

                f1 = problem(child1)
                used += 1
                offspring_X.append(child1)
                offspring_F.append(f1)

                if used >= self.budget:
                    break

                f2 = problem(child2)
                used += 1
                offspring_X.append(child2)
                offspring_F.append(f2)

            if not offspring_X:
                break

            # Combine parent and offspring
            combined_X = np.vstack([X, np.array(offspring_X)])
            combined_F = np.vstack([F, np.array(offspring_F)])
            n_combined = len(combined_F)

            # Non-dominated sort combined
            combined_fronts = non_dominated_sort(combined_F)
            combined_rank = np.zeros(n_combined, dtype=int)
            for k, front in enumerate(combined_fronts):
                for idx in front:
                    combined_rank[idx] = k

            # Select next generation
            next_X = []
            next_F = []
            for front in combined_fronts:
                if len(next_X) + len(front) <= pop_size:
                    for idx in front:
                        next_X.append(combined_X[idx])
                        next_F.append(combined_F[idx])
                else:
                    # Need to truncate this front by crowding distance
                    remaining = pop_size - len(next_X)
                    cds = crowding_distance(combined_F, front)
                    sorted_front = sorted(range(len(front)), key=lambda i: cds[i], reverse=True)
                    for i in sorted_front[:remaining]:
                        idx = front[i]
                        next_X.append(combined_X[idx])
                        next_F.append(combined_F[idx])
                    break

            if next_X:
                X = np.array(next_X)
                F = np.array(next_F)

        return X

# EVOLVE-BLOCK-END
