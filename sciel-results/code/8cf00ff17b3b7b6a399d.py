"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """NSGA-II multi-objective optimizer for DTLZ4 with time-aware budget management."""

    def __init__(self, budget, dim, n_obj):
        self.budget, self.dim, self.n_obj = budget, dim, n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 290  # Safety margin before 300s limit

        rng = np.random.default_rng(0)
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        span = hi - lo

        # Population size
        pop_size = min(40, self.budget // 2)
        pop_size = max(10, pop_size)

        # Initialize population
        X = lo + span * rng.random((pop_size, self.dim))
        F = np.array([problem(x) for x in X])
        used = pop_size

        def fast_non_dominated_sort(F):
            """Fast non-dominated sort returning ranks."""
            n = len(F)
            ranks = np.zeros(n, dtype=int)
            S = [[] for _ in range(n)]
            n_dominates = np.zeros(n, dtype=int)
            fronts = [[]]

            for i in range(n):
                for j in range(i + 1, n):
                    if np.all(F[i] <= F[j]) and np.any(F[i] < F[j]):
                        S[i].append(j)
                        n_dominates[i] += 1
                    elif np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                        S[j].append(i)
                        n_dominates[j] += 1
                if n_dominates[i] == 0:
                    ranks[i] = 0
                    fronts[0].append(i)

            k = 0
            while fronts[k]:
                next_front = []
                for i in fronts[k]:
                    for j in S[i]:
                        n_dominates[j] -= 1
                        if n_dominates[j] == 0:
                            ranks[j] = k + 1
                            next_front.append(j)
                k += 1
                if next_front:
                    fronts.append(next_front)
            return ranks

        def crowding_distance(F, rank_mask):
            """Compute crowding distance for individuals in a rank."""
            n = len(F)
            dist = np.zeros(n)
            idxs = np.where(rank_mask)[0]
            if len(idxs) <= 2:
                dist[idxs] = np.inf
                return dist
            for m in range(self.n_obj):
                sorted_idxs = idxs[np.argsort(F[idxs, m])]
                dist[sorted_idxs[0]] = np.inf
                dist[sorted_idxs[-1]] = np.inf
                f_min = F[sorted_idxs[0], m]
                f_max = F[sorted_idxs[-1], m]
                if f_max - f_min < 1e-15:
                    continue
                for i in range(1, len(sorted_idxs) - 1):
                    dist[sorted_idxs[i]] += (F[sorted_idxs[i + 1], m] - F[sorted_idxs[i - 1], m]) / (f_max - f_min)
            return dist

        def sbx_crossover(x1, x2, eta=20.0):
            """Simulated binary crossover."""
            child1 = x1.copy()
            child2 = x2.copy()
            for j in range(self.dim):
                if rng.random() < 0.5:
                    if abs(x1[j] - x2[j]) < 1e-14:
                        continue
                    u = rng.random()
                    if u <= 0.5:
                        beta = (2 * u) ** (1.0 / (eta + 1.0))
                    else:
                        beta = (1.0 / (2 * (1 - u))) ** (1.0 / (eta + 1.0))
                    c1 = 0.5 * ((1 + beta) * x1[j] + (1 - beta) * x2[j])
                    c2 = 0.5 * ((1 - beta) * x1[j] + (1 + beta) * x2[j])
                    child1[j] = c1
                    child2[j] = c2
            return child1, child2

        def mutate(x, rate=0.1, eta=20.0):
            """Polynomial mutation."""
            child = x.copy()
            for j in range(self.dim):
                if rng.random() < rate:
                    delta1 = (child[j] - lo[j]) / span[j]
                    delta2 = (hi[j] - child[j]) / span[j]
                    u = rng.random()
                    if u < 0.5:
                        xy = (2 * u) ** (1.0 / (eta + 1.0))
                        val = 1 - xy
                    else:
                        xy = (2 * (1 - u)) ** (1.0 / (eta + 1.0))
                        val = xy - 1
                    child[j] += span[j] * val
                    child[j] = np.clip(child[j], lo[j], hi[j])
            return child

        # Main NSGA-II loop
        while used < self.budget:
            if time.time() > deadline:
                break

            # Evaluate ranks
            ranks = fast_non_dominated_sort(F)

            # Create offspring
            offspring_X = []
            offspring_F = []
            while len(offspring_X) < pop_size:
                # Select parents using tournament selection
                p1 = self._tournament_select(F, ranks)
                p2 = self._tournament_select(F, ranks)

                # Crossover
                if rng.random() < 0.9:
                    c1, c2 = sbx_crossover(p1, p2)
                else:
                    c1, c2 = p1.copy(), p2.copy()

                # Mutation
                c1 = mutate(c1)
                c2 = mutate(c2)

                offspring_X.append(c1)
                offspring_F.append(problem(c1))
                used += 1
                if used >= self.budget or time.time() > deadline:
                    break

                if used < self.budget and time.time() <= deadline:
                    offspring_X.append(c2)
                    offspring_F.append(problem(c2))
                    used += 1

            if not offspring_F:
                break

            offspring_F = np.array(offspring_F)
            offspring_X = np.array(offspring_X)

            # Combine parent and offspring
            combined_X = np.vstack([X, offspring_X])
            combined_F = np.vstack([F, offspring_F])

            # Rank combined population
            combined_ranks = fast_non_dominated_sort(combined_F)

            # Select next generation
            new_X = []
            new_F = []
            for rank in range(self.n_obj + 2):  # Max possible rank
                if len(new_X) >= pop_size:
                    break
                mask = combined_ranks == rank
                count = np.sum(mask)
                if count <= pop_size - len(new_X):
                    new_X.append(combined_X[mask])
                    new_F.append(combined_F[mask])
                else:
                    # Need crowding distance to break ties
                    cd = crowding_distance(combined_F, mask)
                    idxs = np.where(mask)[0]
                    sorted_idxs = idxs[np.argsort(-cd[idxs])]
                    take = pop_size - len(new_X)
                    selected = sorted_idxs[:take]
                    new_X.append(combined_X[selected])
                    new_F.append(combined_F[selected])

            if new_X:
                X = np.vstack(new_X)
                F = np.vstack(new_F)

        return X

    def _tournament_select(self, F, ranks, k=2):
        """Binary tournament selection."""
        idxs = rng.integers(0, len(F), size=k)
        best = idxs[0]
        for i in idxs[1:]:
            if ranks[i] < ranks[best]:
                best = i
            elif ranks[i] == ranks[best]:
                if np.sum(F[i]) < np.sum(F[best]):
                    best = i
        return F[best].copy()
# EVOLVE-BLOCK-END
