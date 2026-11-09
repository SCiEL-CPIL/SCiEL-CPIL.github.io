"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


class Optimizer:
    """NSGA-II style multi-objective evolutionary optimizer."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        rng = np.random.default_rng(42)
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        n = self.dim
        m = self.n_obj

        # Population size
        pop_size = min(50, self.budget // 4)
        if pop_size < 5:
            pop_size = 5

        # Initialize population
        X = lo + (hi - lo) * rng.random((pop_size, n))
        F = np.array([problem(x) for x in X])
        used = pop_size

        def fast_non_dominated_sort(fitnesses):
            """Return list of fronts, each front is a list of indices."""
            N = len(fitnesses)
            S = [[] for _ in range(N)]
            n_dom = [0] * N
            rank = [0] * N
            fronts = []

            for i in range(N):
                for j in range(N):
                    if i == j:
                        continue
                    # i dominates j if i is better in all and strictly better in at least one
                    if np.all(fitnesses[i] <= fitnesses[j]) and np.any(fitnesses[i] < fitnesses[j]):
                        S[i].append(j)
                        n_dom[i] += 1
                    elif np.all(fitnesses[j] <= fitnesses[i]) and np.any(fitnesses[j] < fitnesses[i]):
                        n_dom[j] += 1

                if n_dom[i] == 0:
                    rank[i] = 0
                    if not fronts or len(fronts) == 0:
                        fronts.append([])
                    fronts[0].append(i)

            i = 0
            while fronts[i]:
                next_front = []
                for p in fronts[i]:
                    for q in S[p]:
                        n_dom[q] -= 1
                        if n_dom[q] == 0:
                            rank[q] = i + 1
                            next_front.append(q)
                i += 1
                if next_front:
                    fronts.append(next_front)

            return fronts

        def crowding_distance(front_indices, fitnesses):
            """Compute crowding distance for a front."""
            N = len(front_indices)
            if N <= 2:
                return [float('inf')] * N

            dist = [0.0] * N
            for m_idx in range(m):
                # Sort by objective m_idx
                sorted_idx = sorted(front_indices, key=lambda k: fitnesses[k][m_idx])
                dist[front_indices.index(sorted_idx[0])] = float('inf')
                dist[front_indices.index(sorted_idx[-1])] = float('inf')
                f_min = fitnesses[sorted_idx[0]][m_idx]
                f_max = fitnesses[sorted_idx[-1]][m_idx]
                if f_max == f_min:
                    continue
                for i in range(1, N - 1):
                    idx = front_indices.index(sorted_idx[i])
                    dist[idx] += (fitnesses[sorted_idx[i + 1]][m_idx] - fitnesses[sorted_idx[i - 1]][m_idx]) / (f_max - f_min)

            return dist

        def tournament_select(pop_X, pop_F, fronts):
            """Select one individual via tournament selection."""
            k = 2
            candidates = [rng.integers(len(pop_X)) for _ in range(k)]
            best = candidates[0]
            for c in candidates[1:]:
                rank_best = next(i for i, f in enumerate(fronts) if best in f)
                rank_c = next(i for i, f in enumerate(fronts) if c in f)
                if rank_c < rank_best:
                    best = c
                elif rank_c == rank_best:
                    # Crowding distance tie-break
                    front = fronts[rank_best]
                    cd = crowding_distance(front, pop_F)
                    cd_best = cd[front.index(best)]
                    cd_c = cd[front.index(c)]
                    if cd_c > cd_best:
                        best = c
            return best

        while used < self.budget:
            # Create offspring
            offspring_X = []
            offspring_F = []
            batch = min(pop_size, self.budget - used)

            fronts = fast_non_dominated_sort(F)

            for _ in range(batch):
                p1 = tournament_select(X, F, fronts)
                p2 = tournament_select(X, F, fronts)
                # Crossover (SBX-like, simplified)
                if rng.random() < 0.9:
                    # Simulated binary crossover approximation
                    child = (X[p1] + X[p2]) / 2.0
                    # Add some mutation
                    child = child + 0.1 * (hi - lo) * rng.standard_normal(n)
                else:
                    child = X[p1].copy()
                    child = child + 0.1 * (hi - lo) * rng.standard_normal(n)
                child = np.clip(child, lo, hi)
                offspring_X.append(child)
                offspring_F.append(problem(child))
                used += 1

            offspring_F = np.array(offspring_F)

            # Combine parent and offspring
            combined_X = np.vstack([X, np.array(offspring_X)])
            combined_F = np.vstack([F, offspring_F])

            # Non-dominated sort on combined
            combined_fronts = fast_non_dominated_sort(combined_F)

            # Select new population
            new_X = []
            new_F = []
            for front in combined_fronts:
                if len(new_X) + len(front) <= pop_size:
                    for idx in front:
                        new_X.append(combined_X[idx])
                        new_F.append(combined_F[idx])
                else:
                    # Need to select from this front based on crowding distance
                    remaining = pop_size - len(new_X)
                    cd = crowding_distance(front, combined_F)
                    # Sort front by crowding distance descending
                    sorted_front = sorted(front, key=lambda i: cd[front.index(i)], reverse=True)
                    for idx in sorted_front[:remaining]:
                        new_X.append(combined_X[idx])
                        new_F.append(combined_F[idx])
                    break

            X = np.array(new_X)
            F = np.array(new_F)

        return X
# EVOLVE-BLOCK-END
