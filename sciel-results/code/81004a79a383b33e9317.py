"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """NSGA-II style multi-objective optimizer with deadline guard."""

    def __init__(self, budget, dim, n_obj):
        self.budget, self.dim, self.n_obj = budget, dim, n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 310  # Safety margin before 330s limit

        rng = np.random.default_rng(0)
        lo = np.array([b[0] for b in bounds], dtype=float)
        hi = np.array([b[1] for b in bounds], dtype=float)
        span = hi - lo

        # Population size
        pop_size = min(60, self.budget)
        if pop_size < 10:
            pop_size = 10

        # Initial population
        X = lo + span * rng.random((pop_size, self.dim))
        F = np.array([problem(x) for x in X])
        used = pop_size

        # Reference point for dominance checks (use a large value)
        ref = np.full(self.n_obj, 1e6)

        def fast_non_dominated_sort(F_arr):
            """Returns list of fronts, each a list of indices."""
            n = len(F_arr)
            S = [[] for _ in range(n)]  # S[i] = indices dominated by i
            n_doms = [0] * n  # number of solutions that dominate i
            fronts = [[]]

            for i in range(n):
                for j in range(n):
                    if i == j:
                        continue
                    # Check if i dominates j
                    if np.all(F_arr[i] <= F_arr[j]) and np.any(F_arr[i] < F_arr[j]):
                        S[i].append(j)
                        n_doms[j] += 1
                    elif np.all(F_arr[j] <= F_arr[i]) and np.any(F_arr[j] < F_arr[i]):
                        n_doms[i] += 1

            for i in range(n):
                if n_doms[i] == 0:
                    fronts[0].append(i)

            k = 0
            while len(fronts[k]) > 0:
                next_front = []
                for i in fronts[k]:
                    for j in S[i]:
                        n_doms[j] -= 1
                        if n_doms[j] == 0:
                            next_front.append(j)
                k += 1
                if next_front:
                    fronts.append(next_front)
                else:
                    break

            return fronts

        def crowding_distance(F_arr, indices):
            """Compute crowding distances for given indices."""
            if len(indices) <= 2:
                return np.full(len(indices), np.inf)
            m = self.n_obj
            distances = np.zeros(len(indices))
            for m_i in range(m):
                f_vals = F_arr[indices, m_i]
                sorted_idx = np.argsort(f_vals)
                distances[sorted_idx[0]] = np.inf
                distances[sorted_idx[-1]] = np.inf
                f_range = f_vals[sorted_idx[-1]] - f_vals[sorted_idx[0]]
                if f_range > 1e-10:
                    for i in range(1, len(sorted_idx) - 1):
                        distances[sorted_idx[i]] += (f_vals[sorted_idx[i + 1]] - f_vals[sorted_idx[i - 1]]) / f_range
            return distances

        def tournament_select(F_arr, fronts, crowding, k=2):
            """Tournament selection based on rank and crowding distance."""
            n = len(F_arr)
            selected = []
            for _ in range(k):
                idx = rng.integers(n)
                selected.append(idx)
            # Compare by rank first, then crowding
            best = selected[0]
            for idx in selected[1:]:
                rank_best = fronts.index(next(f for f in fronts if best in f))
                rank_idx = fronts.index(next(f for f in fronts if idx in f))
                if rank_idx < rank_best:
                    best = idx
                elif rank_idx == rank_best:
                    # Compare crowding distance
                    crowding_best = next(crowding[f] for f in fronts if best in f)[fronts.index(next(f for f in fronts if best in f)).index(best) if best in next(f for f in fronts if best in f) else 0]
                    # Simpler: just pick randomly if same rank
                    if rng.random() < 0.5:
                        best = idx
            return best

        # Simpler tournament: just pick two random, compare rank then crowding
        def select_one(F_arr, fronts, crowd_map):
            i = rng.integers(len(F_arr))
            j = rng.integers(len(F_arr))
            rank_i = next(k for k, f in enumerate(fronts) if i in f)
            rank_j = next(k for k, f in enumerate(fronts) if j in f)
            if rank_i < rank_j:
                return i
            elif rank_j < rank_i:
                return j
            else:
                # Same rank, use crowding
                ci = crowd_map.get(i, 0)
                cj = crowd_map.get(j, 0)
                return i if ci >= cj else j

        while used < self.budget:
            # Check deadline
            if time.time() > deadline:
                break

            # Compute fronts
            fronts = fast_non_dominated_sort(F)
            
            # Compute crowding distances per front
            crowd_map = {}
            for front in fronts:
                if len(front) > 0:
                    cd = crowding_distance(F, front)
                    for i, idx in enumerate(front):
                        crowd_map[idx] = cd[i]

            # Generate offspring
            offspring_X = []
            for _ in range(pop_size):
                # Select two parents
                p1 = select_one(F, fronts, crowd_map)
                p2 = select_one(F, fronts, crowd_map)
                
                # SBX crossover
                if rng.random() < 0.9:
                    child = np.zeros(self.dim)
                    for d in range(self.dim):
                        if rng.random() < 0.5:
                            beta = 1.0 + 2.0 * min(X[p1, d], X[p2, d]) / max(1e-10, abs(X[p1, d] - X[p2, d])) if X[p1, d] != X[p2, d] else 1.0
                            beta = max(beta, 1e-6)
                            if rng.random() < 0.5:
                                child[d] = 0.5 * ((1 + 1.0 / beta) * X[p1, d] + (1 - 1.0 / beta) * X[p2, d])
                            else:
                                child[d] = 0.5 * ((1 + beta) * X[p1, d] + (1 - beta) * X[p2, d])
                        else:
                            beta = 1.0 + 2.0 * min(X[p2, d], X[p1, d]) / max(1e-10, abs(X[p1, d] - X[p2, d])) if X[p1, d] != X[p2, d] else 1.0
                            beta = max(beta, 1e-6)
                            if rng.random() < 0.5:
                                child[d] = 0.5 * ((1 + 1.0 / beta) * X[p2, d] + (1 - 1.0 / beta) * X[p1, d])
                            else:
                                child[d] = 0.5 * ((1 + beta) * X[p2, d] + (1 - beta) * X[p1, d])
                else:
                    child = X[p1].copy()

                # Polynomial mutation
                if rng.random() < 1.0 / self.dim:
                    d = rng.integers(self.dim)
                    if child[d] > lo[d]:
                        delta1 = (child[d] - lo[d]) / (hi[d] - lo[d])
                    else:
                        delta1 = (hi[d] - child[d]) / (hi[d] - lo[d])
                    if rng.random() < 0.5:
                        xy = 1.0 + 2.0 * (1.0 - delta1)
                    else:
                        xy = 1.0 + 2.0 * delta1
                    gz = (rng.random() * 2.0) ** (1.0 / (20.0 + 1.0))
                    if rng.random() < 0.5:
                        deltaq = (1.0 - gz) ** 20.0
                    else:
                        deltaq = gz ** 20.0
                    if child[d] > lo[d]:
                        child[d] = child[d] - deltaq * (hi[d] - lo[d])
                    else:
                        child[d] = child[d] + deltaq * (hi[d] - lo[d])

                child = np.clip(child, lo, hi)
                offspring_X.append(child)

            # Evaluate offspring
            offspring_F = np.array([problem(x) for x in offspring_X])
            used += pop_size

            # Combine parent and offspring
            combined_X = np.vstack([X, np.array(offspring_X)])
            combined_F = np.vstack([F, offspring_F])

            # Selection
            combined_fronts = fast_non_dominated_sort(combined_F)
            
            new_X = []
            new_F = []
            for front in combined_fronts:
                if len(new_X) + len(front) <= pop_size:
                    for idx in front:
                        new_X.append(combined_X[idx])
                        new_F.append(combined_F[idx])
                else:
                    # Need to select from this front
                    needed = pop_size - len(new_X)
                    cd = crowding_distance(combined_F, front)
                    sorted_front = np.argsort(-cd)
                    for i in range(needed):
                        idx = front[sorted_front[i]]
                        new_X.append(combined_X[idx])
                        new_F.append(combined_F[idx])
                    break

            X = np.array(new_X)
            F = np.array(new_F)

        return X
# EVOLVE-BLOCK-END
