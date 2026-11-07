"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


class Optimizer:
    """NSGA-II style multi-objective optimizer for WFG9."""

    def __init__(self, budget, dim, n_obj):
        self.budget, self.dim, self.n_obj = budget, dim, n_obj

    def __call__(self, problem, bounds):
        rng = np.random.default_rng(0)
        lo = np.array([b[0] for b in bounds], dtype=float)
        hi = np.array([b[1] for b in bounds], dtype=float)
        span = hi - lo
        n = self.dim
        budget = self.budget

        # Population size: use a reasonable chunk of the budget
        pop_size = min(60, max(10, budget // 5))
        
        # Initialize population
        X = lo + span * rng.random((pop_size, n))
        F = np.array([problem(x) for x in X], dtype=float)
        used = pop_size

        def fast_non_dominated_sort(F_arr):
            """Returns list of fronts (each front is a list of indices)."""
            m = F_arr.shape[0]
            k = F_arr.shape[1]
            S = [[] for _ in range(m)]
            n_dom = np.zeros(m, dtype=int)
            fronts = []
            current_front = []
            for p in range(m):
                for q in range(m):
                    if p == q:
                        continue
                    if np.all(F_arr[p] <= F_arr[q]) and np.any(F_arr[p] < F_arr[q]):
                        S[p].append(q)
                        n_dom[p] += 1
                    elif np.all(F_arr[q] <= F_arr[p]) and np.any(F_arr[q] < F_arr[p]):
                        n_dom[q] += 1
                if n_dom[p] == 0:
                    current_front.append(p)
            if current_front:
                fronts.append(current_front)
            i = 0
            while fronts[i]:
                next_front = []
                for p in fronts[i]:
                    for q in S[p]:
                        n_dom[q] -= 1
                        if n_dom[q] == 0:
                            next_front.append(q)
                i += 1
                if next_front:
                    fronts.append(next_front)
            return fronts

        def crowding_distance(F_arr, indices):
            """Returns crowding distance for each index in indices."""
            m = len(indices)
            if m <= 2:
                return np.ones(m) * np.inf
            cd = np.zeros(m)
            F_sub = F_arr[indices]
            k = F_sub.shape[1]
            for j in range(k):
                obj = F_sub[:, j]
                obj_min = obj.min()
                obj_max = obj.max()
                if obj_max - obj_min < 1e-15:
                    continue
                sorted_idx = np.argsort(obj)
                cd[sorted_idx[0]] = np.inf
                cd[sorted_idx[-1]] = np.inf
                for idx in range(1, m - 1):
                    cd[sorted_idx[idx]] += (obj[sorted_idx[idx + 1]] - obj[sorted_idx[idx - 1]]) / (obj_max - obj_min)
            return cd

        while used < budget:
            # Create offspring
            # Use binary tournament selection based on rank and crowding distance
            fronts = fast_non_dominated_sort(F)
            rank = np.zeros(len(F), dtype=int)
            for r, front in enumerate(fronts):
                for idx in front:
                    rank[idx] = r
            
            # Compute crowding distance per front
            cd = np.zeros(len(F))
            for front in fronts:
                if len(front) > 0:
                    cd_vals = crowding_distance(F, front)
                    for i, idx in enumerate(front):
                        cd[idx] = cd_vals[i]

            def tournament():
                i1 = rng.integers(len(F))
                i2 = rng.integers(len(F))
                if rank[i1] < rank[i2]:
                    return i1
                elif rank[i2] < rank[i1]:
                    return i2
                else:
                    return i1 if cd[i1] > cd[i2] else i2

            # Simulated binary crossover (SBX)
            def sbx(parent1, parent2):
                if rng.random() < 0.5:
                    beta = 1.0
                    for _ in range(n):
                        if rng.random() < 0.5:
                            if rng.random() < 0.5:
                                beta = (2 * rng.random()) ** (1 / (2 + 1))
                            else:
                                beta = (1 / (2 * (1 - rng.random()))) ** (1 / (2 + 1))
                        child1 = 0.5 * ((1 + beta) * parent1 + (1 - beta) * parent2)
                        child2 = 0.5 * ((1 - beta) * parent1 + (1 + beta) * parent2)
                        return np.clip(child1, lo, hi), np.clip(child2, lo, hi)
                    return np.clip(parent1, lo, hi), np.clip(parent2, lo, hi)
                else:
                    return np.clip(parent1, lo, hi), np.clip(parent2, lo, hi)

            # Polynomial mutation
            def mutate(x):
                for i in range(n):
                    if rng.random() < 1.0 / n:
                        delta = rng.random()
                        if delta < 0.5:
                            delta = (2 * delta) ** (1 / (2 + 20)) - 1
                        else:
                            delta = 1 - (2 * (1 - delta)) ** (1 / (2 + 20))
                        x[i] += delta * (hi[i] - lo[i])
                return np.clip(x, lo, hi)

            # Generate offspring
            n_offspring = pop_size
            offspring_X = []
            for _ in range(n_offspring):
                p1 = tournament()
                p2 = tournament()
                c1, c2 = sbx(X[p1], X[p2])
                c1 = mutate(c1)
                c2 = mutate(c2)
                offspring_X.append(c1)
                offspring_X.append(c2)
                used += 2
                if used >= budget:
                    break
            
            if len(offspring_X) == 0:
                break

            offspring_F = np.array([problem(x) for x in offspring_X], dtype=float)
            
            # Combine parents and offspring
            X_comb = np.vstack([X, np.array(offspring_X)])
            F_comb = np.vstack([F, offspring_F])
            
            # Selection: keep best pop_size
            fronts = fast_non_dominated_sort(F_comb)
            selected_indices = []
            for front in fronts:
                if len(selected_indices) + len(front) <= pop_size:
                    selected_indices.extend(front)
                else:
                    remaining = pop_size - len(selected_indices)
                    cd_vals = crowding_distance(F_comb, front)
                    front_arr = np.array(front)
                    sorted_by_cd = front_arr[np.argsort(cd_vals)[::-1]]
                    selected_indices.extend(sorted_by_cd[:remaining].tolist())
                    break
            
            X = X_comb[selected_indices]
            F = F_comb[selected_indices]
            
            if used >= budget:
                break

        return X
# EVOLVE-BLOCK-END
