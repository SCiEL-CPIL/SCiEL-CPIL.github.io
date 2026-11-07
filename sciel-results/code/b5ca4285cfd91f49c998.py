"""Multi-objective optimizer for WFG7 using DE + NSGA-II style selection."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-objective optimizer: DE mutation, non-dominated sorting, crowding distance."""

    def __init__(self, dim, n_obj):
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 270.0  # 300s limit, 30s safety margin
        budget = self.budget
        dim = self.dim
        n_obj = self.n_obj

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        span = hi - lo

        # Population size
        pop_size = min(40, max(10, budget // 5))
        if pop_size > budget:
            pop_size = budget

        # Random initialization
        X = lo + span * np.random.random((pop_size, dim))
        F = np.array([problem(x) for x in X])
        used = pop_size

        def fast_non_dominated_sort(F_arr):
            """Return list of fronts (lists of indices)."""
            n = len(F_arr)
            S = [[] for _ in range(n)]
            nd = [0] * n
            rank = [0] * n
            fronts = []
            for i in range(n):
                for j in range(n):
                    if i == j:
                        continue
                    if np.all(F_arr[i] <= F_arr[j]) and np.any(F_arr[i] < F_arr[j]):
                        S[i].append(j)
                        nd[i] += 0
                        nd[j] += 1
                    elif np.all(F_arr[j] <= F_arr[i]) and np.any(F_arr[j] < F_arr[i]):
                        S[j].append(i)
                        nd[i] += 1
                if nd[i] == 0:
                    rank[i] = 0
            front = []
            for i in range(n):
                if nd[i] == 0:
                    front.append(i)
            k = 0
            while front:
                fronts.append(front)
                next_front = []
                for i in front:
                    for j in S[i]:
                        nd[j] -= 1
                        if nd[j] == 0:
                            rank[j] = k + 1
                            next_front.append(j)
                front = next_front
                k += 1
            return fronts

        def crowding_distance(F_arr, indices):
            """Compute crowding distance for given indices."""
            n = len(indices)
            if n <= 2:
                return [float('inf')] * n
            dist = [0.0] * n
            for m in range(n_obj):
                vals = [F_arr[indices[i], m] for i in range(n)]
                sorted_idx = np.argsort(vals)
                dist_sorted = [0.0] * n
                dist_sorted[0] = float('inf')
                dist_sorted[-1] = float('inf')
                fmin = vals[sorted_idx[0]]
                fmax = vals[sorted_idx[-1]]
                rng = fmax - fmin
                if rng < 1e-12:
                    continue
                for i in range(1, n - 1):
                    dist_sorted[i] = (vals[sorted_idx[i + 1]] - vals[sorted_idx[i - 1]]) / rng
                for i in range(n):
                    dist[i] += dist_sorted[i]
            return dist

        def select_parents(F_arr, n_parents):
            """Binary tournament selection on non-dominated fronts."""
            fronts = fast_non_dominated_sort(F_arr)
            # Assign ranks
            rank_arr = np.zeros(len(F_arr), dtype=int)
            for fi, front in enumerate(fronts):
                for idx in front:
                    rank_arr[idx] = fi
            # Crowding distance per front
            cd = np.zeros(len(F_arr))
            for front in fronts:
                if len(front) > 2:
                    cds = crowding_distance(F_arr, front)
                    for i, idx in enumerate(front):
                        cd[idx] = cds[i]
                else:
                    for idx in front:
                        cd[idx] = float('inf')
            # Binary tournament
            parents = []
            for _ in range(n_parents):
                i = np.random.randint(len(F_arr))
                j = np.random.randint(len(F_arr))
                if rank_arr[i] < rank_arr[j]:
                    parents.append(i)
                elif rank_arr[j] < rank_arr[i]:
                    parents.append(j)
                else:
                    parents.append(i if cd[i] > cd[j] else j)
            return parents

        # Main loop
        while used < budget:
            if time.time() > deadline:
                break

            # DE mutation: DE/rand/1
            n_children = min(pop_size, budget - used)
            # Check deadline for batch
            if time.time() + 0.5 > deadline and n_children > 1:
                n_children = 1

            parents_idx = select_parents(F, n_children)
            children_X = np.zeros((n_children, dim))
            children_F = np.zeros((n_children, n_obj))

            for c in range(n_children):
                if time.time() > deadline:
                    break
                p1 = parents_idx[c]
                p2 = np.random.randint(pop_size)
                p3 = np.random.randint(pop_size)
                while p2 == p1 or p3 == p1:
                    p2 = np.random.randint(pop_size)
                    p3 = np.random.randint(pop_size)

                # DE/rand/1 mutation
                CR = 0.7
                F_mut = 0.8
                v = X[p1] + F_mut * (X[p2] - X[p3])
                # Binary crossover
                mask = np.random.random(dim) < CR
                if not np.any(mask):
                    mask[np.random.randint(dim)] = True
                child = np.where(mask, v, X[p1])
                child = np.clip(child, lo, hi)
                children_X[c] = child
                children_F[c] = problem(child)
                used += 1

            # Combine and select next generation
            combined_X = np.vstack([X, children_X])
            combined_F = np.vstack([F, children_F])
            n_combined = len(combined_F)

            fronts = fast_non_dominated_sort(combined_F)
            new_X = []
            new_F = []
            for front in fronts:
                if len(new_X) + len(front) <= pop_size:
                    for idx in front:
                        new_X.append(combined_X[idx])
                        new_F.append(combined_F[idx])
                else:
                    remaining = pop_size - len(new_X)
                    cds = crowding_distance(combined_F, front)
                    sorted_front = sorted(range(len(front)), key=lambda i: cds[i], reverse=True)
                    for i in range(remaining):
                        idx = front[sorted_front[i]]
                        new_X.append(combined_X[idx])
                        new_F.append(combined_F[idx])
                    break

            X = np.array(new_X)
            F = np.array(new_F)

        return X
# EVOLVE-BLOCK-END
