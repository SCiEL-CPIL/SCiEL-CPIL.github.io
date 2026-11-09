# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-objective optimizer for WFG7 combining random init, non-dominated sorting,
    and adaptive mutation with deadline guard."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 270.0  # Safety margin before 330s limit

        lo = np.array([b[0] for b in bounds], dtype=np.float64)
        hi = np.array([b[1] for b in bounds], dtype=np.float64)
        rng = np.random.default_rng(0)

        budget = self.budget
        dim = self.dim
        n_obj = self.n_obj

        # Initial population - use a moderate size to balance exploration and exploitation
        pop_size = min(30, budget // 5)
        X = lo + (hi - lo) * rng.random((pop_size, dim))
        F = np.array([problem(x) for x in X], dtype=np.float64)
        used = pop_size

        # Fast non-dominated sort
        def fast_nondominated_sort(F_mat):
            n = len(F_mat)
            S = [[] for _ in range(n)]
            nd = np.zeros(n, dtype=int)
            fronts = [[]]
            for i in range(n):
                for j in range(n):
                    if i == j:
                        continue
                    if np.all(F_mat[i] <= F_mat[j]) and np.any(F_mat[i] < F_mat[j]):
                        S[i].append(j)
                        nd[j] += 1
                    elif np.all(F_mat[j] <= F_mat[i]) and np.any(F_mat[j] < F_mat[i]):
                        nd[i] += 1
                if nd[i] == 0:
                    fronts[0].append(i)
            k = 0
            while fronts[k]:
                next_front = []
                for i in fronts[k]:
                    for j in S[i]:
                        nd[j] -= 1
                        if nd[j] == 0:
                            next_front.append(j)
                k += 1
                fronts.append(next_front)
            if not fronts[-1]:
                fronts.pop()
            return fronts

        # Crowding distance
        def crowding_distance(F_mat, indices):
            if len(indices) <= 2:
                return np.ones(len(indices))
            m = len(indices)
            cd = np.zeros(m)
            F_sub = F_mat[indices]
            for k in range(n_obj):
                order = np.argsort(F_sub[:, k])
                cd[order[0]] = np.inf
                cd[order[-1]] = np.inf
                if F_sub[order[-1], k] - F_sub[order[0], k] > 1e-15:
                    for i in range(1, m - 1):
                        cd[order[i]] += (F_sub[order[i + 1], k] - F_sub[order[i - 1], k]) / (F_sub[order[-1], k] - F_sub[order[0], k])
            return cd

        # Main evolutionary loop
        while used < budget:
            if time.time() > deadline:
                break

            # Get non-dominated fronts
            fronts = fast_nondominated_sort(F)

            # Select parents from the first front (non-dominated)
            if len(fronts[0]) > 0:
                parents = np.array(fronts[0])
            else:
                parents = np.arange(len(F))

            # Select two parents via tournament
            if len(parents) >= 2:
                p1_idx = parents[rng.integers(len(parents))]
                p2_idx = parents[rng.integers(len(parents))]
                p1 = X[p1_idx]
                p2 = X[p2_idx]
            else:
                p1 = X[parents[0]]
                p2 = X[parents[0]]

            # Crossover: blend with random alpha
            if rng.random() < 0.85:
                alpha = rng.random()
                child = alpha * p1 + (1 - alpha) * p2
            else:
                child = p1.copy()

            # Adaptive mutation: step size decreases over time
            progress = used / budget
            base_step = 0.15 * (1.0 - 0.5 * progress)
            step = base_step * (0.5 + rng.random())
            mask = rng.random(dim) < 0.4
            if mask.any():
                child[mask] += step * (hi - lo)[mask] * rng.standard_normal(int(mask.sum()))
            child = np.clip(child, lo, hi)

            fc = problem(child)
            used += 1

            # Replace worst individual: prefer dominated points, then highest sum
            dominated = np.array([np.any(np.all(F <= f, axis=1) & np.any(F < f, axis=1)) for f in F])
            if np.any(dominated):
                dom_indices = np.where(dominated)[0]
                worst = dom_indices[rng.integers(len(dom_indices))]
            else:
                worst = int(np.argmax(np.sum(F, axis=1)))
            X[worst] = child
            F[worst] = fc

        return X
# EVOLVE-BLOCK-END
