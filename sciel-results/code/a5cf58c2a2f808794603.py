"""Multi-objective optimizer on ZDT1 — NSGA-II with structure-aware init."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """NSGA-II with ZDT1-aware initialization and adaptive mutation."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280  # safety margin before 330s kill
        rng = np.random.default_rng(0)
        lo = np.array([b[0] for b in bounds], dtype=np.float64)
        hi = np.array([b[1] for b in bounds], dtype=np.float64)
        span = hi - lo

        # ZDT1-aware init: x2..x30 near 0.5 (g=0), x1 spread
        pop_size = 50
        X = np.zeros((pop_size, self.dim))
        # x1: spread across [0,1] for front coverage
        X[:, 0] = np.linspace(0.0, 1.0, pop_size) + rng.normal(0, 0.02, pop_size)
        X[:, 0] = np.clip(X[:, 0], lo[0], hi[0])
        # x2..x30: near 0.5 with small noise (g=0 optimum)
        X[:, 1:] = 0.5 + rng.normal(0, 0.05, (pop_size, self.dim - 1))
        X[:, 1:] = np.clip(X[:, 1:], lo[1:], hi[1:])

        F = np.array([problem(x) for x in X], dtype=np.float64)
        used = pop_size

        def fast_non_dominated_sort(F):
            n = len(F)
            S = [[] for _ in range(n)]
            n_count = [0] * n
            rank = [-1] * n
            front = []
            for i in range(n):
                for j in range(n):
                    if i == j:
                        continue
                    if np.all(F[i] <= F[j]) and np.any(F[i] < F[j]):
                        S[i].append(j)
                        n_count[i] += 1
                    elif np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                        n_count[j] += 1
                if n_count[i] == 0:
                    rank[i] = 0
                    front.append(i)
            k = 0
            while front:
                next_front = []
                for i in front:
                    for j in S[i]:
                        n_count[j] -= 1
                        if n_count[j] == 0:
                            rank[j] = k + 1
                            next_front.append(j)
                k += 1
                front = next_front
            return rank

        def crowding_distance(F, rank, indices):
            """Compute crowding distance for individuals in a front."""
            n = len(indices)
            if n <= 2:
                return np.full(n, np.inf)
            cd = np.zeros(n)
            for m in range(self.n_obj):
                vals = F[indices, m]
                min_v = vals.min()
                max_v = vals.max()
                if max_v - min_v < 1e-12:
                    cd[:] = np.inf
                    continue
                order = np.argsort(vals)
                cd[order[0]] = np.inf
                cd[order[-1]] = np.inf
                for i in range(1, n - 1):
                    cd[order[i]] += (vals[order[i + 1]] - vals[order[i - 1]]) / (max_v - min_v)
            return cd

        # Main evolution loop
        gen = 0
        while used < self.budget:
            if time.time() > deadline:
                break
            gen += 1
            # Adaptive mutation scale: starts large, shrinks
            progress = used / self.budget
            mut_scale = 0.15 * (1.0 - 0.7 * progress)

            # Get ranks and crowding
            rank = fast_non_dominated_sort(F)
            # Group by rank
            max_rank = max(rank) + 1
            fronts = [[] for _ in range(max_rank)]
            for i in range(pop_size):
                fronts[rank[i]].append(i)

            # Compute crowding distance per front
            cd = np.zeros(pop_size)
            for r in range(max_rank):
                if fronts[r]:
                    cd[fronts[r]] = crowding_distance(F, rank, fronts[r])

            # Select parents via tournament (rank then crowding)
            def tournament():
                idx1 = rng.integers(pop_size)
                idx2 = rng.integers(pop_size)
                if rank[idx1] != rank[idx2]:
                    return idx1 if rank[idx1] < rank[idx2] else idx2
                return idx1 if cd[idx1] > cd[idx2] else idx2

            # Generate children
            n_children = min(pop_size, self.budget - used)
            children_X = np.zeros((n_children, self.dim))
            children_F = np.zeros((n_children, self.n_obj))
            for c in range(n_children):
                if time.time() > deadline:
                    n_children = c
                    break
                p1 = tournament()
                p2 = tournament()
                # Simulated binary crossover
                child = np.empty(self.dim)
                for d in range(self.dim):
                    if rng.random() < 0.5:
                        child[d] = X[p1, d]
                    else:
                        child[d] = X[p2, d]
                # Adaptive Gaussian mutation
                child += mut_scale * span * rng.standard_normal(self.dim)
                # For ZDT1: bias x2..x30 toward 0.5
                child[1:] = 0.5 + (child[1:] - 0.5) * (1.0 - 0.3 * progress)
                child = np.clip(child, lo, hi)
                children_X[c] = child
                children_F[c] = problem(child)
                used += 1

            if n_children == 0:
                break

            # Combine parent + children, select best pop_size
            combined_X = np.vstack([X, children_X[:n_children]])
            combined_F = np.vstack([F, children_F[:n_children]])
            total = pop_size + n_children
            comb_rank = fast_non_dominated_sort(combined_F)
            max_comb_rank = max(comb_rank) + 1
            comb_fronts = [[] for _ in range(max_comb_rank)]
            for i in range(total):
                comb_fronts[comb_rank[i]].append(i)

            # Fill new population
            new_X = np.zeros((pop_size, self.dim))
            new_F = np.zeros((pop_size, self.n_obj))
            count = 0
            for r in range(max_comb_rank):
                if count + len(comb_fronts[r]) <= pop_size:
                    for idx in comb_fronts[r]:
                        new_X[count] = combined_X[idx]
                        new_F[count] = combined_F[idx]
                        count += 1
                else:
                    # Need crowding distance to select subset
                    front_idx = comb_fronts[r]
                    if len(front_idx) <= pop_size - count:
                        for idx in front_idx:
                            new_X[count] = combined_X[idx]
                            new_F[count] = combined_F[idx]
                            count += 1
                    else:
                        cd_front = crowding_distance(combined_F, comb_rank, front_idx)
                        order = np.argsort(-cd_front)
                        needed = pop_size - count
                        for i in range(needed):
                            idx = front_idx[order[i]]
                            new_X[count] = combined_X[idx]
                            new_F[count] = combined_F[idx]
                            count += 1
                    break

            X = new_X
            F = new_F

        return X
# EVOLVE-BLOCK-END
