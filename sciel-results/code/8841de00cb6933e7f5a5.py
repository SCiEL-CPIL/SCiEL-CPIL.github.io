"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-objective evolutionary optimizer using NSGA-II."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 290.0  # Safety margin before 300s limit

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        rng = np.random.default_rng(0)

        # Population size: Balance exploration and exploitation for small budget
        # 200 budget / 20 pop = 10 generations
        pop_size = 20
        n_eval = 0

        # Initialize population
        X = lo + (hi - lo) * rng.random((pop_size, self.dim))
        F = np.array([problem(x) for x in X])
        n_eval += pop_size

        # Helper: Fast non-dominated sorting
        def fast_non_dominated_sort(F):
            n = len(F)
            S = [[] for _ in range(n)]
            nd = np.zeros(n, dtype=int)
            fronts = [[]]

            for i in range(n):
                for j in range(n):
                    if i == j:
                        continue
                    # Does i dominate j?
                    if np.all(F[i] <= F[j]) and np.any(F[i] < F[j]):
                        S[i].append(j)
                    # Does j dominate i?
                    if np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                        nd[i] += 1
                if nd[i] == 0:
                    fronts[0].append(i)

            rank = np.full(n, -1, dtype=int)
            for i in fronts[0]:
                rank[i] = 0

            k = 0
            while fronts[k]:
                next_front = []
                for i in fronts[k]:
                    for j in S[i]:
                        nd[j] -= 1
                        if nd[j] == 0:
                            rank[j] = k + 1
                            next_front.append(j)
                k += 1
                fronts.append(next_front)

            return rank, fronts[:-1]

        # Helper: Crowding distance
        def crowding_distance(F, indices):
            if len(indices) <= 2:
                return np.full(len(indices), np.inf)
            m = len(indices)
            cd = np.zeros(m)
            n_obj = F.shape[1]
            for k in range(n_obj):
                obj_vals = F[indices, k]
                sorted_idx = np.argsort(obj_vals)
                cd[sorted_idx[0]] = np.inf
                cd[sorted_idx[-1]] = np.inf
                f_range = obj_vals[sorted_idx[-1]] - obj_vals[sorted_idx[0]]
                if f_range > 1e-15:
                    for i in range(1, m - 1):
                        cd[sorted_idx[i]] += (
                            obj_vals[sorted_idx[i + 1]] - obj_vals[sorted_idx[i - 1]]
                        ) / f_range
            return cd

        # Helper: Binary tournament selection
        def tournament_select(rank, cd, n_select):
            selected_indices = []
            for _ in range(n_select):
                i, j = rng.integers(len(rank), size=2)
                if rank[i] < rank[j]:
                    winner = i
                elif rank[j] < rank[i]:
                    winner = j
                else:
                    if cd[i] > cd[j]:
                        winner = i
                    else:
                        winner = j
                selected_indices.append(winner)
            return np.array(selected_indices)

        # Helper: SBX Crossover
        def crossover(x1, x2):
            if rng.random() < 0.9:
                eta = 15.0
                child1 = np.zeros(self.dim)
                child2 = np.zeros(self.dim)
                for i in range(self.dim):
                    if x1[i] == x2[i]:
                        child1[i] = x1[i]
                        child2[i] = x2[i]
                    else:
                        u = rng.random()
                        if u <= 0.5:
                            beta = (2 * u) ** (1.0 / (eta + 1))
                        else:
                            beta = (2 * (1 - u)) ** (1.0 / (eta + 1))
                        child1[i] = 0.5 * ((1 + beta) * x1[i] + (1 - beta) * x2[i])
                        child2[i] = 0.5 * ((1 - beta) * x1[i] + (1 + beta) * x2[i])
                return child1, child2
            else:
                return x1.copy(), x2.copy()

        # Helper: Polynomial Mutation
        def mutate(x):
            eta = 20.0
            x_new = x.copy()
            for i in range(self.dim):
                if rng.random() < 1.0 / self.dim:
                    u = rng.random()
                    if u < 0.5:
                        delta = (2 * u) ** (1.0 / (eta + 1)) - 1
                    else:
                        delta = 1 - (2 * (1 - u)) ** (1.0 / (eta + 1))
                    x_new[i] += delta * (hi[i] - lo[i])
            return np.clip(x_new, lo, hi)

        # Main loop
        while n_eval < self.budget:
            if time.time() > deadline:
                break

            # Compute ranks and crowding distances
            rank, fronts = fast_non_dominated_sort(F)
            cd_all = np.zeros(len(F))
            for front in fronts:
                if len(front) > 2:
                    cd_front = crowding_distance(F, np.array(front))
                    for idx, val in zip(front, cd_front):
                        cd_all[idx] = val
                else:
                    for idx in front:
                        cd_all[idx] = np.inf

            # Generate offspring
            offspring_X = []
            offspring_F = []
            
            while len(offspring_X) < pop_size and n_eval < self.budget:
                # Select parents via tournament
                parent_indices = tournament_select(rank, cd_all, 2)
                p1, p2 = parent_indices
                x1, x2 = X[p1], X[p2]
                
                # Crossover
                c1, c2 = crossover(x1, x2)
                
                # Mutate
                c1 = mutate(c1)
                c2 = mutate(c2)
                
                # Evaluate first child
                f1 = problem(c1)
                n_eval += 1
                offspring_X.append(c1)
                offspring_F.append(f1)
                
                # Evaluate second child if budget allows
                if n_eval < self.budget:
                    f2 = problem(c2)
                    n_eval += 1
                    offspring_X.append(c2)
                    offspring_F.append(f2)
                    if len(offspring_X) >= pop_size:
                        break

            # Combine parent and offspring
            X_comb = np.vstack([X, np.array(offspring_X)])
            F_comb = np.vstack([F, np.array(offspring_F)])

            # Sort combined population
            rank_comb, fronts_comb = fast_non_dominated_sort(F_comb)
            
            # Select next population
            next_X = []
            next_F = []
            cd_comb = np.zeros(len(F_comb))
            
            for front in fronts_comb:
                if len(next_X) + len(front) <= pop_size:
                    # Add all
                    for idx in front:
                        next_X.append(X_comb[idx])
                        next_F.append(F_comb[idx])
                else:
                    # Need to select from this front based on crowding distance
                    cd_front = crowding_distance(F_comb, np.array(front))
                    # Sort by crowding distance descending
                    sorted_front = np.argsort(-cd_front)
                    needed = pop_size - len(next_X)
                    for i in range(needed):
                        idx = front[sorted_front[i]]
                        next_X.append(X_comb[idx])
                        next_F.append(F_comb[idx])
                    break

            X = np.array(next_X)
            F = np.array(next_F)

        return X
# EVOLVE-BLOCK-END
