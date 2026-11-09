"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-objective optimizer using NSGA-II style search on ZDT3."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 290  # 10s safety margin before 300s limit

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        dim = self.dim
        budget = self.budget

        # Use a fixed seed for internal RNG for reproducibility within a seed
        # The evaluator sets np.random.seed(seed) before calling us, but we
        # use our own RNG for internal operations
        rng = np.random.default_rng(42)

        # Population size: use a good fraction of budget
        pop_size = min(40, max(10, budget // 5))

        # Initial population
        X = lo + (hi - lo) * rng.random((pop_size, dim))
        F = np.array([problem(x) for x in X])
        used = pop_size

        # Track all evaluated points for hypervolume
        all_X = [X.copy()]
        all_F = [F.copy()]

        def fast_non_dominated_sort(F):
            """NSGA-II style non-dominated sorting."""
            n = len(F)
            S = [[] for _ in range(n)]  # S[i] = set of points dominated by i
            n_dom = np.zeros(n, dtype=int)  # number of points dominating i
            fronts = [[]]

            for i in range(n):
                for j in range(n):
                    if i == j:
                        continue
                    # Check if i dominates j
                    if np.all(F[i] <= F[j]) and np.any(F[i] < F[j]):
                        S[i].append(j)
                    # Check if j dominates i
                    if np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                        n_dom[i] += 1
                if n_dom[i] == 0:
                    fronts[0].append(i)

            k = 0
            while len(fronts[k]) > 0:
                next_front = []
                for i in fronts[k]:
                    for j in S[i]:
                        n_dom[j] -= 1
                        if n_dom[j] == 0:
                            next_front.append(j)
                k += 1
                fronts.append(next_front)

            return fronts[:-1]  # Remove last empty front

        def crowding_distance(F, front_indices):
            """Calculate crowding distance for a front."""
            if len(front_indices) <= 2:
                return np.full(len(front_indices), np.inf)

            m = self.n_obj
            cd = np.zeros(len(front_indices))
            F_front = F[front_indices]

            for m_idx in range(m):
                sorted_idx = np.argsort(F_front[:, m_idx])
                cd[sorted_idx[0]] = np.inf
                cd[sorted_idx[-1]] = np.inf
                f_range = F_front[sorted_idx[-1], m_idx] - F_front[sorted_idx[0], m_idx]
                if f_range > 0:
                    for i in range(1, len(sorted_idx) - 1):
                        cd[sorted_idx[i]] += (F_front[sorted_idx[i + 1], m_idx] - F_front[sorted_idx[i - 1], m_idx]) / f_range

            return cd

        # Main loop
        while used < budget:
            # Check time limit
            if time.time() > deadline:
                break

            # Non-dominated sorting
            fronts = fast_non_dominated_sort(F)

            # Select parents using tournament selection based on rank and crowding distance
            parents = []
            for front in fronts:
                cd = crowding_distance(F, front)
                for idx in front:
                    parents.append((idx, cd[front.index(idx)]))

            # Tournament selection
            def tournament_select(pop_X, pop_F, pop_fronts, pop_cd):
                t1 = rng.integers(len(pop_X))
                t2 = rng.integers(len(pop_X))
                # Compare by front index first, then crowding distance
                rank1 = pop_fronts.index(next(f for f in pop_fronts if t1 in f)) if any(t1 in f for f in pop_fronts) else len(pop_fronts)
                rank2 = pop_fronts.index(next(f for f in pop_fronts if t2 in f)) if any(t2 in f for f in pop_fronts) else len(pop_fronts)
                if rank1 < rank2:
                    return t1
                elif rank2 < rank1:
                    return t2
                else:
                    if pop_cd[t1] > pop_cd[t2]:
                        return t1
                    else:
                        return t2

            # Calculate crowding distances for all fronts
            all_cd = np.zeros(len(F))
            for front in fronts:
                cd = crowding_distance(F, front)
                for i, idx in enumerate(front):
                    all_cd[idx] = cd[i]

            # Generate children
            child_X = []
            child_F = []
            n_children = min(pop_size, budget - used)

            for _ in range(n_children):
                if time.time() > deadline:
                    break

                p1 = tournament_select(X, F, fronts, all_cd)
                p2 = tournament_select(X, F, fronts, all_cd)

                # SBX crossover
                if rng.random() < 0.9:
                    eta_c = 20.0
                    child1 = np.zeros(dim)
                    child2 = np.zeros(dim)
                    for j in range(dim):
                        if rng.random() < 0.5:
                            u = rng.random()
                            if u <= 0.5:
                                beta = (2 * u) ** (1 / (eta_c + 1))
                            else:
                                beta = (1 / (2 * (1 - u))) ** (1 / (eta_c + 1))
                            child1[j] = 0.5 * ((1 + beta) * X[p1, j] + (1 - beta) * X[p2, j])
                            child2[j] = 0.5 * ((1 - beta) * X[p1, j] + (1 + beta) * X[p2, j])
                        else:
                            child1[j] = X[p1, j]
                            child2[j] = X[p2, j]
                    child1 = np.clip(child1, lo, hi)
                    child2 = np.clip(child2, lo, hi)
                    child_X.append(child1)
                    if len(child_X) < n_children:
                        child_X.append(child2)
                else:
                    # No crossover, just copy
                    child_X.append(X[p1].copy())
                    if len(child_X) < n_children:
                        child_X.append(X[p2].copy())

            # Mutate
            for i in range(len(child_X)):
                if rng.random() < 0.9:
                    eta_m = 20.0
                    for j in range(dim):
                        if rng.random() < 1.0 / dim:
                            u = rng.random()
                            if u <= 0.5:
                                delta = (2 * u) ** (1 / (eta_m + 1)) - 1
                            else:
                                delta = 1 - (2 * (1 - u)) ** (1 / (eta_m + 1))
                            child_X[i][j] = child_X[i][j] * (1 + delta)
                            child_X[i][j] = np.clip(child_X[i][j], lo[j], hi[j])

            # Evaluate children
            for cx in child_X:
                if used >= budget:
                    break
                if time.time() > deadline:
                    break
                cf = problem(cx)
                used += 1
                child_F.append(cf)
                all_X.append(cx.reshape(1, -1))
                all_F.append(np.array([cf]).reshape(1, -1))

            if not child_F:
                break

            # Combine parent and child populations
            X_combined = np.vstack([X, np.array(child_X)])
            F_combined = np.vstack([F, np.array(child_F)])

            # Non-dominated sorting on combined population
            fronts_combined = fast_non_dominated_sort(F_combined)

            # Select next generation
            new_X = []
            new_F = []
            for front in fronts_combined:
                if len(new_X) + len(front) <= pop_size:
                    for idx in front:
                        new_X.append(X_combined[idx])
                        new_F.append(F_combined[idx])
                else:
                    # Need to select from this front based on crowding distance
                    cd = crowding_distance(F_combined, front)
                    sorted_front = sorted(front, key=lambda i: cd[front.index(i)], reverse=True)
                    needed = pop_size - len(new_X)
                    for idx in sorted_front[:needed]:
                        new_X.append(X_combined[idx])
                        new_F.append(F_combined[idx])
                    break

            if new_X:
                X = np.array(new_X)
                F = np.array(new_F)

        # Return the best point found (ignored by evaluator, but must not raise)
        return X[0]
# EVOLVE-BLOCK-END
