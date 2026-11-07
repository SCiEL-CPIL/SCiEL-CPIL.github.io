"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """NSGA-II style multi-objective optimizer with deadline guard."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 270.0

        lo = np.array([b[0] for b in bounds], dtype=np.float64)
        hi = np.array([b[1] for b in bounds], dtype=np.float64)
        dim = self.dim
        n_obj = self.n_obj
        budget = self.budget

        rng = np.random.RandomState(0)

        all_points = []  # list of (x, f) tuples for all evaluations

        def eval_point(x):
            x = np.clip(x, lo, hi)
            f = np.asarray(problem(x), dtype=np.float64)
            all_points.append((x.copy(), f.copy()))
            return x, f

        # Initial population
        pop_size = 40
        X = lo + (hi - lo) * rng.random((pop_size, dim))
        F = np.zeros((pop_size, n_obj))
        for i in range(pop_size):
            X[i], F[i] = eval_point(X[i])
        used = pop_size

        def fast_non_dominated_sort(F_mat):
            n = len(F_mat)
            domination_count = np.zeros(n, dtype=int)
            dominated_set = [[] for _ in range(n)]
            for i in range(n):
                for j in range(i + 1, n):
                    if np.all(F_mat[i] <= F_mat[j]) and np.any(F_mat[i] < F_mat[j]):
                        dominated_set[i].append(j)
                        domination_count[j] += 1
                    elif np.all(F_mat[j] <= F_mat[i]) and np.any(F_mat[j] < F_mat[i]):
                        dominated_set[j].append(i)
                        domination_count[i] += 1
            fronts = []
            current = [i for i in range(n) if domination_count[i] == 0]
            while current:
                fronts.append(current)
                next_front = []
                for i in current:
                    for j in dominated_set[i]:
                        domination_count[j] -= 1
                        if domination_count[j] == 0:
                            next_front.append(j)
                current = next_front
            return fronts

        def crowding_distance(F_mat, indices):
            m = len(indices)
            if m <= 2:
                return np.full(m, np.inf)
            dist = np.zeros(m)
            for k in range(n_obj):
                vals = F_mat[indices, k]
                vmin, vmax = vals.min(), vals.max()
                if vmax - vmin < 1e-12:
                    dist[:] = np.inf
                    continue
                order = np.argsort(vals)
                dist[order[0]] = np.inf
                dist[order[-1]] = np.inf
                for j in range(1, m - 1):
                    dist[order[j]] += (vals[order[j + 1]] - vals[order[j - 1]]) / (vmax - vmin)
            return dist

        while used < budget:
            if time.time() > deadline:
                break

            # Create offspring
            offspring = []
            while len(offspring) < pop_size and used + len(offspring) < budget:
                # Binary tournament selection
                i = rng.randint(pop_size)
                j = rng.randint(pop_size)
                # Compare by rank then crowding
                # Simple: pick non-dominated if possible
                fronts = fast_non_dominated_sort(F)
                front_ranks = {}
                for rank, front in enumerate(fronts):
                    for idx in front:
                        front_ranks[idx] = rank
                rank_i = front_ranks[i]
                rank_j = front_ranks[j]
                if rank_i < rank_j:
                    parent_idx = i
                elif rank_j < rank_i:
                    parent_idx = j
                else:
                    # Same front, use crowding
                    front_members = [idx for idx in range(pop_size) if front_ranks[idx] == rank_i]
                    cd = crowding_distance(F, front_members)
                    pos_i = front_members.index(i)
                    pos_j = front_members.index(j)
                    parent_idx = i if cd[pos_i] > cd[pos_j] else j

                # SBX crossover + polynomial mutation
                p1 = X[parent_idx]
                p2 = X[rng.randint(pop_size)]
                child = (p1 + p2) / 2.0
                # Add perturbation
                child += 0.05 * (hi - lo) * rng.standard_normal(dim)
                child = np.clip(child, lo, hi)
                _, fc = eval_point(child)
                offspring.append((child, fc))
                used += 1

            if not offspring:
                break

            # Combine parent and offspring
            X_combined = np.vstack([X, np.array([o[0] for o in offspring])])
            F_combined = np.vstack([F, np.array([o[1] for o in offspring])])

            # Sort into fronts
            fronts = fast_non_dominated_sort(F_combined)

            # Select new population
            new_X = []
            new_F = []
            for front in fronts:
                if len(new_X) + len(front) <= pop_size:
                    for idx in front:
                        new_X.append(X_combined[idx])
                        new_F.append(F_combined[idx])
                else:
                    remaining = pop_size - len(new_X)
                    cd = crowding_distance(F_combined, front)
                    # Sort by crowding distance descending
                    order = np.argsort(-cd)
                    for pos in order[:remaining]:
                        idx = front[pos]
                        new_X.append(X_combined[idx])
                        new_F.append(F_combined[idx])
                    break

            X = np.array(new_X)
            F = np.array(new_F)

        return all_points
# EVOLVE-BLOCK-END
