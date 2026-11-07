"""Multi-objective optimizer for DTLZ1."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """NSGA-II style multi-objective optimizer for DTLZ1."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 290  # safety margin before 330s kill

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        dim = self.dim
        n_obj = self.n_obj
        budget = self.budget

        # Population size: keep small for budget=200
        pop_size = min(20, max(10, budget // 10))

        # Use the global numpy random state (seeded by evaluator)
        def rand_vec():
            return lo + (hi - lo) * np.random.random(dim)

        # Initialize population
        X = np.array([rand_vec() for _ in range(pop_size)])
        F = np.array([problem(x) for x in X])
        used = pop_size

        def fast_non_dominated_sort(F):
            """Return list of fronts (each a list of indices)."""
            n = len(F)
            S = [[] for _ in range(n)]  # domination sets
            n_d = np.zeros(n, dtype=int)  # domination counts
            fronts = [[]]

            for p in range(n):
                for q in range(n):
                    if p == q:
                        continue
                    if np.all(F[p] <= F[q]) and np.any(F[p] < F[q]):
                        S[p].append(q)
                    elif np.all(F[q] <= F[p]) and np.any(F[q] < F[p]):
                        n_d[p] += 1
                if n_d[p] == 0:
                    fronts[0].append(p)

            i = 0
            while fronts[i]:
                next_front = []
                for p in fronts[i]:
                    for q in S[p]:
                        n_d[q] -= 1
                        if n_d[q] == 0:
                            next_front.append(q)
                i += 1
                fronts.append(next_front)

            # Remove the last empty front
            if not fronts[-1]:
                fronts.pop()
            return fronts

        def crowding_distance(F, indices):
            """Assign crowding distance to indices in F."""
            if len(indices) <= 2:
                return {i: np.inf for i in indices}
            dist = {i: 0.0 for i in indices}
            for m in range(n_obj):
                sorted_idx = sorted(indices, key=lambda i: F[i][m])
                dist[sorted_idx[0]] = np.inf
                dist[sorted_idx[-1]] = np.inf
                f_min = F[sorted_idx[0]][m]
                f_max = F[sorted_idx[-1]][m]
                if f_max - f_min < 1e-12:
                    continue
                for k in range(1, len(sorted_idx) - 1):
                    dist[sorted_idx[k]] += (F[sorted_idx[k + 1]][m] - F[sorted_idx[k - 1]][m]) / (f_max - f_min)
            return dist

        # Main evolution loop
        while used < budget:
            if time.time() > deadline:
                break

            # Compute fronts
            fronts = fast_non_dominated_sort(F)

            # Select parents: pick from first front (non-dominated)
            # If first front is small, include more
            parent_pool = fronts[0]
            if len(parent_pool) < 2:
                # Include second front if available
                if len(fronts) > 1:
                    parent_pool = parent_pool + fronts[1]
            if len(parent_pool) < 2:
                parent_pool = list(range(len(F)))

            # Binary tournament selection for two parents
            def tournament():
                a = parent_pool[np.random.randint(len(parent_pool))]
                b = parent_pool[np.random.randint(len(parent_pool))]
                # Pick the one with lower rank (front index)
                rank_a = None
                rank_b = None
                for fi, front in enumerate(fronts):
                    if a in front:
                        rank_a = fi
                        break
                    if b in front:
                        rank_b = fi
                        break
                if rank_a is None:
                    rank_a = 0
                if rank_b is None:
                    rank_b = 0
                if rank_a < rank_b:
                    return a
                elif rank_b < rank_a:
                    return b
                else:
                    # Same front: pick randomly
                    return a if np.random.random() < 0.5 else b

            p1 = tournament()
            p2 = tournament()
            if p1 == p2:
                p2 = parent_pool[np.random.randint(len(parent_pool))]

            # SBX crossover
            child = X[p1].copy()
            if np.random.random() < 0.9:  # crossover probability
                eta = 15.0
                for j in range(dim):
                    u = np.random.random()
                    if u <= 0.5:
                        beta = (2 * u) ** (1.0 / (eta + 1))
                    else:
                        beta = (2 * (1 - u)) ** (1.0 / (eta + 1))
                    child[j] = 0.5 * ((1 + beta) * X[p1][j] + (1 - beta) * X[p2][j])

            # Mutation
            if np.random.random() < 0.9:  # mutation probability
                for j in range(dim):
                    if np.random.random() < 0.5:
                        u = np.random.random()
                        if u <= 0.5:
                            delta = (2 * u) ** (1.0 / (eta + 1)) - 1
                        else:
                            delta = 1 - (2 * (1 - u)) ** (1.0 / (eta + 1))
                        child[j] += delta * (hi[j] - lo[j])

            child = np.clip(child, lo, hi)
            fc = problem(child)
            used += 1

            # Replace worst individual
            # Compute a simple fitness: use front rank and crowding
            # For simplicity, replace the individual with the highest sum of objectives
            # among dominated individuals, or worst overall
            dominated_mask = np.array([
                np.any(np.all(F <= F[i], axis=1) & np.any(F < F[i], axis=1))
                for i in range(len(F))
            ])
            if np.any(dominated_mask):
                # Among dominated, pick the one with highest objective sum
                dominated_indices = np.where(dominated_mask)[0]
                sums = F[dominated_indices].sum(axis=1)
                worst_idx = dominated_indices[np.argmax(sums)]
            else:
                # All non-dominated: pick randomly
                worst_idx = np.random.randint(len(F))

            X[worst_idx] = child
            F[worst_idx] = fc

        return X
# EVOLVE-BLOCK-END
