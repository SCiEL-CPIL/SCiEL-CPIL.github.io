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
        t_start = time.time()
        deadline = t_start + 280.0  # safety margin before 330s kill
        rng = np.random.default_rng(42)
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        span = hi - lo

        # Population size: balance between exploration and exploitation
        pop_size = min(40, max(10, self.budget // 5))

        # Initialize population
        X = lo + span * rng.random((pop_size, self.dim))
        F = np.array([problem(x) for x in X])
        used = pop_size

        def fast_non_dominated_sort(F):
            """Fast non-dominated sorting. Returns list of fronts (each a list of indices)."""
            n = len(F)
            S = [[] for _ in range(n)]  # set of solutions dominated by i
            n_dom = np.zeros(n, dtype=int)  # number of solutions dominating i
            fronts = []
            front0 = []

            for i in range(n):
                for j in range(n):
                    if i == j:
                        continue
                    if np.all(F[i] <= F[j]) and np.any(F[i] < F[j]):
                        S[i].append(j)
                    elif np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                        n_dom[i] += 1
                if n_dom[i] == 0:
                    front0.append(i)

            fronts.append(front0)
            k = 0
            while fronts[k]:
                next_front = []
                for i in fronts[k]:
                    for j in S[i]:
                        n_dom[j] -= 1
                        if n_dom[j] == 0:
                            next_front.append(j)
                k += 1
                if next_front:
                    fronts.append(next_front)
            return fronts

        def crowding_distance(F, front_indices):
            """Calculate crowding distance for a front."""
            n = len(front_indices)
            if n <= 2:
                return np.full(n, np.inf)
            cd = np.zeros(n)
            for m in range(self.n_obj):
                sorted_idx = np.argsort(F[front_indices, m])
                cd[sorted_idx[0]] = np.inf
                cd[sorted_idx[-1]] = np.inf
                f_range = F[front_indices, m].max() - F[front_indices, m].min()
                if f_range > 0:
                    for i in range(1, n - 1):
                        cd[sorted_idx[i]] += (F[front_indices[sorted_idx[i + 1]], m] -
                                              F[front_indices[sorted_idx[i - 1]], m]) / f_range
            return cd

        # Main evolutionary loop
        while used < self.budget:
            # Check deadline
            if time.time() > deadline:
                break

            # Non-dominated sorting
            fronts = fast_non_dominated_sort(F)

            # Select parents from first front(s) using tournament selection
            n_children = min(pop_size, self.budget - used)
            children_X = []
            children_F = []

            for _ in range(n_children):
                if time.time() > deadline:
                    break

                # Tournament selection: pick 2 random, choose one from better front
                i, j = rng.integers(0, pop_size, size=2)
                # Find front ranks
                rank_i = next((k for k, f in enumerate(fronts) if i in f), len(fronts))
                rank_j = next((k for k, f in enumerate(fronts) if j in f), len(fronts))

                if rank_i < rank_j:
                    parent = X[i]
                elif rank_j < rank_i:
                    parent = X[j]
                else:
                    # Same front: use crowding distance
                    front_idx = next(k for k, f in enumerate(fronts) if i in f)
                    front_indices = fronts[front_idx]
                    cd = crowding_distance(F, front_indices)
                    pos_i = front_indices.index(i)
                    pos_j = front_indices.index(j)
                    parent = X[i] if cd[pos_i] > cd[pos_j] else X[j]

                # Crossover and mutation
                if rng.random() < 0.8:  # crossover probability
                    p = rng.random(self.dim)
                    child = np.where(p < 0.5, parent, parent)  # simple: just mutate
                    # Simulated binary crossover approximation
                    eta = 20
                    u = rng.random(self.dim)
                    beta = np.where(u < 0.5, (2 * u) ** (1 / (eta + 1)),
                                     (2 * (1 - u)) ** (1 / (eta + 1)))
                    child = parent + 0.5 * (beta - 1) * span * rng.random(self.dim)
                else:
                    child = parent.copy()

                # Polynomial mutation
                if rng.random() < 0.9:
                    mutate_mask = rng.random(self.dim) < 0.1
                    if np.any(mutate_mask):
                        eta_m = 20
                        u = rng.random(np.sum(mutate_mask))
                        delta = np.where(u < 0.5, (2 * u) ** (1 / (eta_m + 1)) - 1,
                                         1 - (2 * (1 - u)) ** (1 / (eta_m + 1)))
                        child[mutate_mask] += delta * span[mutate_mask] * 0.5

                child = np.clip(child, lo, hi)
                children_X.append(child)
                children_F.append(problem(child))
                used += 1

            if not children_X:
                break

            # Combine population and offspring
            all_X = np.vstack([X, np.array(children_X)])
            all_F = np.vstack([F, np.array(children_F)])

            # Select next generation
            all_fronts = fast_non_dominated_sort(all_F)
            new_X = []
            new_F = []
            for front in all_fronts:
                if len(new_X) + len(front) <= pop_size:
                    for idx in front:
                        new_X.append(all_X[idx])
                        new_F.append(all_F[idx])
                else:
                    # Need to truncate: use crowding distance
                    remaining = pop_size - len(new_X)
                    cd = crowding_distance(all_F, front)
                    sorted_front = sorted(front, key=lambda i: cd[front.index(i)], reverse=True)
                    for idx in sorted_front[:remaining]:
                        new_X.append(all_X[idx])
                        new_F.append(all_F[idx])
                    break

            X = np.array(new_X)
            F = np.array(new_F)

        return X
# EVOLVE-BLOCK-END
