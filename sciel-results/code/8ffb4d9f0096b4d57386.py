# EVOLVE-BLOCK-START
import numpy as np


class Optimizer:
    """NSGA-II style multi-objective optimizer for DTLZ2."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        rng = np.random.default_rng(42)
        lo = np.array([b[0] for b in bounds], dtype=float)
        hi = np.array([b[1] for b in bounds], dtype=float)
        span = hi - lo
        dim = self.dim
        n_obj = self.n_obj
        budget = self.budget

        # Population size
        N = 40
        if budget < 2 * N:
            N = max(2, budget // 2)

        # Initial population
        X = lo + span * rng.random((N, dim))
        F = np.array([problem(x) for x in X])
        used = N

        def fast_non_dominated_sort(F):
            """
            Returns a list of fronts, where each front is a list of indices.
            F is (N, M) array of objectives.
            """
            n = len(F)
            S = [[] for _ in range(n)]  # Set of solutions dominated by i
            n_dom = np.zeros(n, dtype=int)  # Number of solutions dominated by i
            rank = np.zeros(n, dtype=int)
            
            for i in range(n):
                for j in range(i + 1, n):
                    if np.all(F[i] <= F[j]) and np.any(F[i] < F[j]):
                        S[i].append(j)
                        n_dom[j] += 1
                    elif np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                        S[j].append(i)
                        n_dom[i] += 1
            
            fronts = []
            current_front = [i for i in range(n) if n_dom[i] == 0]
            for i in current_front:
                rank[i] = 0
            
            k = 0
            while current_front:
                fronts.append(current_front)
                next_front = []
                for i in current_front:
                    for j in S[i]:
                        n_dom[j] -= 1
                        if n_dom[j] == 0:
                            rank[j] = k + 1
                            next_front.append(j)
                current_front = next_front
                k += 1
            return fronts, rank

        def crowding_distance(F, indices):
            """
            Calculate crowding distance for solutions in indices.
            Returns an array of distances corresponding to indices.
            """
            if not indices:
                return []
            
            n = len(indices)
            if n <= 2:
                return [float('inf')] * n
            
            dist = np.zeros(n)
            for m in range(n_obj):
                f_vals = F[indices, m]
                f_min = np.min(f_vals)
                f_max = np.max(f_vals)
                if f_max - f_min < 1e-12:
                    continue
                # Sort indices by m-th objective
                sorted_idx = np.argsort(f_vals)
                dist[sorted_idx[0]] = float('inf')
                dist[sorted_idx[-1]] = float('inf')
                for i in range(1, n - 1):
                    dist[sorted_idx[i]] += (f_vals[sorted_idx[i + 1]] - f_vals[sorted_idx[i - 1]]) / (f_max - f_min)
            
            return dist

        def select_parents(pop_indices, ranks, crowd_dists, k=2):
            """
            Tournament selection.
            pop_indices: list of indices in the population.
            ranks: array of ranks for all N individuals.
            crowd_dists: array of crowding distances for all N individuals.
            Returns two parent indices.
            """
            def tournament():
                a = rng.integers(len(pop_indices))
                b = rng.integers(len(pop_indices))
                ia = pop_indices[a]
                ib = pop_indices[b]
                if ranks[ia] < ranks[ib]:
                    return ia
                elif ranks[ib] < ranks[ia]:
                    return ib
                else:
                    if crowd_dists[ia] > crowd_dists[ib]:
                        return ia
                    else:
                        return ib
            
            p1 = tournament()
            p2 = tournament()
            return p1, p2

        def sbx_crossover(x1, x2, eta=20.0):
            if rng.random() < 0.5:
                return x1.copy(), x2.copy()
            child1 = x1.copy()
            child2 = x2.copy()
            for d in range(dim):
                if rng.random() < 0.5:
                    if abs(x1[d] - x2[d]) < 1e-14:
                        continue
                    if rng.random() < 0.5:
                        beta = 1.0 / (2.0 * rng.random() + 1e-12)
                    else:
                        beta = 1.0 / (2.0 * (1.0 - rng.random()) + 1e-12)
                    beta += 1.0
                    if beta > 0:
                        alpha = 2.0 - beta ** (-eta)
                        if rng.random() < 0.5:
                            child1[d] = 0.5 * ((1 + alpha) * x1[d] + (1 - alpha) * x2[d])
                        else:
                            child1[d] = 0.5 * ((1 - alpha) * x1[d] + (1 + alpha) * x2[d])
                        if rng.random() < 0.5:
                            child2[d] = 0.5 * ((1 + alpha) * x2[d] + (1 - alpha) * x1[d])
                        else:
                            child2[d] = 0.5 * ((1 - alpha) * x2[d] + (1 + alpha) * x1[d])
            return np.clip(child1, lo, hi), np.clip(child2, lo, hi)

        def polynomial_mutate(x, eta=20.0, prob=1.0 / dim):
            child = x.copy()
            for d in range(dim):
                if rng.random() < prob:
                    delta1 = (child[d] - lo[d]) / span[d]
                    delta2 = (hi[d] - child[d]) / span[d]
                    rand = rng.random()
                    if rand < 0.5:
                        xy = 1.0 + (1.0 + 2.0 * delta1) * rand
                        deltaq = (xy ** (1.0 / (eta + 1.0)) - 1.0)
                    else:
                        xy = 1.0 + (1.0 + 2.0 * delta2) * (1.0 - rand)
                        deltaq = 1.0 - (xy ** (1.0 / (eta + 1.0)))
                    child[d] += deltaq * span[d]
                    child[d] = np.clip(child[d], lo[d], hi[d])
            return child

        # Main loop
        while used < budget:
            # Calculate ranks and crowding distances for current population
            fronts, ranks = fast_non_dominated_sort(F)
            
            # Calculate crowding distance for each front
            crowd_dists = np.zeros(N)
            for front in fronts:
                cd = crowding_distance(F, front)
                for idx, dist in zip(front, cd):
                    crowd_dists[idx] = dist
            
            # Generate offspring
            n_offspring = min(N, budget - used)
            offspring_X = np.zeros((n_offspring, dim))
            offspring_F = np.zeros((n_offspring, n_obj))
            
            for i in range(n_offspring):
                p1_idx, p2_idx = select_parents(list(range(N)), ranks, crowd_dists)
                c1, c2 = sbx_crossover(X[p1_idx], X[p2_idx])
                c1 = polynomial_mutate(c1)
                
                offspring_X[i] = c1
                offspring_F[i] = problem(c1)
                used += 1
                
                if used >= budget:
                    break
            
            # Merge populations
            X_new = np.vstack([X, offspring_X])
            F_new = np.vstack([F, offspring_F])
            
            # Select top N from merged population
            fronts_new, ranks_new = fast_non_dominated_sort(F_new)
            
            # We need to select N individuals
            selected_indices = []
            for front in fronts_new:
                if len(selected_indices) + len(front) <= N:
                    selected_indices.extend(front)
                else:
                    # Need to select only some from this front
                    remaining = N - len(selected_indices)
                    # Calculate crowding distance for this front
                    cd = crowding_distance(F_new, front)
                    # Sort by crowding distance descending
                    sorted_front = sorted(zip(front, cd), key=lambda x: x[1], reverse=True)
                    selected_indices.extend([idx for idx, _ in sorted_front[:remaining]])
                    break
            
            # Update population
            X = X_new[selected_indices]
            F = F_new[selected_indices]

        return X
# EVOLVE-BLOCK-END
