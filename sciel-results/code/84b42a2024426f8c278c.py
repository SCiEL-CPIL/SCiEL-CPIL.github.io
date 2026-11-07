# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """NSGA-II based multi-objective optimizer."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280.0

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        dim = self.dim
        budget = self.budget

        rng = np.random.default_rng(42)

        # Population size
        pop_size = min(40, max(10, budget // 5))

        # Initialize population
        X = lo + (hi - lo) * rng.random((pop_size, dim))
        F = np.array([problem(x) for x in X])
        used = pop_size

        # Keep track of all evaluated points for hypervolume
        all_F = [F.copy()]

        def fast_non_dominated_sort(F):
            """Return list of fronts, each a list of indices."""
            n = len(F)
            S = [[] for _ in range(n)]  # Dominated sets
            n_dom = np.zeros(n, dtype=int)  # Number of dominators
            fronts = []
            for i in range(n):
                for j in range(n):
                    if i == j:
                        continue
                    if np.all(F[i] <= F[j]) and np.any(F[i] < F[j]):
                        S[i].append(j)
                    elif np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                        n_dom[i] += 1
            current_front = [i for i in range(n) if n_dom[i] == 0]
            while current_front:
                fronts.append(current_front)
                next_front = []
                for i in current_front:
                    for j in S[i]:
                        n_dom[j] -= 1
                        if n_dom[j] == 0:
                            next_front.append(j)
                current_front = next_front
            return fronts

        def crowding_distance(F, indices):
            """Compute crowding distance for given indices."""
            if len(indices) <= 2:
                return np.full(len(indices), np.inf)
            dist = np.zeros(len(indices))
            for m in range(F.shape[1]):
                vals = F[indices, m]
                idx = np.argsort(vals)
                dist[idx[0]] = np.inf
                dist[idx[-1]] = np.inf
                for k in range(1, len(idx) - 1):
                    dist[idx[k]] += (vals[idx[k+1]] - vals[idx[k-1]]) / (vals[-1] - vals[0] + 1e-12)
            return dist

        while used < budget and time.time() < deadline - 20.0:
            # Non-dominated sort
            fronts = fast_non_dominated_sort(F)
            
            # Create offspring
            offspring_X = []
            offspring_F = []
            n_offspring = pop_size
            
            while len(offspring_X) < n_offspring and used < budget:
                # Select two parents using tournament selection
                def tournament():
                    i, j = rng.integers(pop_size), rng.integers(pop_size)
                    # Get front rank
                    rank_i = next(k for k, front in enumerate(fronts) if i in front)
                    rank_j = next(k for k, front in enumerate(fronts) if j in front)
                    if rank_i < rank_j:
                        return i
                    elif rank_j < rank_i:
                        return j
                    else:
                        # Crowding distance
                        front_idx = [i for i in fronts[rank_i]]
                        cd = crowding_distance(F, front_idx)
                        cd_i = cd[front_idx.index(i)]
                        cd_j = cd[front_idx.index(j)]
                        return i if cd_i > cd_j else j
                
                p1 = tournament()
                p2 = tournament()
                
                # SBX crossover
                x1, x2 = X[p1], X[p2]
                if rng.random() < 0.9:
                    # Simulated binary crossover
                    child1 = np.zeros(dim)
                    child2 = np.zeros(dim)
                    for d in range(dim):
                        if rng.random() < 0.5:
                            beta = 1.0
                            if x1[d] != x2[d]:
                                if rng.random() <= 2.0 / (beta + 2.0):
                                    beta_q = ((2.0 * min(x1[d], x2[d]) - lo[d] +
                                              (hi[d] - max(x1[d], x2[d])) * (rng.random() ** (1.0 / (beta + 1.0)))) /
                                              (x1[d] - x2[d] + 1e-12))
                                    child1[d] = 0.5 * (x1[d] + x2[d] - (x1[d] - x2[d]) * beta_q)
                                else:
                                    beta_q = ((2.0 * (hi[d] - max(x1[d], x2[d])) +
                                              min(x1[d], x2[d]) - lo[d]) * (rng.random() ** (1.0 / (beta + 1.0)))) / \
                                              (x2[d] - x1[d] + 1e-12)
                                    child1[d] = 0.5 * (x1[d] + x2[d] + (x1[d] - x2[d]) * beta_q)
                            child2[d] = 2.0 * 0.5 - child1[d] + x1[d] + x2[d] - child1[d]
                            child2[d] = x1[d] + x2[d] - child1[d]
                        else:
                            child1[d] = x1[d]
                            child2[d] = x2[d]
                    child1 = np.clip(child1, lo, hi)
                    child2 = np.clip(child2, lo, hi)
                else:
                    child1 = x1.copy()
                    child2 = x2.copy()
                
                # Mutation
                for d in range(dim):
                    if rng.random() < 1.0 / dim:
                        child1[d] = lo[d] + (hi[d] - lo[d]) * rng.random()
                    if rng.random() < 1.0 / dim:
                        child2[d] = lo[d] + (hi[d] - lo[d]) * rng.random()
                
                offspring_X.append(child1)
                offspring_X.append(child2)
                used += 2
                if used > budget:
                    break
            
            offspring_X = np.array(offspring_X[:n_offspring])
            offspring_F = np.array([problem(x) for x in offspring_X])
            used += len(offspring_X)
            
            # Combine parent and offspring
            combined_X = np.vstack([X, offspring_X])
            combined_F = np.vstack([F, offspring_F])
            
            # Select next generation
            combined_fronts = fast_non_dominated_sort(combined_F)
            next_X = []
            next_F = []
            for front in combined_fronts:
                if len(next_X) + len(front) <= pop_size:
                    for idx in front:
                        next_X.append(combined_X[idx])
                        next_F.append(combined_F[idx])
                else:
                    # Need to select from this front based on crowding distance
                    remaining = pop_size - len(next_X)
                    cd = crowding_distance(combined_F, front)
                    order = np.argsort(cd)[::-1]
                    for k in range(remaining):
                        idx = front[order[k]]
                        next_X.append(combined_X[idx])
                        next_F.append(combined_F[idx])
                    break
            
            X = np.array(next_X)
            F = np.array(next_F)

        return X
# EVOLVE-BLOCK-END
