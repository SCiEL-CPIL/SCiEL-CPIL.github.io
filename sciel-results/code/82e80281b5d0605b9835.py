"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """NSGA-II based multi-objective optimizer with time guard."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 310.0  # Safety margin before 330s kill

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        span = hi - lo

        budget = self.budget
        dim = self.dim
        n_obj = self.n_obj

        # Population size: balance exploration with budget
        pop_size = min(40, max(10, budget // 5))
        # Ensure we can do at least 1 generation
        if budget < 2 * pop_size:
            pop_size = max(4, budget // 2)

        # Initial population
        rng = np.random.default_rng(0)
        X = lo + span * rng.random((pop_size, dim))
        F = np.array([problem(x) for x in X])
        used = pop_size

        def fast_non_dominated_sort(F):
            """Return list of fronts, each a list of indices."""
            n = len(F)
            S = [[] for _ in range(n)]
            nd = np.zeros(n, dtype=int)
            fronts = []
            for i in range(n):
                for j in range(n):
                    if i == j:
                        continue
                    if np.all(F[i] <= F[j]) and np.any(F[i] < F[j]):
                        S[i].append(j)
                        nd[i] += 1
                    elif np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                        nd[j] += 1
            current_front = [i for i in range(n) if nd[i] == 0]
            while current_front:
                fronts.append(current_front)
                next_front = []
                for i in current_front:
                    for j in S[i]:
                        nd[j] -= 1
                        if nd[j] == 0:
                            next_front.append(j)
                current_front = next_front
            return fronts

        def crowding_distance(F, indices):
            """Return crowding distances for given indices."""
            n = len(indices)
            if n <= 2:
                return np.full(n, np.inf)
            cd = np.zeros(n)
            for m in range(n_obj):
                vals = F[indices, m]
                mn, mx = vals.min(), vals.max()
                if mx - mn < 1e-15:
                    continue
                order = np.argsort(vals)
                cd[order[0]] = np.inf
                cd[order[-1]] = np.inf
                for k in range(1, n - 1):
                    cd[order[k]] += (vals[order[k + 1]] - vals[order[k - 1]]) / (mx - mn)
            return cd

        while used < budget:
            if time.time() > deadline - 5.0:
                break

            # Create offspring
            # Tournament selection
            def tournament_select(F, fronts, n_sel=2):
                n = len(F)
                # Assign rank to each individual
                rank = np.zeros(n, dtype=int)
                for r, front in enumerate(fronts):
                    for idx in front:
                        rank[idx] = r
                selected = []
                for _ in range(n_sel):
                    candidates = rng.integers(0, n, size=2)
                    if rank[candidates[0]] < rank[candidates[1]]:
                        selected.append(candidates[0])
                    elif rank[candidates[1]] < rank[candidates[0]]:
                        selected.append(candidates[1])
                    else:
                        # Same rank, pick randomly
                        selected.append(candidates[rng.integers(2)])
                return selected

            fronts = fast_non_dominated_sort(F)

            # Generate offspring
            offspring_X = []
            offspring_F = []
            n_offspring = pop_size
            for _ in range(n_offspring):
                p1, p2 = tournament_select(F, fronts, 2)
                # SBX crossover
                if rng.random() < 0.9:
                    # Simulated binary crossover
                    beta = 15.0
                    child1 = np.zeros(dim)
                    child2 = np.zeros(dim)
                    for d in range(dim):
                        if rng.random() < 0.5:
                            if abs(X[p1, d] - X[p2, d]) > 1e-15:
                                if rng.random() < 0.5:
                                    u = rng.random()
                                    if u <= 1.0 / (beta + 1):
                                        beta1 = (2 * u) ** (1.0 / (beta + 1))
                                    else:
                                        beta1 = (1.0 / (2 * (1 - u))) ** (1.0 / (beta + 1))
                                    child1[d] = 0.5 * ((1 + beta1) * X[p1, d] + (1 - beta1) * X[p2, d])
                                    child2[d] = 0.5 * ((1 - beta1) * X[p1, d] + (1 + beta1) * X[p2, d])
                                else:
                                    u = rng.random()
                                    if u <= 1.0 / (beta + 1):
                                        beta1 = (2 * u) ** (1.0 / (beta + 1))
                                    else:
                                        beta1 = (1.0 / (2 * (1 - u))) ** (1.0 / (beta + 1))
                                    child1[d] = 0.5 * ((1 + beta1) * X[p2, d] + (1 - beta1) * X[p1, d])
                                    child2[d] = 0.5 * ((1 - beta1) * X[p2, d] + (1 + beta1) * X[p1, d])
                                if rng.random() < 0.5:
                                    child1[d], child2[d] = child2[d], child1[d]
                            else:
                                child1[d] = X[p1, d]
                                child2[d] = X[p2, d]
                        else:
                            child1[d] = X[p1, d]
                            child2[d] = X[p2, d]
                    # Mutate
                    for d in range(dim):
                        if rng.random() < 1.0 / dim:
                            if rng.random() < 0.5:
                                u = rng.random()
                                if u <= 1.0 / (beta + 1):
                                    beta1 = (2 * u) ** (1.0 / (beta + 1))
                                else:
                                    beta1 = (1.0 / (2 * (1 - u))) ** (1.0 / (beta + 1))
                                child1[d] = X[p1, d] + beta1 * (hi[d] - lo[d]) * (rng.random() - 0.5)
                            else:
                                child1[d] = X[p1, d] + 0.1 * span[d] * rng.standard_normal()
                        if rng.random() < 1.0 / dim:
                            if rng.random() < 0.5:
                                u = rng.random()
                                if u <= 1.0 / (beta + 1):
                                    beta1 = (2 * u) ** (1.0 / (beta + 1))
                                else:
                                    beta1 = (1.0 / (2 * (1 - u))) ** (1.0 / (beta + 1))
                                child2[d] = X[p2, d] + beta1 * (hi[d] - lo[d]) * (rng.random() - 0.5)
                            else:
                                child2[d] = X[p2, d] + 0.1 * span[d] * rng.standard_normal()
                    child1 = np.clip(child1, lo, hi)
                    child2 = np.clip(child2, lo, hi)
                    offspring_X.append(child1)
                    offspring_X.append(child2)
                    offspring_F.append(problem(child1))
                    offspring_F.append(problem(child2))
                    used += 2
                    if used >= budget:
                        break
                else:
                    # No crossover, just mutate
                    child1 = X[p1].copy()
                    for d in range(dim):
                        if rng.random() < 1.0 / dim:
                            child1[d] = X[p1, d] + 0.1 * span[d] * rng.standard_normal()
                    child1 = np.clip(child1, lo, hi)
                    offspring_X.append(child1)
                    offspring_F.append(problem(child1))
                    used += 1
                    if used >= budget:
                        break

            if not offspring_X:
                break

            offspring_X = np.array(offspring_X)
            offspring_F = np.array(offspring_F)

            # Combine and select
            all_X = np.vstack([X, offspring_X])
            all_F = np.vstack([F, offspring_F])
            all_fronts = fast_non_dominated_sort(all_F)

            # Select next population
            new_X = []
            new_F = []
            for front in all_fronts:
                if len(new_X) + len(front) <= pop_size:
                    for idx in front:
                        new_X.append(all_X[idx])
                        new_F.append(all_F[idx])
                else:
                    remaining = pop_size - len(new_X)
                    cd = crowding_distance(all_F, front)
                    order = np.argsort(-cd)
                    for k in range(min(remaining, len(front))):
                        idx = front[order[k]]
                        new_X.append(all_X[idx])
                        new_F.append(all_F[idx])
                    break

            X = np.array(new_X)
            F = np.array(new_F)

        return X
# EVOLVE-BLOCK-END
