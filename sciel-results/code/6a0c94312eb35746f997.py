"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """NSGA-II with DE-style mutation and deadline guard for RE21."""

    def __init__(self, budget, dim, n_obj):
        self.budget, self.dim, self.n_obj = budget, dim, n_obj

    def __call__(self, problem, bounds):
        t0 = time.time()
        deadline = t0 + 270.0  # Safety margin before 300s limit

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        dim = self.dim
        n_obj = self.n_obj
        budget = self.budget

        rng = np.random.default_rng(0)

        # Population size: balance between exploration and computation
        pop_size = min(20, max(8, budget // 10))

        # Initial random population
        X = lo + (hi - lo) * rng.random((pop_size, dim))
        F = np.array([problem(x) for x in X])
        used = pop_size

        def non_dominated_sort(F):
            """Return list of fronts (indices) sorted by domination rank."""
            n = len(F)
            S = [[] for _ in range(n)]
            nd = [0] * n
            rank = [0] * n
            fronts = []

            for i in range(n):
                for j in range(n):
                    if i == j:
                        continue
                    if np.all(F[i] <= F[j]) and np.any(F[i] < F[j]):
                        S[i].append(j)
                    elif np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                        nd[i] += 1
                if nd[i] == 0:
                    rank[i] = 0
                    fronts.append([i])

            i = 0
            while i < len(fronts):
                next_front = []
                for k in fronts[i]:
                    for m in S[k]:
                        nd[m] -= 1
                        if nd[m] == 0:
                            rank[m] = i + 1
                            next_front.append(m)
                i += 1
                if next_front:
                    fronts.append(next_front)

            return fronts

        def crowding_distance(F, indices):
            """Compute crowding distance for given indices."""
            if len(indices) <= 2:
                return np.full(len(indices), np.inf)

            n = len(indices)
            cd = np.zeros(n)
            for m in range(n_obj):
                f_vals = F[indices, m]
                f_min, f_max = f_vals.min(), f_vals.max()
                if f_max - f_min < 1e-30:
                    cd[:] = np.inf
                    continue
                # Sort by objective m
                order = np.argsort(f_vals)
                cd[order[0]] = np.inf
                cd[order[-1]] = np.inf
                for k in range(1, n - 1):
                    cd[order[k]] += (f_vals[order[k + 1]] - f_vals[order[k - 1]]) / (f_max - f_min)
            return cd

        # DE-style parameters
        F_de = 0.8  # Differential weight
        CR = 0.9    # Crossover rate

        while used < budget:
            # Check deadline
            if time.time() > deadline:
                break

            # Non-dominated sort
            fronts = non_dominated_sort(F)

            # Select parents from first few fronts
            parent_pool = []
            for front in fronts:
                if len(parent_pool) + len(front) >= pop_size:
                    # Need crowding distance for this front
                    cd = crowding_distance(F, front)
                    # Take top by crowding distance
                    take = pop_size - len(parent_pool)
                    top_idx = np.argsort(-cd)[:take]
                    parent_pool.extend([front[i] for i in top_idx])
                    break
                else:
                    parent_pool.extend(front)

            # Generate children
            n_children = min(pop_size, budget - used)
            if n_children <= 0:
                break

            children_X = []
            children_F = []
            for _ in range(n_children):
                # DE mutation: pick 3 distinct parents
                if len(parent_pool) >= 3:
                    p1, p2, p3 = rng.choice(len(parent_pool), size=3, replace=False)
                    base = X[parent_pool[p1]]
                    mutant = base + F_de * (X[parent_pool[p2]] - X[parent_pool[p3]])
                else:
                    # Fallback: simple mutation
                    p1 = rng.integers(len(parent_pool))
                    base = X[parent_pool[p1]]
                    mutant = base + 0.1 * (hi - lo) * rng.standard_normal(dim)

                # Binomial crossover
                child = base.copy()
                jrand = rng.integers(dim)
                for j in range(dim):
                    if rng.random() < CR or j == jrand:
                        child[j] = mutant[j]

                # Clip to bounds
                child = np.clip(child, lo, hi)

                fc = problem(child)
                used += 1
                children_X.append(child)
                children_F.append(fc)

                # Early exit if deadline
                if time.time() > deadline:
                    break

            if not children_X:
                break

            children_X = np.array(children_X)
            children_F = np.array(children_F)

            # Combine parent and child populations
            X_comb = np.vstack([X, children_X])
            F_comb = np.vstack([F, children_F])

            # Select next generation
            fronts_comb = non_dominated_sort(F_comb)
            next_X = []
            next_F = []
            for front in fronts_comb:
                if len(next_X) + len(front) <= pop_size:
                    for idx in front:
                        next_X.append(X_comb[idx])
                        next_F.append(F_comb[idx])
                else:
                    # Need crowding distance to select
                    remaining = pop_size - len(next_X)
                    cd = crowding_distance(F_comb, front)
                    top_idx = np.argsort(-cd)[:remaining]
                    for idx in top_idx:
                        next_X.append(X_comb[front[idx]])
                        next_F.append(F_comb[front[idx]])
                    break

            X = np.array(next_X)
            F = np.array(next_F)

        return X
# EVOLVE-BLOCK-END
