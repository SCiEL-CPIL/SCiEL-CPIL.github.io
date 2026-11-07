"""Multi-objective optimizer on WFG7 with deadline guard."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """NSGA-II-style multi-objective optimizer with explicit deadline guard."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 270  # 30s safety margin before 300s limit

        lo = np.array([b[0] for b in bounds], dtype=np.float64)
        hi = np.array([b[1] for b in bounds], dtype=np.float64)
        dim = self.dim
        n_obj = self.n_obj
        budget = self.budget

        # Population size: use as much of the budget as reasonable for diversity
        # WFG7 is a deceptive problem; larger populations help explore
        pop_size = min(budget, 100)
        if budget < 20:
            pop_size = max(5, budget)

        # Use the global numpy random state (seeded by evaluator)
        # Generate initial population
        X = lo + (hi - lo) * np.random.random((pop_size, dim))
        F = np.array([problem(x) for x in X], dtype=np.float64)
        evals = pop_size

        # Reference point for dominance (only for internal selection)
        def fast_non_dominated_sort(F):
            """Return list of fronts, each a list of indices."""
            n = len(F)
            S = [[] for _ in range(n)]
            n_dom = [0] * n
            fronts = []
            for i in range(n):
                for j in range(n):
                    if i == j:
                        continue
                    if np.all(F[i] <= F[j]) and np.any(F[i] < F[j]):
                        S[i].append(j)
                        n_dom[i] += 1
                    elif np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                        n_dom[j] += 1
            # Front 0
            front0 = [i for i in range(n) if n_dom[i] == 0]
            fronts.append(front0)
            k = 0
            while fronts[k]:
                next_front = []
                for i in fronts[k]:
                    for j in S[i]:
                        n_dom[j] -= 1
                        if n_dom[j] == 0:
                            next_front.append(j)
                if next_front:
                    fronts.append(next_front)
                    k += 1
            return fronts

        def crowding_distance(F, indices):
            """Compute crowding distance for given indices."""
            n = len(indices)
            if n <= 2:
                return np.ones(n)
            cd = np.zeros(n)
            for m in range(n_obj):
                obj_vals = F[indices, m]
                sorted_idx = np.argsort(obj_vals)
                cd[sorted_idx[0]] = np.inf
                cd[sorted_idx[-1]] = np.inf
                if obj_vals[sorted_idx[-1]] - obj_vals[sorted_idx[0]] > 1e-15:
                    for i in range(1, n - 1):
                        cd[sorted_idx[i]] += (obj_vals[sorted_idx[i + 1]] - obj_vals[sorted_idx[i - 1]]) / (
                            obj_vals[sorted_idx[-1]] - obj_vals[sorted_idx[0]])
            return cd

        def tournament_select(F, fronts, n_sel):
            """Tournament selection based on non-dominated rank and crowding distance."""
            # Assign rank
            rank = np.zeros(len(F), dtype=int)
            for r, front in enumerate(fronts):
                for idx in front:
                    rank[idx] = r
            # Compute crowding distance per front
            cd = np.zeros(len(F))
            for front in fronts:
                if len(front) > 0:
                    cd[front] = crowding_distance(F, front)

            selected = []
            for _ in range(n_sel):
                candidates = np.random.choice(len(F), size=2, replace=False)
                i, j = candidates
                if rank[i] < rank[j]:
                    selected.append(i)
                elif rank[i] > rank[j]:
                    selected.append(j)
                else:
                    if cd[i] > cd[j]:
                        selected.append(i)
                    else:
                        selected.append(j)
            return selected

        def crossover(parent1, parent2):
            """Simulated binary crossover (SBX)."""
            eta = 20
            child1 = parent1.copy()
            child2 = parent2.copy()
            for i in range(dim):
                if np.random.random() < 0.5:
                    if abs(parent1[i] - parent2[i]) > 1e-14:
                        if np.random.random() < 0.5:
                            beta = (0.5 * (1 + (parent1[i] - parent2[i]) / (hi[i] - lo[i] + 1e-15)))
                        else:
                            beta = (0.5 * (1 + (parent2[i] - parent1[i]) / (hi[i] - lo[i] + 1e-15)))
                        beta_q = (2 * beta) ** (-1.0 / (eta + 1))
                        beta_p = (2 * (1 - beta)) ** (-1.0 / (eta + 1))
                        child1[i] = 0.5 * ((parent1[i] + parent2[i]) - beta_q * (parent1[i] - parent2[i]))
                        child2[i] = 0.5 * ((parent1[i] + parent2[i]) + beta_p * (parent1[i] - parent2[i]))
            return np.clip(child1, lo, hi), np.clip(child2, lo, hi)

        def mutate(x, sigma):
            """Gaussian mutation with adaptive step size."""
            y = x + sigma * (hi - lo) * np.random.standard_normal(dim)
            return np.clip(y, lo, hi)

        # Main evolutionary loop
        # Adaptive mutation sigma: start large, shrink over time
        sigma0 = 0.5
        total_gen = max(1, budget - pop_size)

        while evals < budget:
            if time.time() > deadline:
                break

            # Compute fronts
            fronts = fast_non_dominated_sort(F)

            # Determine how many offspring to generate this generation
            # Generate pop_size offspring per generation
            n_offspring = min(pop_size, budget - evals)
            if n_offspring <= 0:
                break

            # Create mating pool from current population
            parents_idx = tournament_select(F, fronts, n_offspring // 2 + 1)

            children_X = []
            children_F = []
            for k in range(0, len(parents_idx) - 1, 2):
                if evals >= budget or time.time() > deadline:
                    break
                p1 = X[parents_idx[k]]
                p2 = X[parents_idx[k + 1]]
                c1, c2 = crossover(p1, p2)
                # Mutation with decreasing sigma
                gen_frac = evals / max(1, budget)
                sigma = sigma0 * (1.0 - 0.7 * gen_frac)
                c1 = mutate(c1, sigma)
                c2 = mutate(c2, sigma)
                f1 = problem(c1)
                evals += 1
                children_X.append(c1)
                children_F.append(f1)
                if evals >= budget or time.time() > deadline:
                    break
                f2 = problem(c2)
                evals += 1
                children_X.append(c2)
                children_F.append(f2)

            if not children_X:
                break

            children_X = np.array(children_X)
            children_F = np.array(children_F)

            # Combine parent and offspring
            X_comb = np.vstack([X, children_X])
            F_comb = np.vstack([F, children_F])
            n_comb = len(F_comb)

            # Non-dominated sort on combined
            fronts_comb = fast_non_dominated_sort(F_comb)

            # Select next generation
            new_X = []
            new_F = []
            for front in fronts_comb:
                if len(new_X) + len(front) <= pop_size:
                    for idx in front:
                        new_X.append(X_comb[idx])
                        new_F.append(F_comb[idx])
                else:
                    remaining = pop_size - len(new_X)
                    if remaining <= 0:
                        break
                    # Need to truncate this front by crowding distance
                    cd = crowding_distance(F_comb, front)
                    # Sort by crowding distance descending
                    order = np.argsort(-cd)
                    for i in range(min(remaining, len(order))):
                        idx = front[order[i]]
                        new_X.append(X_comb[idx])
                        new_F.append(F_comb[idx])
                    break

            if len(new_X) == 0:
                break

            X = np.array(new_X)
            F = np.array(new_F)

        return X
# EVOLVE-BLOCK-END
