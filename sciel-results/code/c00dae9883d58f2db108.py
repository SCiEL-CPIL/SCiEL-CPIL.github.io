# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-objective optimizer for ZDT3 using NSGA-II style search."""

    def __init__(self, dim, n_obj, budget):
        self.dim = dim
        self.n_obj = n_obj
        self.budget = budget

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 290.0  # safety margin before 300s limit

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        dim = self.dim
        n_obj = self.n_obj
        budget = self.budget

        # Population size
        pop_size = min(100, max(20, budget // 2))
        if pop_size > budget:
            pop_size = max(10, budget // 2)

        # Initialize population
        rng = np.random.default_rng(42)
        X = lo + (hi - lo) * rng.random((pop_size, dim))
        F = np.array([problem(x) for x in X])
        used = pop_size

        def dominates(a, b):
            """Check if a dominates b (minimization)."""
            return np.all(a <= b) and np.any(a < b)

        def fast_non_dominated_sort(F):
            """NSGA-II fast non-dominated sorting. Returns list of fronts."""
            n = len(F)
            S = [[] for _ in range(n)]  # S[i] = indices dominated by i
            n_d = np.zeros(n, dtype=int)  # number of solutions that dominate i
            fronts = [[]]

            for i in range(n):
                for j in range(n):
                    if i == j:
                        continue
                    if dominates(F[i], F[j]):
                        S[i].append(j)
                    elif dominates(F[j], F[i]):
                        n_d[i] += 1
                if n_d[i] == 0:
                    fronts[0].append(i)

            i = 0
            while fronts[i]:
                next_front = []
                for k in fronts[i]:
                    for m in S[k]:
                        n_d[m] -= 1
                        if n_d[m] == 0:
                            next_front.append(m)
                i += 1
                if next_front:
                    fronts.append(next_front)

            return fronts

        def crowding_distance(F, front_indices):
            """Compute crowding distance for solutions in a front."""
            n = len(front_indices)
            if n <= 2:
                return np.ones(n)
            
            dist = np.zeros(n)
            for m in range(n_obj):
                vals = F[front_indices, m]
                min_val = np.min(vals)
                max_val = np.max(vals)
                if max_val - min_val < 1e-15:
                    continue
                # Sort by objective m
                order = np.argsort(vals)
                dist[order[0]] = np.inf
                dist[order[-1]] = np.inf
                for k in range(1, n - 1):
                    dist[order[k]] += (vals[order[k + 1]] - vals[order[k - 1]]) / (max_val - min_val)
            return dist

        def tournament_select(pop_F, fronts, k=2):
            """Tournament selection based on rank and crowding distance."""
            n = len(pop_F)
            # Assign ranks
            rank = np.zeros(n, dtype=int)
            for i, front in enumerate(fronts):
                for idx in front:
                    rank[idx] = i + 1

            # Compute crowding distances per front
            cd = np.zeros(n)
            for front in fronts:
                if len(front) > 0:
                    cd[front] = crowding_distance(pop_F, front)

            def compare(i, j):
                if rank[i] != rank[j]:
                    return rank[i] < rank[j]
                return cd[i] > cd[j]

            winners = []
            for _ in range(len(pop_F)):  # We'll select as many as needed
                candidates = rng.integers(0, n, size=k)
                best = candidates[0]
                for c in candidates[1:]:
                    if compare(c, best):
                        best = c
                winners.append(best)
            return winners

        # Main loop
        while used < budget:
            if time.time() > deadline:
                break

            # Evaluate current population
            fronts = fast_non_dominated_sort(F)

            # Select parents via tournament
            n_children = min(pop_size, budget - used)
            if n_children <= 0:
                break

            parent_indices = tournament_select(F, fronts, k=2)
            parents = X[parent_indices]

            # Generate children via mutation and crossover
            children = np.zeros((n_children, dim))
            child_F = np.zeros((n_children, n_obj))

            for i in range(n_children):
                if i % 2 == 0 and i + 1 < n_children and n_children > 1:
                    # Simulated binary crossover (SBX)
                    p1 = parents[i]
                    p2 = parents[i + 1]
                    eta = 20.0
                    c1 = p1.copy()
                    c2 = p2.copy()
                    for d in range(dim):
                        if rng.random() < 0.5:
                            u = rng.random()
                            if u <= 0.5:
                                beta = (2 * u) ** (1 / (eta + 1))
                            else:
                                beta = (1 / (2 * (1 - u))) ** (1 / (eta + 1))
                            c1[d] = 0.5 * ((1 + beta) * p1[d] + (1 - beta) * p2[d])
                            c2[d] = 0.5 * ((1 - beta) * p1[d] + (1 + beta) * p2[d])
                    # Mutation
                    for d in range(dim):
                        if rng.random() < 1.0 / dim:
                            c1[d] += 0.1 * (hi[d] - lo[d]) * rng.standard_normal()
                        if rng.random() < 1.0 / dim:
                            c2[d] += 0.1 * (hi[d] - lo[d]) * rng.standard_normal()
                    c1 = np.clip(c1, lo, hi)
                    c2 = np.clip(c2, lo, hi)
                    children[i] = c1
                    if i + 1 < n_children:
                        children[i + 1] = c2
                    used += 1
                    child_F[i] = problem(c1)
                    if i + 1 < n_children:
                        used += 1
                        child_F[i + 1] = problem(c2)
                    i += 1  # Skip next since we handled both
                else:
                    # Mutation only
                    p = parents[i]
                    c = p + 0.1 * (hi - lo) * rng.standard_normal(dim)
                    c = np.clip(c, lo, hi)
                    children[i] = c
                    child_F[i] = problem(c)
                    used += 1

            # Combine parents and children
            X_new = np.vstack([X, children])
            F_new = np.vstack([F, child_F])

            # Non-dominated sort on combined population
            fronts_new = fast_non_dominated_sort(F_new)

            # Select next generation
            X_next = []
            F_next = []
            total_selected = 0
            for front in fronts_new:
                if total_selected + len(front) <= pop_size:
                    X_next.append(X_new[front])
                    F_next.append(F_new[front])
                    total_selected += len(front)
                else:
                    # Need to select subset of this front based on crowding distance
                    remaining = pop_size - total_selected
                    cd = crowding_distance(F_new, front)
                    order = np.argsort(-cd)
                    selected = front[order[:remaining]]
                    X_next.append(X_new[selected])
                    F_next.append(F_new[selected])
                    total_selected += remaining
                    break

            X = np.vstack(X_next)
            F = np.vstack(F_next)

        return X
# EVOLVE-BLOCK-END
