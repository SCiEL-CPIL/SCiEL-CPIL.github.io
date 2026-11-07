"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-objective optimizer on ZDT6 using structured evolutionary search."""

    def __init__(self, budget, dim, n_obj):
        self.budget, self.dim, self.n_obj = budget, dim, n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280  # safety margin before 330s limit
        rng = np.random.default_rng(0)
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        dim = self.dim
        budget = self.budget

        # ZDT6 has a specific structure:
        # g(x) = 1 + 10*n
        # f1 = 1 - sqrt(x1)
        # f2 = g * h(f1, g)
        # where h = 1 - sqrt(f1/g) - (f1/g)*sin(10*pi*f1)
        #
        # The Pareto front for ZDT6 is not uniformly distributed.
        # We'll use a multi-objective evolutionary algorithm (MOEA/D-like or NSGA-II style)
        # but with a twist: exploit the structure by focusing on x1 for f1 and
        # the rest for g.

        # Population size
        pop_size = min(50, max(20, budget // 4))

        # Initialize population
        X = np.zeros((pop_size, dim))
        for i in range(pop_size):
            X[i, 0] = rng.random()  # x1 controls f1
            X[i, 1:] = rng.random(dim - 1)  # rest controls g

        F = np.array([problem(x) for x in X])
        used = pop_size

        # Track all evaluated points for hypervolume
        all_points = [F[i].tolist() for i in range(pop_size)]

        # NSGA-II style: non-dominated sorting + crowding distance
        def non_dominated_sort(F):
            n = len(F)
            S = [[] for _ in range(n)]  # S[i] = indices dominated by i
            n_dom = np.zeros(n, dtype=int)  # number of points dominating i
            rank = np.zeros(n, dtype=int)
            front = [0]  # current front index

            for i in range(n):
                for j in range(n):
                    if i == j:
                        continue
                    # i dominates j?
                    if np.all(F[i] <= F[j]) and np.any(F[i] < F[j]):
                        S[i].append(j)
                    elif np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                        n_dom[i] += 1
            # First front
            front0 = [i for i in range(n) if n_dom[i] == 0]
            rank[front0] = 0
            k = 0
            while front0:
                next_front = []
                for i in front0:
                    for j in S[i]:
                        n_dom[j] -= 1
                        if n_dom[j] == 0:
                            rank[j] = k + 1
                            next_front.append(j)
                k += 1
                front0 = next_front
            return rank

        def crowding_distance(F, rank, front_size):
            n = len(F)
            cd = np.zeros(n)
            # For each front, compute crowding
            for r in range(max(rank) + 1):
                members = [i for i in range(n) if rank[i] == r]
                if len(members) <= 2:
                    for i in members:
                        cd[i] = np.inf
                    continue
                for m in range(F.shape[1]):
                    vals = F[members, m]
                    min_v, max_v = vals.min(), vals.max()
                    if max_v - min_v < 1e-10:
                        continue
                    # Sort by objective m
                    order = np.argsort(vals)
                    members_sorted = [members[k] for k in order]
                    cd[members_sorted[0]] = np.inf
                    cd[members_sorted[-1]] = np.inf
                    for idx in range(1, len(members_sorted) - 1):
                        i = members_sorted[idx]
                        prev_i = members_sorted[idx - 1]
                        next_i = members_sorted[idx + 1]
                        cd[i] += (F[next_i, m] - F[prev_i, m]) / (max_v - min_v)
            return cd

        # Main loop
        while used < budget:
            # Check time
            if time.time() > deadline:
                break

            # Non-dominated sort
            rank = non_dominated_sort(F)

            # Select parents from best fronts
            # Use tournament selection based on rank and crowding
            cd = crowding_distance(F, rank, pop_size)

            # Create child
            # Pick two parents using binary tournament
            def tournament():
                i1 = rng.integers(pop_size)
                i2 = rng.integers(pop_size)
                if rank[i1] < rank[i2]:
                    return i1
                elif rank[i2] < rank[i1]:
                    return i2
                else:
                    return i1 if cd[i1] > cd[i2] else i2

            p1 = tournament()
            p2 = tournament()

            # Simulated binary crossover (SBX)
            child = X[p1].copy()
            if rng.random() < 0.9:
                for d in range(dim):
                    if rng.random() < 0.5:
                        # SBX
                        u = rng.random()
                        if u <= 0.5:
                            beta = (2 * u) ** (1 / (2 + 1))
                        else:
                            beta = (1 / (2 * (1 - u))) ** (1 / (2 + 1))
                        if rng.random() < 0.5:
                            child[d] = 0.5 * ((1 + beta) * X[p1, d] + (1 - beta) * X[p2, d])
                        else:
                            child[d] = 0.5 * ((1 - beta) * X[p1, d] + (1 + beta) * X[p2, d])

            # Polynomial mutation
            if rng.random() < 0.9:
                for d in range(dim):
                    if rng.random() < 1.0 / dim:
                        u = rng.random()
                        delta = (1 + 2 * (child[d] - lo[d]) / (hi[d] - lo[d])) ** (-1 / (2 + 1)) if u <= 0.5 else \
                                (1 + 2 * (hi[d] - child[d]) / (hi[d] - lo[d])) ** (-1 / (2 + 1))
                        if u <= 0.5:
                            child[d] += delta * (hi[d] - lo[d])
                        else:
                            child[d] -= delta * (hi[d] - lo[d])

            child = np.clip(child, lo, hi)
            fc = problem(child)
            used += 1
            all_points.append(fc.tolist())

            # Replace worst individual in population
            # Find the worst by rank, then crowding
            worst_idx = None
            max_rank = max(rank)
            worst_candidates = [i for i in range(pop_size) if rank[i] == max_rank]
            if worst_candidates:
                worst_idx = worst_candidates[rng.integers(len(worst_candidates))]
            else:
                worst_idx = rng.integers(pop_size)

            X[worst_idx] = child
            F[worst_idx] = fc

        return X
# EVOLVE-BLOCK-END
