"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """NSGA-II style multi-objective optimizer for DTLZ7."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280  # safety margin before 300s limit

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        dim = self.dim
        n_obj = self.n_obj
        budget = self.budget

        # Population size
        pop_size = 100

        # Initialize population with random points
        rng = np.random.default_rng(0)
        X = lo + (hi - lo) * rng.random((pop_size, dim))
        F = np.array([problem(x) for x in X])
        used = pop_size

        # Track best known points for hypervolume
        best_points = []

        def fast_non_dominated_sort(F):
            """Fast non-dominated sort. Returns list of fronts (lists of indices)."""
            n = len(F)
            S = [set() for _ in range(n)]  # S[i] = set of indices dominated by i
            n_dom = [0] * n  # n_dom[i] = number of solutions dominating i
            fronts = [[]]

            for i in range(n):
                for j in range(i + 1, n):
                    if np.all(F[i] <= F[j]) and np.any(F[i] < F[j]):
                        S[i].add(j)
                        n_dom[j] += 1
                    elif np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                        S[j].add(i)
                        n_dom[i] += 1
                if n_dom[i] == 0:
                    fronts[0].append(i)

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
            """Compute crowding distance for a front."""
            n = len(front_indices)
            if n <= 2:
                return [float('inf')] * n
            cd = [0.0] * n
            for m in range(n_obj):
                sorted_idx = sorted(front_indices, key=lambda i: F[i][m])
                cd[sorted_idx[0]] = float('inf')
                cd[sorted_idx[-1]] = float('inf')
                fmin = F[sorted_idx[0]][m]
                fmax = F[sorted_idx[-1]][m]
                if fmax - fmin > 1e-15:
                    for i in range(1, n - 1):
                        cd[sorted_idx[i]] += (F[sorted_idx[i + 1]][m] - F[sorted_idx[i - 1]][m]) / (fmax - fmin)
            return cd

        def select_parents(F, X):
            """Select parents using tournament selection based on rank and crowding."""
            fronts = fast_non_dominated_sort(F)
            n = len(F)
            rank = [0] * n
            for fi, front in enumerate(fronts):
                for idx in front:
                    rank[idx] = fi

            # Compute crowding distance for each front
            cd = [0.0] * n
            for front in fronts:
                front_cd = crowding_distance(F, front)
                for i, idx in enumerate(front):
                    cd[idx] = front_cd[i]

            # Tournament selection
            parents = []
            while len(parents) < n:
                # Tournament of size 2
                i = rng.integers(n)
                j = rng.integers(n)
                if rank[i] < rank[j]:
                    parents.append(i)
                elif rank[j] < rank[i]:
                    parents.append(j)
                else:
                    if cd[i] > cd[j]:
                        parents.append(i)
                    else:
                        parents.append(j)
            return parents

        def crossover(x1, x2):
            """SBX crossover."""
            if rng.random() < 0.9:
                eta = 20.0
                child1 = np.empty(dim)
                child2 = np.empty(dim)
                for d in range(dim):
                    if rng.random() < 0.5:
                        u = rng.random()
                        if u <= 1.0 / (eta + 1):
                            beta = (2 * u) ** (1.0 / (eta + 1))
                        elif u >= 1.0 - 1.0 / (eta + 1):
                            beta = (2 * (1 - u)) ** (1.0 / (eta + 1))
                        else:
                            beta = 1.0
                        child1[d] = 0.5 * ((1 + beta) * x1[d] + (1 - beta) * x2[d])
                        child2[d] = 0.5 * ((1 - beta) * x1[d] + (1 + beta) * x2[d])
                    else:
                        child1[d] = x2[d]
                        child2[d] = x1[d]
                return child1, child2
            else:
                return x1.copy(), x2.copy()

        def mutate(x):
            """Polynomial mutation."""
            eta = 20.0
            x_mut = x.copy()
            for d in range(dim):
                if rng.random() < 1.0 / dim:
                    u = rng.random()
                    xl = lo[d]
                    xu = hi[d]
                    delta = xu - xl
                    if delta > 1e-15:
                        if u <= 0.5:
                            delta1 = (2 * u) ** (1.0 / (eta + 1)) - 1
                        else:
                            delta1 = 1 - (2 * (1 - u)) ** (1.0 / (eta + 1))
                        x_mut[d] += delta1 * delta
            return np.clip(x_mut, lo, hi)

        # Main loop
        while used < budget:
            # Check time
            if time.time() > deadline:
                break

            # Select parents
            parents = select_parents(F, X)

            # Generate children
            children_X = []
            children_F = []
            n_children = min(pop_size, budget - used)

            for i in range(0, n_children, 2):
                p1 = parents[i % len(parents)]
                p2 = parents[(i + 1) % len(parents)]
                c1, c2 = crossover(X[p1], X[p2])
                c1 = mutate(c1)
                c2 = mutate(c2)

                if i % 2 == 0 and used + 1 <= budget:
                    fc1 = problem(c1)
                    children_X.append(c1)
                    children_F.append(fc1)
                    used += 1

                if used + 1 <= budget:
                    fc2 = problem(c2)
                    children_X.append(c2)
                    children_F.append(fc2)
                    used += 1

            if not children_X:
                break

            # Combine parent and child populations
            X_comb = np.vstack([X, np.array(children_X)])
            F_comb = np.vstack([F, np.array(children_F)])

            # Fast non-dominated sort on combined population
            fronts = fast_non_dominated_sort(F_comb)

            # Select next generation
            new_X = []
            new_F = []
            for front in fronts:
                if len(new_X) + len(front) <= pop_size:
                    for idx in front:
                        new_X.append(X_comb[idx])
                        new_F.append(F_comb[idx])
                else:
                    # Sort by crowding distance and take the best
                    front_cd = crowding_distance(F_comb, front)
                    sorted_front = sorted(front, key=lambda i: front_cd[i], reverse=True)
                    needed = pop_size - len(new_X)
                    for idx in sorted_front[:needed]:
                        new_X.append(X_comb[idx])
                        new_F.append(F_comb[idx])
                    break

            X = np.array(new_X)
            F = np.array(new_F)

            # Update best points (keep non-dominated set for tracking)
            # We'll just track all evaluated points that are non-dominated in the final population

        # At the end, return the current population
        return X

# EVOLVE-BLOCK-END
