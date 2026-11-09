"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """NSGA-II based multi-objective optimizer for WFG9."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 300.0  # Safety margin before 330s kill

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        dim = self.dim
        n_obj = self.n_obj
        budget = self.budget

        # Population size
        pop_size = min(40, budget // 2)
        if pop_size < 10:
            pop_size = 10

        # Latin Hypercube Sampling for initial population
        def lhs_sample(n, d):
            """Latin Hypercube Sampling."""
            samples = np.zeros((n, d))
            for j in range(d):
                perm = np.random.permutation(n)
                samples[:, j] = (perm + np.random.random(n)) / n
            return samples

        # Use the global numpy random state (seeded by evaluator)
        X0 = lo + (hi - lo) * lhs_sample(pop_size, dim)
        F0 = np.array([problem(x) for x in X0])
        used = pop_size

        # Archive: keep all non-dominated points found
        archive_X = [X0[i] for i in range(pop_size)]
        archive_F = [F0[i] for i in range(pop_size)]

        def is_dominated(f1, f2):
            """Check if f1 dominates f2 (minimization)."""
            return np.all(f1 <= f2) and np.any(f1 < f2)

        def non_dominated_sort(F):
            """Return list of fronts (each front is list of indices)."""
            n = len(F)
            S = [[] for _ in range(n)]  # S[i] = indices dominated by i
            n_dom = [0] * n  # number of solutions that dominate i
            rank = [0] * n
            fronts = []

            for i in range(n):
                for j in range(n):
                    if i == j:
                        continue
                    if is_dominated(F[i], F[j]):
                        S[i].append(j)
                    elif is_dominated(F[j], F[i]):
                        n_dom[i] += 1
                if n_dom[i] == 0:
                    rank[i] = 0
                    fronts.append([i])

            k = 0
            while fronts[k]:
                next_front = []
                for i in fronts[k]:
                    for j in S[i]:
                        n_dom[j] -= 1
                        if n_dom[j] == 0:
                            rank[j] = k + 1
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
            dist = [0.0] * n
            for m in range(n_obj):
                fvals = np.array([F[i][m] for i in front_indices])
                sorted_idx = np.argsort(fvals)
                dist[sorted_idx[0]] = float('inf')
                dist[sorted_idx[-1]] = float('inf')
                f_range = fvals[sorted_idx[-1]] - fvals[sorted_idx[0]]
                if f_range > 1e-10:
                    for k in range(1, n - 1):
                        dist[sorted_idx[k]] += (fvals[sorted_idx[k + 1]] - fvals[sorted_idx[k - 1]]) / f_range
            return dist

        def sbx_crossover(x1, x2, eta=20.0):
            """Simulated Binary Crossover."""
            child1 = np.zeros_like(x1)
            child2 = np.zeros_like(x2)
            for i in range(dim):
                if np.random.random() < 0.5:
                    if abs(x1[i] - x2[i]) < 1e-14:
                        child1[i] = x1[i]
                        child2[i] = x2[i]
                    else:
                        if np.random.random() <= 0.5:
                            beta = (2 * np.random.random()) ** (1 / (eta + 1))
                        else:
                            beta = (1 / (2 * np.random.random())) ** (1 / (eta + 1))
                        child1[i] = 0.5 * ((1 + beta) * x1[i] + (1 - beta) * x2[i])
                        child2[i] = 0.5 * ((1 - beta) * x1[i] + (1 + beta) * x2[i])
                else:
                    child1[i] = x2[i]
                    child2[i] = x1[i]
            return child1, child2

        def polynomial_mutation(x, eta=20.0):
            """Polynomial Mutation."""
            y = x.copy()
            for i in range(dim):
                if np.random.random() < 1.0 / dim:
                    y[i] = x[i]
                    delta = hi[i] - lo[i]
                    if delta < 1e-14:
                        continue
                    if np.random.random() <= 0.5:
                        val = 2 * np.random.random()
                        mut_pow = 2 + 1 / (eta + 1)
                        y[i] = x[i] + delta * ((val) ** mut_pow - 1)
                    else:
                        val = 1 - 2 * np.random.random()
                        mut_pow = 2 + 1 / (eta + 1)
                        y[i] = x[i] + delta * (1 - (val) ** mut_pow)
                    y[i] = np.clip(y[i], lo[i], hi[i])
            return y

        # Main loop
        X = X0.copy()
        F = F0.copy()

        while used < budget:
            # Check time
            if time.time() > deadline - 5.0:
                break

            # Non-dominated sort
            fronts = non_dominated_sort(F)

            # Select parents using tournament selection on fronts
            parents = []
            while len(parents) < 2:
                # Tournament: pick 2 random, choose one from better front
                i1 = np.random.randint(len(F))
                i2 = np.random.randint(len(F))
                # Find front ranks
                rank1 = 0
                rank2 = 0
                for k, front in enumerate(fronts):
                    if i1 in front:
                        rank1 = k
                        break
                    if i2 in front:
                        rank2 = k
                        break
                if rank1 < rank2:
                    parent = i1
                elif rank2 < rank1:
                    parent = i2
                else:
                    # Same front, use crowding distance
                    front_idx = rank1
                    front = fronts[front_idx]
                    cd = crowding_distance(F, front)
                    # Map indices to positions in front
                    pos1 = front.index(i1)
                    pos2 = front.index(i2)
                    parent = i1 if cd[pos1] >= cd[pos2] else i2
                if parent not in parents:
                    parents.append(parent)
                elif len(parents) == 1:
                    # Need different parent
                    pass
                if len(parents) >= 2:
                    break

            # Crossover
            c1, c2 = sbx_crossover(X[parents[0]], X[parents[1]])
            # Mutation
            c1 = polynomial_mutation(c1)
            c2 = polynomial_mutation(c2)

            # Evaluate children
            fc1 = problem(c1)
            used += 1
            fc2 = problem(c2)
            used += 1

            # Add to archive
            archive_X.append(c1)
            archive_F.append(fc1)
            archive_X.append(c2)
            archive_F.append(fc2)

            # Form new population: combine parents and children
            # Keep best pop_size based on non-dominated sorting and crowding
            new_X = np.vstack([X, c1.reshape(1, -1), c2.reshape(1, -1)])
            new_F = np.vstack([F, fc1.reshape(1, -1), fc2.reshape(1, -1)])

            # Truncate to pop_size
            new_fronts = non_dominated_sort(new_F)
            selected = []
            for front in new_fronts:
                if len(selected) + len(front) <= pop_size:
                    selected.extend(front)
                else:
                    # Need to pick from this front based on crowding
                    cd = crowding_distance(new_F, front)
                    remaining = pop_size - len(selected)
                    sorted_front = sorted(range(len(front)), key=lambda i: cd[i], reverse=True)
                    for i in sorted_front[:remaining]:
                        selected.append(front[i])
                    break

            X = new_X[selected]
            F = new_F[selected]

        return X
# EVOLVE-BLOCK-END
