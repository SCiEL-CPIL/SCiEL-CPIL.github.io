# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-objective optimizer using NSGA-II-like approach with deadline guard."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 290.0  # Safety margin before 330s limit

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        range_vec = hi - lo

        # Population size
        pop_size = min(50, self.budget // 4)
        if pop_size < 10:
            pop_size = 10

        # Initialize population
        X = lo + range_vec * np.random.rand(pop_size, self.dim)
        F = np.array([problem(x) for x in X])
        used = pop_size

        # Reference point for hypervolume calculation
        ref = np.array([3.0, 5.0, 7.0])

        def dominates(a, b):
            """Check if a dominates b (a is better than or equal in all, better in at least one)"""
            return np.all(a <= b) and np.any(a < b)

        def fast_non_dominated_sort(F):
            """NSGA-II fast non-dominated sort"""
            n = len(F)
            S = [[] for _ in range(n)]  # S[i] = indices dominated by i
            n_doms = np.zeros(n, dtype=int)  # number of solutions dominating i
            fronts = [[]]

            for i in range(n):
                for j in range(n):
                    if i == j:
                        continue
                    if dominates(F[i], F[j]):
                        S[i].append(j)
                    elif dominates(F[j], F[i]):
                        n_doms[i] += 1

                if n_doms[i] == 0:
                    fronts[0].append(i)

            k = 0
            while len(fronts[k]) > 0:
                next_front = []
                for i in fronts[k]:
                    for j in S[i]:
                        n_doms[j] -= 1
                        if n_doms[j] == 0:
                            next_front.append(j)
                k += 1
                if next_front:
                    fronts.append(next_front)

            return fronts

        def crowding_distance(F, indices):
            """Calculate crowding distance for a front"""
            n = len(indices)
            if n <= 2:
                return np.full(n, np.inf)

            cd = np.zeros(n)
            for m in range(self.n_obj):
                # Sort by objective m
                sorted_idx = sorted(range(n), key=lambda i: F[indices[i]][m])
                cd[sorted_idx[0]] = np.inf
                cd[sorted_idx[-1]] = np.inf

                min_val = F[indices[sorted_idx[0]]][m]
                max_val = F[indices[sorted_idx[-1]]][m]
                span = max_val - min_val

                if span > 0:
                    for i in range(1, n - 1):
                        cd[sorted_idx[i]] += (
                            F[indices[sorted_idx[i + 1]]][m] -
                            F[indices[sorted_idx[i - 1]]][m]
                        ) / span

            return cd

        # Main evolutionary loop
        while used < self.budget:
            # Check deadline
            if time.time() > deadline:
                break

            # Non-dominated sort
            fronts = fast_non_dominated_sort(F)

            # Selection: choose parents from best fronts
            parents_pool = []
            for front in fronts:
                if len(parents_pool) + len(front) <= pop_size:
                    parents_pool.extend(front)
                else:
                    # Use crowding distance to select
                    cd = crowding_distance(F, front)
                    sorted_front = sorted(front, key=lambda i: cd[front.index(i)], reverse=True)
                    parents_pool.extend(sorted_front[:pop_size - len(parents_pool)])
                    break

            # Crossover and mutation to create offspring
            offspring_X = []
            offspring_F = []

            while len(offspring_X) < pop_size and used < self.budget:
                if time.time() > deadline:
                    break

                # Select two parents
                p1_idx = np.random.randint(len(parents_pool))
                p2_idx = np.random.randint(len(parents_pool))
                p1 = X[parents_pool[p1_idx]]
                p2 = X[parents_pool[p2_idx]]

                # Simulated binary crossover (SBX)
                child1 = np.zeros(self.dim)
                child2 = np.zeros(self.dim)
                for d in range(self.dim):
                    if np.random.rand() < 0.5:
                        beta = 1.0
                        if p1[d] != p2[d]:
                            u = np.random.rand()
                            if u <= 0.5:
                                beta = (2 * u) ** (1 / (2 + 1))
                            else:
                                beta = (1 / (2 * (1 - u))) ** (1 / (2 + 1))
                        child1[d] = 0.5 * ((1 + beta) * p1[d] + (1 - beta) * p2[d])
                        child2[d] = 0.5 * ((1 - beta) * p1[d] + (1 + beta) * p2[d])
                    else:
                        child1[d] = p1[d]
                        child2[d] = p2[d]

                # Polynomial mutation
                for d in range(self.dim):
                    if np.random.rand() < 1.0 / self.dim:
                        delta = np.random.rand()
                        if delta < 0.5:
                            delta = (2 * delta) ** (1 / (2 + 1)) - 1
                        else:
                            delta = 1 - (2 * (1 - delta)) ** (1 / (2 + 1))
                        child1[d] += delta * range_vec[d] * 0.1
                    if np.random.rand() < 1.0 / self.dim:
                        delta = np.random.rand()
                        if delta < 0.5:
                            delta = (2 * delta) ** (1 / (2 + 1)) - 1
                        else:
                            delta = 1 - (2 * (1 - delta)) ** (1 / (2 + 1))
                        child2[d] += delta * range_vec[d] * 0.1

                child1 = np.clip(child1, lo, hi)
                child2 = np.clip(child2, lo, hi)

                offspring_X.append(child1)
                offspring_F.append(problem(child1))
                used += 1

                if used < self.budget and time.time() < deadline:
                    offspring_X.append(child2)
                    offspring_F.append(problem(child2))
                    used += 1

            if not offspring_X:
                break

            # Combine parent and offspring populations
            combined_X = np.vstack([X, np.array(offspring_X)])
            combined_F = np.vstack([F, np.array(offspring_F)])

            # Non-dominated sort on combined population
            fronts = fast_non_dominated_sort(combined_F)

            # Select next generation
            new_X = []
            new_F = []
            for front in fronts:
                if len(new_X) + len(front) <= pop_size:
                    for idx in front:
                        new_X.append(combined_X[idx])
                        new_F.append(combined_F[idx])
                else:
                    cd = crowding_distance(combined_F, front)
                    sorted_front = sorted(front, key=lambda i: cd[front.index(i)], reverse=True)
                    for idx in sorted_front[:pop_size - len(new_X)]:
                        new_X.append(combined_X[idx])
                        new_F.append(combined_F[idx])
                    break

            if len(new_X) == 0:
                break

            X = np.array(new_X)
            F = np.array(new_F)

        return X
# EVOLVE-BLOCK-END
