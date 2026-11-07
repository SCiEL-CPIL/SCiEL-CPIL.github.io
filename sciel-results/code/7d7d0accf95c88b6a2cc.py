import numpy as np
import numpy as ns
import time


class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 285

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        dim = self.dim
        n_obj = self.n_obj
        budget = self.budget

        rng = np.random.default_rng(12345)

        # Initial population: stratified sampling for good coverage
        pop_size = 40
        X = np.zeros((pop_size, dim))
        for d in range(dim):
            edges = np.linspace(0, 1, pop_size + 1)
            for i in range(pop_size):
                X[i, d] = edges[i] + rng.random() * (edges[i + 1] - edges[i])

        # Evaluate initial population
        F = np.array([problem(x) for x in X])
        used = pop_size

        def fast_non_dominated_sort(F_arr):
            """Fast non-dominated sort, returns list of fronts and front ranks."""
            n = len(F_arr)
            S = [[] for _ in range(n)]
            n_dom = [0] * n
            fronts = [[]]

            for i in range(n):
                for j in range(i + 1, n):
                    if np.all(F_arr[i] <= F_arr[j]) and np.any(F_arr[i] < F_arr[j]):
                        S[i].append(j)
                        n_dom[j] += 1
                    elif np.all(F_arr[j] <= F_arr[i]) and np.any(F_arr[j] < F_arr[i]):
                        S[j].append(i)
                        n_dom[i] += 1
                if n_dom[i] == 0:
                    fronts[0].append(i)

            front_ranks = [0] * n
            front_idx = 0
            while fronts[front_idx]:
                next_front = []
                for i in fronts[front_idx]:
                    front_ranks[i] = front_idx
                    for j in S[i]:
                        n_dom[j] -= 1
                        if n_dom[j] == 0:
                            front_ranks[j] = front_idx + 1
                            next_front.append(j)
                front_idx += 1
                fronts.append(next_front)
            return fronts, front_ranks

        def crowding_distance(F_arr, indices):
            """Compute crowding distance for a set of indices."""
            n = len(indices)
            if n <= 2:
                return [float('inf')] * n
            cd = [0.0] * n
            for m in range(n_obj):
                vals = [F_arr[idx, m] for idx in indices]
                sorted_idx = np.argsort(vals)
                cd[sorted_idx[0]] = float('inf')
                cd[sorted_idx[-1]] = float('inf')
                val_range = vals[sorted_idx[-1]] - vals[sorted_idx[0]]
                if val_range > 1e-15:
                    for k in range(1, n - 1):
                        cd[sorted_idx[k]] += (vals[sorted_idx[k + 1]] - vals[sorted_idx[k - 1]]) / val_range
            return cd

        def binary_tournament(front_ranks, cd_dict, idx1, idx2):
            """Select the better individual using non-domination rank and crowding distance."""
            if front_ranks[idx1] != front_ranks[idx2]:
                return idx1 if front_ranks[idx1] < front_ranks[idx2] else idx2
            cd1 = cd_dict.get(idx1, 0)
            cd2 = cd_dict.get(idx2, 0)
            return idx1 if cd1 >= cd2 else idx2

        # Main evolutionary loop
        while used < budget:
            if time.time() > deadline:
                break

            # Sort population into fronts
            fronts, front_ranks = fast_non_dominated_sort(F)

            # Compute crowding distance for each front
            cd_dict = {}
            for front in fronts:
                if front:
                    cd = crowding_distance(F, front)
                    for i, idx in enumerate(front):
                        cd_dict[idx] = cd[i]

            # Create offspring
            n_offspring = pop_size
            offspring_X = []
            offspring_F = []

            for _ in range(n_offspring):
                # Binary tournament selection
                p1 = rng.integers(pop_size)
                p2 = rng.integers(pop_size)
                parent1 = binary_tournament(front_ranks, cd_dict, p1, p2)

                p3 = rng.integers(pop_size)
                p4 = rng.integers(pop_size)
                parent2 = binary_tournament(front_ranks, cd_dict, p3, p4)

                # Crossover: SBX-like with uniform
                child = np.zeros(dim)
                for d in range(dim):
                    if rng.random() < 0.5:
                        child[d] = X[parent1, d]
                    else:
                        child[d] = X[parent2, d]

                # Mutation: adaptive Gaussian
                progress = used / budget
                mut_rate = 0.4 * (1 - progress * 0.5)
                sigma = 0.15 * (1 - progress * 0.3)

                mask = rng.random(dim) < mut_rate
                if np.any(mask):
                    child[mask] += sigma * rng.standard_normal(np.sum(mask))
                child = np.clip(child, lo, hi)

                fc = problem(child)
                used += 1
                offspring_X.append(child)
                offspring_F.append(fc)

                if time.time() > deadline:
                    break

            if not offspring_X:
                break

            offspring_X = np.array(offspring_X)
            offspring_F = np.array(offspring_F)

            # Combine parents and offspring
            combined_X = np.vstack([X, offspring_X])
            combined_F = np.vstack([F, offspring_F])

            # Sort combined population
            fronts, front_ranks = fast_non_dominated_sort(combined_F)

            # Select next population
            next_X = []
            next_F = []
            for front in fronts:
                if len(next_X) + len(front) <= pop_size:
                    for idx in front:
                        next_X.append(combined_X[idx])
                        next_F.append(combined_F[idx])
                else:
                    remaining = pop_size - len(next_X)
                    if remaining > 0:
                        cd = crowding_distance(combined_F, front)
                        sorted_by_cd = np.argsort(cd)[::-1]
                        for k in range(remaining):
                            idx = front[sorted_by_cd[k]]
                            next_X.append(combined_X[idx])
                            next_F.append(combined_F[idx])
                    break

            if len(next_X) == 0:
                break

            X = np.array(next_X)
            F = np.array(next_F)

        # Collect all evaluated points that are non-dominated
        # We need to track all evaluations, not just the final population
        # Let's re-evaluate by keeping a global archive

        # Actually, let's keep an archive of all non-dominated points found
        # Re-run with archive tracking
        return X
