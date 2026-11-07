"""NSGA-II-based multi-objective optimizer for WFG9."""
import numpy as np
import time


class Optimizer:
    """NSGA-II multi-objective optimizer for WFG9."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def _fast_non_dominated_sort(self, F):
        """
        Perform fast non-dominated sorting.
        Returns a list of fronts, where each front is a list of indices.
        """
        n = len(F)
        S = [set() for _ in range(n)]  # Dominance set
        n_dom = [0] * n  # Number of solutions dominated by i
        fronts = []
        front0 = set()

        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                # Check if i dominates j
                if np.all(F[i] <= F[j]) and np.any(F[i] < F[j]):
                    S[i].add(j)
                # Check if j dominates i
                elif np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                    n_dom[i] += 1

            if n_dom[i] == 0:
                front0.add(i)

        fronts.append(list(front0))
        k = 0
        while fronts[k]:
            next_front = set()
            for i in fronts[k]:
                for j in S[i]:
                    n_dom[j] -= 1
                    if n_dom[j] == 0:
                        next_front.add(j)
            if next_front:
                fronts.append(list(next_front))
            k += 1

        return fronts

    def _crowding_distance(self, F, indices):
        """
        Calculate crowding distance for solutions in a front.
        """
        if len(indices) <= 2:
            return {i: float('inf') for i in indices}

        m = len(F[indices[0]])
        dist = {i: 0.0 for i in indices}

        for k in range(m):
            # Sort by k-th objective
            sorted_indices = sorted(indices, key=lambda i: F[i][k])
            dist[sorted_indices[0]] = float('inf')
            dist[sorted_indices[-1]] = float('inf')

            f_min = F[sorted_indices[0]][k]
            f_max = F[sorted_indices[-1]][k]
            range_f = f_max - f_min

            if range_f == 0:
                continue

            for i in range(1, len(sorted_indices) - 1):
                idx = sorted_indices[i]
                dist[idx] += (F[sorted_indices[i + 1]][k] - F[sorted_indices[i - 1]][k]) / range_f

        return dist

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280.0  # 280s safety margin

        dim = self.dim
        n_obj = self.n_obj
        budget = self.budget

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        rng = np.random.default_rng(0)

        # Population size
        # We have 200 evaluations. Let's use N=20 for 10 generations (1 init + 9 offspring gens? No, standard is N initial, then loop N offspring).
        # Total evals = N + G*N. If N=20, G=9 -> 200.
        N = 20
        
        # Initialize population
        X = lo + (hi - lo) * rng.random((N, dim))
        F = np.array([problem(x) for x in X])
        used = N

        while used < budget:
            if time.time() > deadline:
                break

            # Pre-compute ranks and crowding for current X, F
            fronts = self._fast_non_dominated_sort(F)
            rank = np.zeros(N, dtype=int)
            for i, front in enumerate(fronts):
                for idx in front:
                    rank[idx] = i
                
            # Crowding distance per front
            cd = np.zeros(N)
            for front in fronts:
                if len(front) > 0:
                    cd_front = self._crowding_distance(F, front)
                    for idx in front:
                        cd[idx] = cd_front[idx]

            # Selection function
            def select():
                i1 = rng.integers(N)
                i2 = rng.integers(N)
                if rank[i1] < rank[i2]:
                    return i1
                elif rank[i2] < rank[i1]:
                    return i2
                else:
                    if cd[i1] > cd[i2]:
                        return i1
                    else:
                        return i2

            # Generate offspring
            offspring_X = []
            offspring_F = []

            for _ in range(N):
                if used >= budget:
                    break
                if time.time() > deadline:
                    break

                p1 = select()
                p2 = select()

                # Crossover: Simulated Binary Crossover (SBX) or simple blend
                if rng.random() < 0.9:
                    # SBX-like
                    alpha = rng.random()
                    child = (1 - alpha) * X[p1] + alpha * X[p2]
                    # Add Gaussian noise
                    sigma = 0.1 * (hi - lo)
                    child = child + sigma * rng.standard_normal(dim)
                    child = np.clip(child, lo, hi)
                else:
                    # Mutation only
                    parent = X[p1] if rng.random() < 0.5 else X[p2]
                    child = parent + 0.1 * (hi - lo) * rng.standard_normal(dim)
                    child = np.clip(child, lo, hi)

                fc = problem(child)
                used += 1
                offspring_X.append(child)
                offspring_F.append(fc)

            if not offspring_X:
                break

            # Combine parent and offspring
            X_combined = np.vstack([X, np.array(offspring_X)])
            F_combined = np.vstack([F, np.array(offspring_F)])

            # Non-dominated sorting
            fronts = self._fast_non_dominated_sort(F_combined)

            # Select next generation
            next_X = []
            next_F = []
            
            for front in fronts:
                if len(next_X) + len(front) <= N:
                    # Add all
                    for idx in front:
                        next_X.append(X_combined[idx])
                        next_F.append(F_combined[idx])
                else:
                    # Need to select subset
                    cd_front = self._crowding_distance(F_combined, front)
                    sorted_front = sorted(front, key=lambda i: cd_front[i], reverse=True)
                    slots_needed = N - len(next_X)
                    for idx in sorted_front[:slots_needed]:
                        next_X.append(X_combined[idx])
                        next_F.append(F_combined[idx])
                    break

            if len(next_X) == 0:
                break

            X = np.array(next_X)
            F = np.array(next_F)

        return X
