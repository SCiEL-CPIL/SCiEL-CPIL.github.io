"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """NSGA-II style multi-objective optimizer for ZDT2."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280.0  # Safety margin before 300s limit

        lo = np.array([b[0] for b in bounds], dtype=float)
        hi = np.array([b[1] for b in bounds], dtype=float)
        n_dim = self.dim
        n_obj = self.n_obj
        budget = self.budget

        # Population size - larger for better exploration
        pop_size = min(50, max(20, budget // 4))
        
        # Initial population
        rng = np.random.default_rng(42)
        X = lo + (hi - lo) * rng.random((pop_size, n_dim))
        F = np.array([problem(x) for x in X])
        used = pop_size

        # Track all evaluated points for hypervolume
        all_points = list(zip(F[:, 0], F[:, 1]))

        def fast_non_dominated_sort(F):
            """Fast non-dominated sorting."""
            n = len(F)
            S = [[] for _ in range(n)]  # Solutions dominated by i
            n_count = np.zeros(n, dtype=int)  # Number of solutions dominating i
            rank = np.zeros(n, dtype=int)
            front = [0]  # List of fronts, front[0] is list of indices in first front

            for p in range(n):
                for q in range(n):
                    if p == q:
                        continue
                    if np.all(F[p] <= F[q]) and np.any(F[p] < F[q]):
                        S[p].append(q)
                    elif np.all(F[q] <= F[p]) and np.any(F[q] < F[p]):
                        n_count[p] += 1
                if n_count[p] == 0:
                    rank[p] = 0
                    front[0].append(p)

            i = 0
            while len(front[i]) > 0:
                next_front = []
                for p in front[i]:
                    for q in S[p]:
                        n_count[q] -= 1
                        if n_count[q] == 0:
                            rank[q] = i + 1
                            next_front.append(q)
                i += 1
                front.append(next_front)
                if not next_front:
                    break

            return rank, front

        def crowding_distance(F, front_indices):
            """Calculate crowding distance for a front."""
            n = len(front_indices)
            if n <= 2:
                return np.full(n, np.inf)

            dist = np.zeros(n)
            for m in range(n_obj):
                f_vals = F[front_indices, m]
                f_min = f_vals.min()
                f_max = f_vals.max()
                if f_max - f_min < 1e-10:
                    dist[:] = np.inf
                    continue
                order = np.argsort(f_vals)
                dist[order[0]] = np.inf
                dist[order[-1]] = np.inf
                for i in range(1, n - 1):
                    dist[order[i]] += (f_vals[order[i + 1]] - f_vals[order[i - 1]]) / (f_max - f_min)
            return dist

        def tournament_select(rank, F, X, k=2):
            """Tournament selection based on rank and crowding distance."""
            candidates = rng.integers(0, len(X), size=k)
            best = candidates[0]
            for c in candidates[1:]:
                if rank[c] < rank[best]:
                    best = c
                elif rank[c] == rank[best]:
                    # Compare crowding distance
                    # Simplified: prefer the one with lower objective sum
                    if np.sum(F[c]) < np.sum(F[best]):
                        best = c
            return X[best].copy(), best

        while used < budget:
            if time.time() > deadline - 5.0:
                break

            # Fast non-dominated sort
            rank, fronts = fast_non_dominated_sort(F)

            # Create offspring population
            offspring_X = []
            offspring_F = []

            while len(offspring_X) < pop_size:
                # Select parents
                parent1, idx1 = tournament_select(rank, F, X)
                parent2, idx2 = tournament_select(rank, F, X)

                # Simulated binary crossover
                if rng.random() < 0.9:
                    child = np.empty(n_dim)
                    for i in range(n_dim):
                        if rng.random() < 0.5:
                            child[i] = parent1[i]
                        else:
                            child[i] = parent2[i]
                else:
                    child = parent1.copy()

                # Polynomial mutation
                if rng.random() < 0.9:
                    eta = 20.0
                    for i in range(n_dim):
                        if rng.random() < 0.5:
                            delta = 2.0 * rng.random()
                            if delta <= 0.5:
                                mut_factor = (delta * (eta + 1.0)) ** (1.0 / (eta + 1.0))
                            else:
                                mut_factor = (2.0 - delta * 2.0) ** (1.0 / (eta + 1.0)) * (2.0 - 2.0 * delta * 2.0) ** 0.5
                            if rng.random() < 0.5:
                                child[i] -= mut_factor
                            else:
                                child[i] += mut_factor

                child = np.clip(child, lo, hi)
                fc = problem(child)
                used += 1
                all_points.append((fc[0], fc[1]))
                offspring_X.append(child)
                offspring_F.append(fc)

                if used >= budget or time.time() > deadline - 5.0:
                    break

            if not offspring_X:
                break

            offspring_F = np.array(offspring_F)
            offspring_X = np.array(offspring_X)

            # Combine parent and offspring
            combined_X = np.vstack([X, offspring_X])
            combined_F = np.vstack([F, offspring_F])

            # Non-dominated sort on combined
            rank_c, fronts_c = fast_non_dominated_sort(combined_F)

            # Select next generation
            new_X = []
            new_F = []
            for front in fronts_c:
                if len(new_X) + len(front) <= pop_size:
                    for idx in front:
                        new_X.append(combined_X[idx])
                        new_F.append(combined_F[idx])
                else:
                    # Need to select from this front using crowding distance
                    remaining = pop_size - len(new_X)
                    cd = crowding_distance(combined_F, front)
                    order = np.argsort(-cd)
                    for i in range(remaining):
                        idx = front[order[i]]
                        new_X.append(combined_X[idx])
                        new_F.append(combined_F[idx])
                    break

            X = np.array(new_X)
            F = np.array(new_F)

        return X
# EVOLVE-BLOCK-END
