"""Multi-objective optimizer on WFG4 using a structured Sobol + NSGA-II hybrid approach."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import qmc


class Optimizer:
    """Structured Sobol sampling + NSGA-II refinement for WFG4."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280.0  # Safety margin for 330s limit

        lo = np.array([b[0] for b in bounds], dtype=float)
        hi = np.array([b[1] for b in bounds], dtype=float)
        span = hi - lo
        dim = self.dim
        n_obj = self.n_obj
        budget = self.budget

        # Use Sobol sequence for deterministic, low-discrepancy initial sampling
        # This provides better coverage than LHS or random sampling
        # Generate 64 points from Sobol sequence (power of 2 for efficiency)
        n_sobol = 64
        sobol = qmc.Sobol(d=dim, scramble=True, seed=42)
        sobol_points_unit = sobol.random(n=n_sobol)
        # Scale to bounds
        sobol_points = lo + span * sobol_points_unit

        # Population size for NSGA-II phase
        pop_size = 24
        if pop_size > budget // 3:
            pop_size = max(10, budget // 3)

        # Phase 1: Evaluate Sobol points to build initial population
        X = []
        F = []
        evaluations = 0

        for x in sobol_points:
            if evaluations >= budget:
                break
            if time.time() > deadline - 5.0:
                break
            f = problem(x)
            evaluations += 1
            X.append(x)
            F.append(f)

        # Ensure we have at least pop_size points
        while len(X) < pop_size and evaluations < budget:
            if time.time() > deadline - 5.0:
                break
            x = lo + span * np.random.rand(dim)
            f = problem(x)
            evaluations += 1
            X.append(x)
            F.append(f)

        # Truncate to pop_size
        X = X[:pop_size]
        F = np.array(F[:pop_size])

        # Helper: Fast non-dominated sort
        def fast_non_dominated_sort(F_arr):
            n = len(F_arr)
            S = [[] for _ in range(n)]
            nd = np.zeros(n, dtype=int)
            fronts = []

            for i in range(n):
                for j in range(n):
                    if i == j:
                        continue
                    if np.all(F_arr[i] <= F_arr[j]) and np.any(F_arr[i] < F_arr[j]):
                        S[i].append(j)
                        nd[j] += 1
                    elif np.all(F_arr[j] <= F_arr[i]) and np.any(F_arr[j] < F_arr[i]):
                        S[j].append(i)
                        nd[i] += 1

            current_front = [i for i in range(n) if nd[i] == 0]
            while current_front:
                fronts.append(current_front)
                next_front = []
                for p in current_front:
                    for q in S[p]:
                        nd[q] -= 1
                        if nd[q] == 0:
                            next_front.append(q)
                current_front = next_front

            return fronts

        # Helper: Crowding distance
        def crowding_distance(F_arr, indices):
            if len(indices) <= 2:
                return {i: float('inf') for i in indices}

            distances = {i: 0.0 for i in indices}
            for m in range(n_obj):
                values = [F_arr[i][m] for i in indices]
                min_val = min(values)
                max_val = max(values)
                if max_val - min_val < 1e-12:
                    continue
                sorted_idx = sorted(indices, key=lambda i: F_arr[i][m])
                distances[sorted_idx[0]] = float('inf')
                distances[sorted_idx[-1]] = float('inf')
                for k in range(1, len(sorted_idx) - 1):
                    distances[sorted_idx[k]] += (
                        F_arr[sorted_idx[k + 1]][m] - F_arr[sorted_idx[k - 1]][m]
                    ) / (max_val - min_val)
            return distances

        # Helper: SBX crossover
        def sbx_crossover(x1, x2):
            eta = 20.0
            child1 = np.zeros_like(x1)
            child2 = np.zeros_like(x2)
            for d in range(dim):
                u = np.random.rand()
                if u <= 0.5:
                    beta = (2 * u) ** (1.0 / (eta + 1))
                else:
                    beta = (1 / (2 * (1 - u))) ** (1.0 / (eta + 1))
                child1[d] = 0.5 * ((1 + beta) * x1[d] + (1 - beta) * x2[d])
                child2[d] = 0.5 * ((1 - beta) * x1[d] + (1 + beta) * x2[d])
            return child1, child2

        # Helper: Polynomial mutation
        def polynomial_mutation(x, sigma=0.1):
            eta_m = 20.0
            result = x.copy()
            for d in range(dim):
                if np.random.rand() < 0.1:
                    u = np.random.rand()
                    if u < 0.5:
                        delta = (2 * u) ** (1.0 / (eta_m + 1)) - 1
                    else:
                        delta = 1 - (2 * (1 - u)) ** (1.0 / (eta_m + 1))
                    result[d] += sigma * span[d] * delta
            return np.clip(result, lo, hi)

        # Helper: Tournament selection
        def tournament_select(pop_F, rank, cd, k=2):
            n = len(pop_F)
            candidates = np.random.choice(n, k, replace=False)
            best = candidates[0]
            for c in candidates[1:]:
                if rank[c] < rank[best]:
                    best = c
                elif rank[c] == rank[best] and cd[c] > cd[best]:
                    best = c
            return int(best)

        # Phase 2: NSGA-II refinement
        while evaluations < budget:
            if time.time() > deadline - 5.0:
                break

            # Compute ranks and crowding distances
            fronts = fast_non_dominated_sort(F)
            rank = np.zeros(len(F), dtype=int)
            for r, front in enumerate(fronts):
                for idx in front:
                    rank[idx] = r

            cd = np.zeros(len(F))
            for r in np.unique(rank):
                idx = np.where(rank == r)[0]
                cd_dict = crowding_distance(F, idx)
                for i in idx:
                    cd[i] = cd_dict[i]

            # Create offspring
            offspring_X = []
            offspring_F = []
            n_offspring = min(pop_size, budget - evaluations)

            for _ in range(n_offspring // 2):
                if evaluations >= budget:
                    break
                if time.time() > deadline - 5.0:
                    break

                # Select parents
                p1 = tournament_select(F, rank, cd)
                p2 = tournament_select(F, rank, cd)

                # Crossover
                child1, child2 = sbx_crossover(X[p1], X[p2])

                # Mutation
                child1 = polynomial_mutation(child1)
                child2 = polynomial_mutation(child2)

                # Evaluate child1
                if evaluations < budget:
                    f1 = problem(child1)
                    evaluations += 1
                    offspring_X.append(child1)
                    offspring_F.append(f1)

                # Evaluate child2
                if evaluations < budget:
                    f2 = problem(child2)
                    evaluations += 1
                    offspring_X.append(child2)
                    offspring_F.append(f2)

            if len(offspring_X) == 0:
                break

            # Combine parent and offspring
            all_X = np.array(X + offspring_X)
            all_F = np.array(F + offspring_F)

            # Select next generation
            n_select = min(pop_size, len(all_X))
            fronts = fast_non_dominated_sort(all_F)
            selected = []
            for front in fronts:
                if len(selected) + len(front) <= n_select:
                    selected.extend(front)
                else:
                    cd_dict = crowding_distance(all_F, front)
                    sorted_front = sorted(front, key=lambda i: cd_dict[i], reverse=True)
                    selected.extend(sorted_front[:n_select - len(selected)])
                    break

            # Update population
            X = [all_X[i] for i in selected]
            F = all_F[selected]

        return X
# EVOLVE-BLOCK-END
