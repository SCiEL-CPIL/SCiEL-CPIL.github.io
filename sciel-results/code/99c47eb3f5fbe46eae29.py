"""Adaptive NSGA-II optimizer for ZDT3.
Instead of relying on analytic knowledge of the Pareto manifold (x[1:]=0.5),
this uses a standard multi-objective evolutionary algorithm with:
- Binary tournament selection
- SBX crossover + polynomial mutation
- Non-dominated sorting + crowding distance
This allows the algorithm to discover the optimal structure from the evaluator.
"""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start = time.time()
        deadline = start + 290.0  # safety margin before 300s wall-clock limit

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        n_budget = self.budget
        dim = self.dim
        n_obj = self.n_obj

        # Population size: use a fraction of budget for exploration
        N = min(n_budget, 40)
        if N < 4:
            N = 4

        # Initialize population uniformly in [lo, hi]
        pop = np.random.uniform(lo, hi, size=(N, dim))
        F = np.empty((N, n_obj))
        eval_count = 0

        for i in range(N):
            if time.time() > deadline:
                break
            F[i] = problem(pop[i])
            eval_count += 1

        # If we ran out of budget during init, return what we have
        if eval_count >= n_budget:
            return pop[:eval_count]

        # NSGA-II loop
        while eval_count < n_budget:
            # Non-dominated sort of current population
            fronts = self._non_dominated_sort(F)
            # Crowding distance within each front
            cd = np.zeros(N)
            for front in fronts:
                if len(front) > 2:
                    cd[front] = self._crowding_distance(F, front)

            # Generate N offspring
            offspring = np.empty((N, dim))
            for i in range(N):
                if time.time() > deadline:
                    break
                idx1 = self._tournament_select(fronts, cd, N)
                idx2 = self._tournament_select(fronts, cd, N)
                child1, _ = self._sbx_crossover(pop[idx1], pop[idx2], lo, hi)
                child1 = self._polynomial_mutate(child1, lo, hi)
                offspring[i] = child1

            # Evaluate offspring
            F_off = np.empty((N, n_obj))
            for i in range(N):
                if time.time() > deadline:
                    break
                if eval_count >= n_budget:
                    break
                F_off[i] = problem(offspring[i])
                eval_count += 1

            # Combine parent and offspring
            pop_combined = np.vstack([pop, offspring])
            F_combined = np.vstack([F, F_off])
            N_combined = 2 * N

            # Non-dominated sort of combined
            fronts_comb = self._non_dominated_sort(F_combined)
            cd_comb = np.zeros(N_combined)
            for front in fronts_comb:
                if len(front) > 2:
                    cd_comb[front] = self._crowding_distance(F_combined, front)

            # Select next population of size N
            next_indices = []
            for front in fronts_comb:
                if len(next_indices) + len(front) <= N:
                    next_indices.extend(front)
                else:
                    remaining = N - len(next_indices)
                    front_cd = cd_comb[front]
                    order = np.argsort(-front_cd)  # descending crowding
                    next_indices.extend(front[order[:remaining]])
                    break

            pop = pop_combined[next_indices]
            F = F_combined[next_indices]

        return pop

    def _non_dominated_sort(self, F):
        """Return list of fronts (lists of indices), from best to worst."""
        N = len(F)
        if N == 0:
            return []
        domination_count = np.zeros(N, dtype=int)
        dominated_by = [[] for _ in range(N)]
        fronts = []

        for i in range(N):
            for j in range(i + 1, N):
                if np.all(F[i] <= F[j]) and np.any(F[i] < F[j]):
                    dominated_by[i].append(j)
                    domination_count[j] += 1
                elif np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                    dominated_by[j].append(i)
                    domination_count[i] += 1

        current_front = [i for i in range(N) if domination_count[i] == 0]
        while current_front:
            fronts.append(current_front)
            next_front = []
            for i in current_front:
                for j in dominated_by[i]:
                    domination_count[j] -= 1
                    if domination_count[j] == 0:
                        next_front.append(j)
            current_front = next_front

        # Add any remaining (safety)
        all_in_fronts = set()
        for front in fronts:
            all_in_fronts.update(front)
        if len(all_in_fronts) < N:
            remaining = [i for i in range(N) if i not in all_in_fronts]
            fronts.append(remaining)

        return fronts

    def _crowding_distance(self, F, indices):
        """Compute crowding distance for points at given indices."""
        n = len(indices)
        cd = np.zeros(n)
        n_obj = F.shape[1]
        for k in range(n_obj):
            obj_vals = F[indices, k]
            min_val = np.min(obj_vals)
            max_val = np.max(obj_vals)
            if max_val - min_val < 1e-12:
                cd[:] = np.inf
                continue
            order = np.argsort(obj_vals)
            cd[order[0]] = np.inf
            cd[order[-1]] = np.inf
            for i in range(1, n - 1):
                cd[order[i]] += (obj_vals[order[i + 1]] - obj_vals[order[i - 1]]) / (max_val - min_val)
        return cd

    def _tournament_select(self, fronts, cd, N):
        """Tournament selection: pick 2 random, return better one."""
        i1 = np.random.randint(0, N)
        i2 = np.random.randint(0, N)
        front_rank = np.zeros(N, dtype=int)
        for rank, front in enumerate(fronts):
            for idx in front:
                front_rank[idx] = rank
        if front_rank[i1] < front_rank[i2]:
            return i1
        elif front_rank[i2] < front_rank[i1]:
            return i2
        else:
            if cd[i1] > cd[i2]:
                return i1
            else:
                return i2

    def _sbx_crossover(self, p1, p2, lo, hi):
        """Simulated Binary Crossover."""
        eta = 20.0
        child1 = np.zeros_like(p1)
        child2 = np.zeros_like(p2)
        for i in range(len(p1)):
            if np.random.rand() < 0.5:
                if abs(p1[i] - p2[i]) > 1e-14:
                    if np.random.rand() < 0.5:
                        beta = (2 * np.random.rand()) ** (1.0 / (eta + 1))
                    else:
                        beta = (1.0 / (2 * np.random.rand())) ** (1.0 / (eta + 1))
                    child1[i] = 0.5 * ((1 + beta) * p1[i] + (1 - beta) * p2[i])
                    child2[i] = 0.5 * ((1 - beta) * p1[i] + (1 + beta) * p2[i])
                else:
                    child1[i] = p1[i]
                    child2[i] = p2[i]
            else:
                child1[i] = p2[i]
                child2[i] = p1[i]
        child1 = np.clip(child1, lo, hi)
        child2 = np.clip(child2, lo, hi)
        return child1, child2

    def _polynomial_mutate(self, x, lo, hi):
        """Polynomial mutation."""
        eta = 20.0
        for i in range(len(x)):
            if np.random.rand() < 0.9:
                y = x[i]
                if y <= lo[i] or y >= hi[i]:
                    y = (lo[i] + hi[i]) / 2.0
                delta1 = (y - lo[i]) / (hi[i] - lo[i] + 1e-12)
                delta2 = (hi[i] - y) / (hi[i] - lo[i] + 1e-12)
                r = np.random.rand()
                if r < 0.5:
                    xy = (2 * r + (1 - 2 * r) * (1 - delta1) ** (eta + 1)) ** (1.0 / (eta + 1))
                    deltaq = xy - 1.0
                else:
                    xy = (2 * (1 - r) + (2 * (r - 0.5)) * (1 - delta2) ** (eta + 1)) ** (1.0 / (eta + 1))
                    deltaq = xy - 1.0
                x[i] = y + deltaq * (hi[i] - lo[i])
                x[i] = np.clip(x[i], lo[i], hi[i])
        return x
# EVOLVE-BLOCK-END
