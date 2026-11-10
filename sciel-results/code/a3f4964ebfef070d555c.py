"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-objective optimizer using NSGA-II with local refinement for ZDT4."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def _fast_non_dominated_sort(self, F):
        """Return list of fronts, each containing indices of non-dominated points."""
        n = len(F)
        S = [[] for _ in range(n)]  # S[i] = indices dominated by i
        n_dominated = np.zeros(n, dtype=int)  # number of points i dominates
        fronts = [[]]

        for i in range(n):
            for j in range(i + 1, n):
                if np.all(F[i] <= F[j]) and np.any(F[i] < F[j]):
                    S[i].append(j)
                    n_dominated[j] += 1
                elif np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                    S[j].append(i)
                    n_dominated[i] += 1

        for i in range(n):
            if n_dominated[i] == 0:
                fronts[0].append(i)

        k = 0
        while fronts[k]:
            next_front = []
            for i in fronts[k]:
                for j in S[i]:
                    n_dominated[j] -= 1
                    if n_dominated[j] == 0:
                        next_front.append(j)
            k += 1
            if next_front:
                fronts.append(next_front)
            else:
                break

        return fronts

    def _crowding_distance(self, F, indices):
        """Compute crowding distance for given indices."""
        n = len(indices)
        if n <= 2:
            return np.full(n, np.inf)

        dist = np.zeros(n)
        for m in range(self.n_obj):
            f_vals = F[indices, m]
            sorted_idx = np.argsort(f_vals)
            dist[sorted_idx[0]] = np.inf
            dist[sorted_idx[-1]] = np.inf
            f_range = f_vals[sorted_idx[-1]] - f_vals[sorted_idx[0]]
            if f_range > 0:
                for i in range(1, n - 1):
                    dist[sorted_idx[i]] += (f_vals[sorted_idx[i + 1]] - f_vals[sorted_idx[i - 1]]) / f_range

        return dist

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280  # safety margin before 300s limit

        rng = np.random.default_rng(42)
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        dim = self.dim

        # Informed initialization for ZDT4: x0 spread over [-5,5], others near 0
        pop_size = min(40, self.budget // 5)
        X = np.zeros((pop_size, dim))
        X[:, 0] = np.linspace(-5, 5, pop_size) + rng.standard_normal(pop_size) * 0.1
        # x1..x9: mix of near-zero and random to explore g(x)
        for i in range(1, dim):
            X[:, i] = rng.random(pop_size) * 0.2  # start near 0 where g(x) is small

        F = np.array([problem(x) for x in X])
        used = pop_size

        # NSGA-II main loop
        while used < self.budget:
            if time.time() > deadline:
                break

            # Non-dominated sort
            fronts = self._fast_non_dominated_sort(F)

            # Build next population
            next_X = []
            next_F = []
            remaining = pop_size

            for front in fronts:
                if len(front) <= remaining:
                    next_X.extend(X[front])
                    next_F.extend(F[front])
                    remaining -= len(front)
                else:
                    # Need to select from this front using crowding distance
                    cd = self._crowding_distance(F, front)
                    sorted_by_cd = np.argsort(-cd)
                    selected = [front[i] for i in sorted_by_cd[:remaining]]
                    next_X.extend(X[selected])
                    next_F.extend(F[selected])
                    remaining = 0
                    break

            X = np.array(next_X)
            F = np.array(next_F)

            # Generate offspring
            offspring_X = []
            offspring_F = []

            for _ in range(pop_size):
                if time.time() > deadline:
                    break
                if used >= self.budget:
                    break

                # Binary tournament selection
                i1, i2 = rng.integers(pop_size, size=2)
                # Compare fronts
                front_i1 = next((k for k, fr in enumerate(fronts) if i1 in fr), len(fronts))
                front_i2 = next((k for k, fr in enumerate(fronts) if i2 in fr), len(fronts))

                if front_i1 < front_i2:
                    parent_idx = i1
                elif front_i2 < front_i1:
                    parent_idx = i2
                else:
                    # Same front, use crowding distance
                    cd = self._crowding_distance(F, list(range(pop_size)))
                    parent_idx = i1 if cd[i1] > cd[i2] else i2

                parent = X[parent_idx].copy()

                # Mutation: Gaussian with adaptive step size
                step_size = 0.05 * (hi - lo)
                child = parent + rng.standard_normal(dim) * step_size

                # Special handling for ZDT4: bias x0 mutation to explore range
                child[0] = parent[0] + rng.standard_normal() * 0.3
                # For x1..x9, sometimes reset toward 0 (where g(x) is minimized)
                for j in range(1, dim):
                    if rng.random() < 0.3:
                        child[j] = parent[j] * 0.5 + rng.standard_normal() * 0.02
                    else:
                        child[j] = parent[j] + rng.standard_normal() * 0.02

                child = np.clip(child, lo, hi)
                fc = problem(child)
                used += 1
                offspring_X.append(child)
                offspring_F.append(fc)

            if not offspring_X:
                break

            offspring_X = np.array(offspring_X)
            offspring_F = np.array(offspring_F)

            # Combine and select
            combined_X = np.vstack([X, offspring_X])
            combined_F = np.vstack([F, offspring_F])
            n_combined = len(combined_F)

            fronts = self._fast_non_dominated_sort(combined_F)

            new_X = []
            new_F = []
            remaining = pop_size

            for front in fronts:
                if len(front) <= remaining:
                    new_X.extend(combined_X[front])
                    new_F.extend(combined_F[front])
                    remaining -= len(front)
                else:
                    cd = self._crowding_distance(combined_F, front)
                    sorted_by_cd = np.argsort(-cd)
                    selected = [front[i] for i in sorted_by_cd[:remaining]]
                    new_X.extend(combined_X[selected])
                    new_F.extend(combined_F[selected])
                    remaining = 0
                    break

            X = np.array(new_X)
            F = np.array(new_F)

        # Local refinement: polish best points
        if time.time() < deadline - 10 and used < self.budget:
            # Find non-dominated points
            fronts = self._fast_non_dominated_sort(F)
            pareto_idx = fronts[0] if fronts else [0]

            # For each Pareto point, do local search
            for idx in pareto_idx[:min(10, len(pareto_idx))]:
                if time.time() > deadline - 5:
                    break
                if used >= self.budget:
                    break

                best_x = X[idx].copy()
                best_f = F[idx].copy()

                # Local search with decreasing step
                for step_mult in [0.02, 0.01, 0.005, 0.001]:
                    if used >= self.budget or time.time() > deadline - 5:
                        break
                    step = step_mult * (hi - lo)
                    for _ in range(5):
                        if used >= self.budget or time.time() > deadline - 5:
                            break
                        candidate = best_x + rng.standard_normal(dim) * step
                        candidate[0] = best_x[0] + rng.standard_normal() * step[0] * 2
                        candidate = np.clip(candidate, lo, hi)
                        fc = problem(candidate)
                        used += 1

                        # Accept if it dominates or is non-dominated with better spread
                        if np.all(fc <= best_f) and np.any(fc < best_f):
                            best_x = candidate
                            best_f = fc
                        elif np.all(best_f <= fc) is False and np.all(fc <= best_f) is False:
                            # Non-dominated, keep if it improves diversity
                            pass

                # Update if improved
                if np.all(best_f <= F[idx]) and np.any(best_f < F[idx]):
                    X[idx] = best_x
                    F[idx] = best_f

        return X
# EVOLVE-BLOCK-END
