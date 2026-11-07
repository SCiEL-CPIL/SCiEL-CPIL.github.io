"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-objective optimizer for DTLZ4 using a simple evolutionary strategy with
    non-dominated sorting and crowding distance."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280.0  # Safety margin before 300s limit

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        span = hi - lo

        # For DTLZ4, the Pareto front is determined by the first (n_obj-1) variables.
        # The remaining variables are "position" variables that affect convergence.
        # We use a population-based approach with elitism.

        # Population size
        pop_size = min(40, max(20, self.budget // 5))
        n_gen = max(1, (self.budget - pop_size) // pop_size)

        # Initialize population
        rng = np.random.default_rng(42)
        X = lo + (hi - lo) * rng.random((pop_size, self.dim))
        F = np.array([problem(x) for x in X])
        used = pop_size

        def non_dominated_sort(F):
            """Return fronts (list of arrays of indices) and rank array."""
            n = len(F)
            S = [[] for _ in range(n)]
            n_dom = np.zeros(n, dtype=int)
            rank = np.zeros(n, dtype=int)
            fronts = []
            current_front = []

            for i in range(n):
                for j in range(n):
                    if i == j:
                        continue
                    if np.all(F[i] <= F[j]) and np.any(F[i] < F[j]):
                        S[i].append(j)
                    elif np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                        n_dom[i] += 1
                if n_dom[i] == 0:
                    rank[i] = 0
                    current_front.append(i)

            fronts.append(np.array(current_front))
            while current_front:
                next_front = []
                for i in current_front:
                    for j in S[i]:
                        n_dom[j] -= 1
                        if n_dom[j] == 0:
                            rank[j] = rank[i] + 1
                            next_front.append(j)
                if next_front:
                    fronts.append(np.array(next_front))
                current_front = next_front
            return fronts, rank

        def crowding_distance(F, indices):
            """Compute crowding distance for a set of indices."""
            n = len(indices)
            if n <= 2:
                return np.full(n, np.inf)
            cd = np.zeros(n)
            F_sub = F[indices]
            for m in range(F_sub.shape[1]):
                sorted_idx = np.argsort(F_sub[:, m])
                cd[sorted_idx[0]] = np.inf
                cd[sorted_idx[-1]] = np.inf
                F_min = F_sub[sorted_idx[0], m]
                F_max = F_sub[sorted_idx[-1], m]
                if F_max - F_min > 0:
                    for i in range(1, n - 1):
                        cd[sorted_idx[i]] += (F_sub[sorted_idx[i + 1], m] - F_sub[sorted_idx[i - 1], m]) / (F_max - F_min)
            return cd

        # Main loop
        while used < self.budget and time.time() < deadline:
            # Create offspring
            offspring_X = np.zeros((pop_size, self.dim))
            offspring_F = np.zeros((pop_size, self.n_obj))

            for i in range(pop_size):
                if time.time() > deadline:
                    break
                # Select parents using tournament selection
                parent1_idx, parent2_idx = self._tournament_select(F, rng)
                # Crossover (SBX-like)
                alpha = 20
                child1, child2 = self._sbx_crossover(X[parent1_idx], X[parent2_idx], alpha, lo, hi, rng)
                # Mutation
                child1 = self._polynomial_mutation(child1, lo, hi, self.dim, alpha, rng)
                child2 = self._polynomial_mutation(child2, lo, hi, self.dim, alpha, rng)
                offspring_X[i] = child1
                offspring_F[i] = problem(child1)
                used += 1

                if used < self.budget and i + 1 < pop_size:
                    offspring_X[i + 1] = child2
                    offspring_F[i + 1] = problem(child2)
                    used += 1
                    i += 1

            # Combine and select
            combined_X = np.vstack([X, offspring_X])
            combined_F = np.vstack([F, offspring_F])

            fronts, rank = non_dominated_sort(combined_F)

            # Select next generation
            next_X = np.zeros((pop_size, self.dim))
            next_F = np.zeros((pop_size, self.n_obj))
            selected = 0
            for front in fronts:
                if selected + len(front) <= pop_size:
                    next_X[selected:selected + len(front)] = combined_X[front]
                    next_F[selected:selected + len(front)] = combined_F[front]
                    selected += len(front)
                else:
                    # Need to select from this front using crowding distance
                    remaining = pop_size - selected
                    cd = crowding_distance(combined_F, front)
                    sorted_front = front[np.argsort(-cd)]
                    next_X[selected:selected + remaining] = combined_X[sorted_front[:remaining]]
                    next_F[selected:selected + remaining] = combined_F[sorted_front[:remaining]]
                    selected += remaining
                    break

            X, F = next_X, next_F

        return X

    def _tournament_select(self, F, rng, k=2):
        """Tournament selection: return indices of two parents."""
        n = len(F)
        p1 = rng.integers(n, size=k).tolist()
        p2 = rng.integers(n, size=k).tolist()

        # Find non-dominated among the tournament members
        def is_dominated(i, candidates):
            for j in candidates:
                if np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                    return True
            return False

        best1 = p1[0]
        for i in p1[1:]:
            if not is_dominated(i, p1[:p1.index(i)]) and is_dominated(best1, p1):
                best1 = i
        # Simplified: just pick the one with lower sum
        best1 = min(p1, key=lambda i: np.sum(F[i]))
        best2 = min(p2, key=lambda i: np.sum(F[i]))
        return best1, best2

    def _sbx_crossover(self, p1, p2, eta, lo, hi, rng):
        """Simulated Binary Crossover."""
        c1, c2 = p1.copy(), p2.copy()
        for i in range(len(p1)):
            if rng.random() < 0.5:
                if abs(p1[i] - p2[i]) > 1e-14:
                    if rng.random() <= 1.0 / (eta + 1):
                        beta = (2 * rng.random()) ** (1.0 / (eta + 1))
                    else:
                        beta = (1.0 / (2.0 - 2 * rng.random())) ** (1.0 / (eta + 1))
                    c1[i] = 0.5 * ((1 + beta) * p1[i] + (1 - beta) * p2[i])
                    c2[i] = 0.5 * ((1 - beta) * p1[i] + (1 + beta) * p2[i])
        c1 = np.clip(c1, lo, hi)
        c2 = np.clip(c2, lo, hi)
        return c1, c2

    def _polynomial_mutation(self, x, lo, hi, dim, eta, rng):
        """Polynomial mutation."""
        y = x.copy()
        for i in range(dim):
            if rng.random() < 0.5:
                delta = (y[i] - lo[i]) / (hi[i] - lo[i]) if hi[i] > lo[i] else 0
                if rng.random() <= 2.0 / (eta + 1):
                    beta = (1 + 2 * (1 - delta)) ** (-1.0 / (eta + 1))
                else:
                    u = rng.random()
                    beta = (1 + 2 * delta) ** (-1.0 / (eta + 1)) if u > 0.5 else (1.0 / (2.0 - 2 * u)) ** (1.0 / (eta + 1))
                y[i] = y[i] + (hi[i] - lo[i]) * (beta - 1)
        return np.clip(y, lo, hi)
# EVOLVE-BLOCK-END
