"""Multi-objective optimizer for DTLZ4 using a focused NSGA-II variant."""
import numpy as np
import time


class Optimizer:
    """NSGA-II style multi-objective optimizer with deadline guard."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = 285.0  # safety margin before 300s limit

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        span = hi - lo

        # Population size tuned for budget=200, dim=12, n_obj=3
        pop_size = min(60, max(20, self.budget // 3))

        # Initialize population with random sampling
        X = lo + span * np.random.random((pop_size, self.dim))
        F = np.array([problem(x) for x in X])
        used = pop_size

        # If budget is very small, just do random sampling
        if self.budget <= pop_size:
            return X

        # Main evolutionary loop
        while used < self.budget:
            # Check deadline
            if time.time() - start_time > deadline:
                break

            # Fast non-dominated sorting
            fronts = self._fast_non_dominated_sort(F)

            # Compute crowding distances for each front
            crowding = self._crowding_distance(F, fronts)

            # Create mating pool using tournament selection
            parents = []
            for _ in range(2):
                candidates = np.random.choice(len(F), size=min(3, len(F)), replace=False)
                best_candidate = candidates[0]
                for c in candidates[1:]:
                    if fronts[c] < fronts[best_candidate]:
                        best_candidate = c
                    elif fronts[c] == fronts[best_candidate]:
                        if crowding[c] > crowding[best_candidate]:
                            best_candidate = c
                parents.append(X[best_candidate])

            # Crossover (SBX) and mutation
            child = self._sbx_crossover(parents[0], parents[1], lo, hi)
            child = self._polynomial_mutation(child, lo, hi, span)

            fc = problem(child)
            used += 1

            # Combine parent and child populations
            X_new = np.vstack([X, child.reshape(1, -1)])
            F_new = np.vstack([F, fc.reshape(1, -1)])

            # Select next generation using non-dominated sorting + crowding
            X, F = self._select_next_generation(X_new, F_new, pop_size)

        return X

    def _fast_non_dominated_sort(self, F):
        n = len(F)
        fronts = np.zeros(n, dtype=int)
        dominated_count = np.zeros(n, dtype=int)
        dominates = [[] for _ in range(n)]

        for i in range(n):
            for j in range(i + 1, n):
                if np.all(F[i] <= F[j]) and np.any(F[i] < F[j]):
                    dominates[i].append(j)
                    dominated_count[j] += 1
                elif np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                    dominates[j].append(i)
                    dominated_count[i] += 1

        # First front
        current_front = [i for i in range(n) if dominated_count[i] == 0]
        front_idx = 0
        while current_front:
            next_front = []
            for i in current_front:
                fronts[i] = front_idx
                for j in dominates[i]:
                    dominated_count[j] -= 1
                    if dominated_count[j] == 0:
                        next_front.append(j)
            front_idx += 1
            current_front = next_front

        return fronts

    def _crowding_distance(self, F, fronts):
        n = len(F)
        crowding = np.zeros(n)

        # Group indices by front
        front_indices = {}
        for i in range(n):
            front_idx = fronts[i]
            if front_idx not in front_indices:
                front_indices[front_idx] = []
            front_indices[front_idx].append(i)

        for front_idx, indices in front_indices.items():
            if len(indices) <= 2:
                for i in indices:
                    crowding[i] = np.inf
                continue

            for obj in range(self.n_obj):
                # Sort indices by objective value
                sorted_indices = sorted(indices, key=lambda i: F[i, obj])
                crowding[sorted_indices[0]] = np.inf
                crowding[sorted_indices[-1]] = np.inf

                f_min = F[sorted_indices[0], obj]
                f_max = F[sorted_indices[-1], obj]
                f_range = f_max - f_min
                if f_range == 0:
                    continue

                for k in range(1, len(sorted_indices) - 1):
                    prev_idx = sorted_indices[k - 1]
                    next_idx = sorted_indices[k + 1]
                    crowding[sorted_indices[k]] += (F[next_idx, obj] - F[prev_idx, obj]) / f_range

        return crowding

    def _sbx_crossover(self, p1, p2, lo, hi):
        eta = 20.0
        child = np.zeros_like(p1)
        for i in range(len(p1)):
            u = np.random.rand()
            if u <= 0.5:
                beta = (2 * u) ** (1 / (eta + 1))
            else:
                beta = (1 / (2 * (1 - u))) ** (1 / (eta + 1))
            child[i] = 0.5 * ((1 + beta) * p1[i] + (1 - beta) * p2[i])
        return np.clip(child, lo, hi)

    def _polynomial_mutation(self, x, lo, hi, span):
        eta = 20.0
        mutation_rate = 1.0 / self.dim
        for i in range(len(x)):
            if np.random.rand() < mutation_rate:
                u = np.random.rand()
                if u < 0.5:
                    delta = (2 * u) ** (1 / (eta + 1)) - 1
                else:
                    delta = 1 - (2 * (1 - u)) ** (1 / (eta + 1))
                x[i] += delta * span[i]
        return np.clip(x, lo, hi)

    def _select_next_generation(self, X, F, pop_size):
        n = len(F)
        if n <= pop_size:
            return X, F

        fronts = self._fast_non_dominated_sort(F)
        crowding = self._crowding_distance(F, fronts)

        # Create a list of (front, -crowding, index) for sorting
        # We want lowest front first, then highest crowding
        sorted_indices = sorted(range(n), key=lambda i: (fronts[i], -crowding[i]))

        selected = sorted_indices[:pop_size]
        return X[selected], F[selected]
