"""NSGA-II-style multi-objective optimizer for WFG9."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """NSGA-II-style dominance-based multi-objective optimizer."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 270  # safety margin before 300s limit

        lo = np.array([b[0] for b in bounds], dtype=np.float64)
        hi = np.array([b[1] for b in bounds], dtype=np.float64)
        span = hi - lo

        n_obj = self.n_obj
        dim = self.dim
        budget = self.budget

        # Population size
        # With budget 200, we want enough generations.
        # NSGA-II creates N offspring per generation.
        # If N=50, we get 4 generations.
        # If N=20, we get 10 generations.
        # Let's try N=40 for a balance.
        pop_size = 40

        # Initialize population
        X = lo + span * np.random.random((pop_size, dim))
        F = np.array([problem(x) for x in X])

        # Track evaluations used
        used = pop_size

        # Main loop
        while used + pop_size <= budget:
            # Check deadline
            if time.time() > deadline:
                break

            # Generate offspring
            offspring_X = np.zeros((pop_size, dim))
            offspring_F = np.zeros((pop_size, n_obj))

            for i in range(pop_size):
                # Binary tournament selection based on non-dominated rank and crowding distance
                p1, p2 = self._tournament_select(X, F)

                # SBX Crossover
                child = self._sbx_crossover(X[p1], X[p2], lo, hi, 20.0)

                # Gaussian Mutation
                # Adapt mutation rate? Keep simple: 1/dim probability, sigma=0.1*span
                mut_mask = np.random.random(dim) < 0.5
                if np.any(mut_mask):
                    child[mut_mask] += 0.1 * span[mut_mask] * np.random.standard_normal(np.sum(mut_mask))
                    child = np.clip(child, lo, hi)

                offspring_X[i] = child
                offspring_F[i] = problem(child)
                used += 1

            # Combine parents and offspring
            combined_X = np.vstack([X, offspring_X])
            combined_F = np.vstack([F, offspring_F])

            # Non-dominated sorting and crowding distance
            ranks, distances = self._non_dominated_sort(combined_F)

            # Select next population
            # Sort by rank, then by crowding distance (descending)
            # We want the best ranks (lowest) and highest crowding distance
            # Create indices sorted by rank, then by -distance
            sort_idx = np.lexsort((-distances, ranks))
            next_X = combined_X[sort_idx[:pop_size]]
            next_F = combined_F[sort_idx[:pop_size]]

            X = next_X
            F = next_F

        # If we have leftover budget, use it for local search or more random perturbations?
        # For simplicity, just return the current population.
        # The evaluator records ALL evaluated points.
        # So we should ensure we return a diverse set, but the evaluator keeps history.
        # Returning X is sufficient.

        return X

    def _tournament_select(self, X, F):
        """Binary tournament selection based on non-dominated rank and crowding distance."""
        pop_size = len(X)
        i1 = np.random.randint(pop_size)
        i2 = np.random.randint(pop_size)

        # Calculate ranks and distances for the whole population (inefficient but simple for small N)
        # Optimization: We can cache these, but for N=40, it's fast enough.
        ranks, distances = self._non_dominated_sort(F)

        # Compare i1 and i2
        # Lower rank is better. If same rank, higher crowding distance is better.
        if ranks[i1] < ranks[i2]:
            return i1, i2
        elif ranks[i2] < ranks[i1]:
            return i2, i1
        else:
            if distances[i1] > distances[i2]:
                return i1, i2
            else:
                return i2, i1

    def _non_dominated_sort(self, F):
        """Perform non-dominated sorting and calculate crowding distance."""
        n = len(F)
        n_obj = F.shape[1]
        ranks = np.zeros(n, dtype=int)
        distances = np.zeros(n, dtype=float)

        # Non-dominated sorting
        S = [set() for _ in range(n)]  # Dominated sets
        n_dominated = np.zeros(n, dtype=int)  # Number of solutions dominated by i
        front = set()
        for i in range(n):
            front.add(i)
        current_front = 0

        while front:
            next_front = set()
            for i in front:
                for j in range(n):
                    if j == i:
                        continue
                    # Check if i dominates j
                    if self._dominates(F[i], F[j]):
                        if j not in S[i]:
                            S[i].add(j)
                            n_dominated[j] += 1
                        if n_dominated[j] == 0:
                            next_front.add(j)
                    # Check if j dominates i
                    elif self._dominates(F[j], F[i]):
                        if i not in S[j]:
                            S[j].add(i)
                            n_dominated[i] += 1

            for i in front:
                ranks[i] = current_front

            # Calculate crowding distance for current front
            self._crowding_distance(F, list(front), distances)

            front = next_front
            current_front += 1

        return ranks, distances

    def _dominates(self, f1, f2):
        """Check if f1 dominates f2 (minimization)."""
        # f1 dominates f2 if f1 <= f2 in all objectives and f1 < f2 in at least one
        return np.all(f1 <= f2) and np.any(f1 < f2)

    def _crowding_distance(self, F, front_indices, distances):
        """Calculate crowding distance for a set of non-dominated solutions."""
        n = len(front_indices)
        if n <= 2:
            for i in front_indices:
                distances[i] = np.inf
            return

        # Sort by each objective
        for m in range(F.shape[1]):
            # Get indices sorted by objective m
            sorted_indices = sorted(front_indices, key=lambda idx: F[idx, m])
            # Set boundaries to inf
            distances[sorted_indices[0]] = np.inf
            distances[sorted_indices[-1]] = np.inf

            # Calculate distance for interior points
            f_min = F[sorted_indices[0], m]
            f_max = F[sorted_indices[-1], m]
            range_val = f_max - f_min
            if range_val < 1e-10:
                continue

            for k in range(1, n - 1):
                idx = sorted_indices[k]
                dist = (F[sorted_indices[k + 1], m] - F[sorted_indices[k - 1], m]) / range_val
                distances[idx] += dist

    def _sbx_crossover(self, x1, x2, lo, hi, eta):
        """Simulated Binary Crossover."""
        child = np.zeros_like(x1)
        for i in range(len(x1)):
            if np.random.random() < 0.5:
                if abs(x1[i] - x2[i]) < 1e-14:
                    child[i] = x1[i]
                else:
                    u = np.random.random()
                    if u <= 0.5:
                        beta = (2 * u) ** (1.0 / (eta + 1))
                    else:
                        beta = (1.0 / (2 * (1 - u))) ** (1.0 / (eta + 1))
                    child[i] = 0.5 * ((x1[i] + x2[i]) - beta * (x1[i] - x2[i]))
                child[i] = np.clip(child[i], lo[i], hi[i])
            else:
                if abs(x1[i] - x2[i]) < 1e-14:
                    child[i] = x1[i]
                else:
                    u = np.random.random()
                    if u <= 0.5:
                        beta = (2 * u) ** (1.0 / (eta + 1))
                    else:
                        beta = (1.0 / (2 * (1 - u))) ** (1.0 / (eta + 1))
                    child[i] = 0.5 * ((x1[i] + x2[i]) + beta * (x1[i] - x2[i]))
                child[i] = np.clip(child[i], lo[i], hi[i])
        return child
# EVOLVE-BLOCK-END
