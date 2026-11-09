"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """NSGA-II based multi-objective optimizer for DTLZ1."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280  # safety margin before 300s limit

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        dim = self.dim
        n_obj = self.n_obj
        budget = self.budget

        # Population size
        pop_size = min(40, max(20, budget // 5))

        # Random initialization
        rng = np.random.default_rng(42)
        X = lo + (hi - lo) * rng.random((pop_size, dim))
        F = np.array([problem(x) for x in X])
        used = pop_size

        # NSGA-II parameters
        crossover_rate = 0.9
        mutation_rate = 0.1
        sigma = 0.15  # mutation step size relative to (hi-lo)

        def fast_non_dominated_sort(F):
            """Return list of fronts (lists of indices)."""
            n = len(F)
            S = [[] for _ in range(n)]  # domination count
            n_dom = np.zeros(n, dtype=int)  # number of solutions dominated by i
            fronts = []

            for i in range(n):
                for j in range(n):
                    if i == j:
                        continue
                    # i dominates j?
                    if np.all(F[i] <= F[j]) and np.any(F[i] < F[j]):
                        S[i].append(j)
                    elif np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                        n_dom[i] += 1

            # First front: solutions not dominated by any other
            current_front = [i for i in range(n) if n_dom[i] == 0]
            fronts.append(current_front)

            while current_front:
                next_front = []
                for i in current_front:
                    for j in S[i]:
                        n_dom[j] -= 1
                        if n_dom[j] == 0:
                            next_front.append(j)
                if next_front:
                    fronts.append(next_front)
                current_front = next_front

            return fronts

        def crowding_distance(F, front_indices):
            """Assign crowding distance to solutions in a front."""
            n = len(front_indices)
            if n <= 2:
                return {i: float('inf') for i in front_indices}

            dist = {i: 0.0 for i in front_indices}
            for m in range(n_obj):
                vals = [F[i, m] for i in front_indices]
                idx_sorted = sorted(front_indices, key=lambda i: F[i, m])
                # Assign infinity to extremes
                dist[idx_sorted[0]] = float('inf')
                dist[idx_sorted[-1]] = float('inf')
                range_m = F[idx_sorted[-1], m] - F[idx_sorted[0], m]
                if range_m > 1e-15:
                    for k in range(1, n - 1):
                        dist[idx_sorted[k]] += (F[idx_sorted[k + 1], m] - F[idx_sorted[k - 1], m]) / range_m
            return dist

        def crossover(parent1, parent2):
            """Simulated binary crossover (SBX) for real-valued vectors."""
            if rng.random() > crossover_rate:
                return parent1.copy(), parent2.copy()
            child1 = np.zeros(dim)
            child2 = np.zeros(dim)
            for i in range(dim):
                if rng.random() < 0.5:
                    child1[i] = parent1[i]
                    child2[i] = parent2[i]
                else:
                    # SBX
                    eta = 20
                    u = rng.random()
                    if u <= 0.5:
                        beta = (2 * u) ** (1.0 / (eta + 1))
                    else:
                        beta = (1.0 / (2 * (1 - u))) ** (1.0 / (eta + 1))
                    x1, x2 = min(parent1[i], parent2[i]), max(parent1[i], parent2[i])
                    child1[i] = 0.5 * ((1 + beta) * x1 + (1 - beta) * x2)
                    child2[i] = 0.5 * ((1 - beta) * x1 + (1 + beta) * x2)
            return child1, child2

        def mutate(x):
            """Polynomial mutation."""
            eta = 20
            x_mut = x.copy()
            for i in range(dim):
                if rng.random() < mutation_rate:
                    u = rng.random()
                    if u < 0.5:
                        delta_q = (2 * u) ** (1.0 / (eta + 1)) - 1
                    else:
                        delta_q = 1 - (2 * (1 - u)) ** (1.0 / (eta + 1))
                    x_mut[i] = x[i] + delta_q * (hi[i] - lo[i])
            return np.clip(x_mut, lo, hi)

        # Main evolution loop
        while used < budget:
            # Check deadline
            if time.time() > deadline:
                break

            # Generate offspring
            offspring_X = []
            offspring_F = []
            n_offspring = pop_size

            for _ in range(n_offspring // 2):
                if time.time() > deadline:
                    break
                # Tournament selection
                p1_idx = self._tournament_select(F, pop_size)
                p2_idx = self._tournament_select(F, pop_size)
                c1, c2 = crossover(X[p1_idx], X[p2_idx])
                c1 = mutate(c1)
                c2 = mutate(c2)
                offspring_X.append(c1)
                offspring_F.append(problem(c1))
                used += 1
                if used >= budget:
                    break
                if time.time() > deadline:
                    break
                offspring_X.append(c2)
                offspring_F.append(problem(c2))
                used += 1
                if used >= budget:
                    break

            if not offspring_X:
                break

            offspring_X = np.array(offspring_X)
            offspring_F = np.array(offspring_F)

            # Combine parent and offspring
            combined_X = np.vstack([X, offspring_X])
            combined_F = np.vstack([F, offspring_F])
            n_combined = len(combined_F)

            # Non-dominated sorting
            fronts = fast_non_dominated_sort(combined_F)

            # Select next population
            next_X = []
            next_F = []
            for front in fronts:
                if len(next_X) + len(front) <= pop_size:
                    for idx in front:
                        next_X.append(combined_X[idx])
                        next_F.append(combined_F[idx])
                else:
                    # Need to truncate this front
                    remaining = pop_size - len(next_X)
                    cd = crowding_distance(combined_F, front)
                    # Sort by crowding distance descending
                    sorted_front = sorted(front, key=lambda i: cd[i], reverse=True)
                    for idx in sorted_front[:remaining]:
                        next_X.append(combined_X[idx])
                        next_F.append(combined_F[idx])
                    break

            X = np.array(next_X)
            F = np.array(next_F)

        return X

    def _tournament_select(self, F, k=2):
        """Tournament selection: return index of best individual."""
        n = len(F)
        indices = np.random.choice(n, size=k, replace=False)
        # Find non-dominated among the k
        best = indices[0]
        for idx in indices[1:]:
            # idx dominates best?
            if np.all(F[idx] <= F[best]) and np.any(F[idx] < F[best]):
                best = idx
        return best
# EVOLVE-BLOCK-END
