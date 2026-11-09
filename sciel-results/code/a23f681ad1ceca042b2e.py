"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import differential_evolution
from scipy.optimize import minimize


class Optimizer:
    """Multi-objective optimizer using NSGA-II style evolutionary search on ZDT4."""

    def __init__(self, budget, dim, n_obj):
        self.budget, self.dim, self.n_obj = budget, dim, n_obj

    def _dominates(self, a, b):
        """Check if a dominates b (minimization)."""
        return np.all(a <= b) and np.any(a < b)

    def _non_dominated_sort(self, F):
        """Perform non-dominated sorting on objective matrix F.
        Returns list of fronts, each a list of indices."""
        n = len(F)
        S = [[] for _ in range(n)]  # domination relations
        n_dominates = np.zeros(n, dtype=int)  # number of solutions dominating i
        fronts = []
        current_front = []

        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                if self._dominates(F[i], F[j]):
                    S[i].append(j)
                elif self._dominates(F[j], F[i]):
                    n_dominates[i] += 1

            if n_dominates[i] == 0:
                current_front.append(i)

        while current_front:
            fronts.append(current_front)
            next_front = []
            for i in current_front:
                for j in S[i]:
                    n_dominates[j] -= 1
                    if n_dominates[j] == 0:
                        next_front.append(j)
            current_front = next_front

        # If some points weren't assigned (shouldn't happen), add them to last front
        assigned = set()
        for front in fronts:
            assigned.update(front)
        unassigned = [i for i in range(n) if i not in assigned]
        if unassigned:
            fronts.append(unassigned)

        return fronts

    def _crowding_distance(self, F, front_indices):
        """Compute crowding distances for a front."""
        if len(front_indices) <= 2:
            return np.full(len(front_indices), np.inf)

        distances = np.zeros(len(front_indices))
        F_front = F[front_indices]
        m = self.n_obj

        for k in range(m):
            obj_vals = F_front[:, k]
            sorted_idx = np.argsort(obj_vals)
            min_val = obj_vals[sorted_idx[0]]
            max_val = obj_vals[sorted_idx[-1]]
            if max_val - min_val < 1e-10:
                continue
            distances[sorted_idx[0]] = np.inf
            distances[sorted_idx[-1]] = np.inf
            for i in range(1, len(sorted_idx) - 1):
                distances[sorted_idx[i]] += (obj_vals[sorted_idx[i + 1]] - obj_vals[sorted_idx[i - 1]]) / (max_val - min_val)

        return distances

    def __call__(self, problem, bounds):
        rng = np.random.default_rng(42)
        lo = np.array([b[0] for b in bounds], dtype=float)
        hi = np.array([b[1] for b in bounds], dtype=float)
        span = hi - lo

        pop_size = 30
        X = lo + span * rng.random((pop_size, self.dim))
        F = np.array([problem(x) for x in X])
        used = pop_size

        # NSGA-II main loop
        while used < self.budget:
            # Combine parent and child populations
            # For simplicity, we do a small population update each iteration
            # Create offspring
            n_offspring = pop_size

            # Select parents using tournament selection based on rank and crowding
            ranks = np.zeros(pop_size, dtype=int)
            fronts = self._non_dominated_sort(F)
            for rank, front in enumerate(fronts):
                for idx in front:
                    ranks[idx] = rank

            # Compute crowding distances per front
            crowding = np.zeros(pop_size)
            for front in fronts:
                if len(front) > 0:
                    cd = self._crowding_distance(F, front)
                    for i, idx in enumerate(front):
                        crowding[idx] = cd[i]

            # Tournament selection
            def select_parent():
                i = rng.integers(pop_size)
                j = rng.integers(pop_size)
                if ranks[i] < ranks[j]:
                    return i
                elif ranks[j] < ranks[i]:
                    return j
                else:
                    if crowding[i] > crowding[j]:
                        return i
                    else:
                        return j

            offspring_X = []
            offspring_F = []

            for _ in range(n_offspring):
                p1 = select_parent()
                p2 = select_parent()

                # SBX crossover (simulated binary crossover)
                if rng.random() < 0.9:
                    eta = 20.0
                    child = np.zeros(self.dim)
                    for d in range(self.dim):
                        u = rng.random()
                        if u <= 0.5:
                            beta = (2 * u) ** (1.0 / (eta + 1))
                        else:
                            beta = (2 * (1 - u)) ** (1.0 / (eta + 1))
                        child[d] = 0.5 * ((1 + beta) * X[p1, d] + (1 - beta) * X[p2, d])
                else:
                    child = X[p1].copy()

                # Mutation: polynomial mutation
                if rng.random() < 0.5:
                    eta_m = 20.0
                    for d in range(self.dim):
                        if rng.random() < 0.5:
                            u = rng.random()
                            if u < 0.5:
                                delta_q = (2 * u) ** (1.0 / (eta_m + 1)) - 1
                            else:
                                delta_q = 1 - (2 * (1 - u)) ** (1.0 / (eta_m + 1))
                            child[d] += delta_q * (hi[d] - lo[d])

                child = np.clip(child, lo, hi)
                fc = problem(child)
                used += 1
                offspring_X.append(child)
                offspring_F.append(fc)

                if used >= self.budget:
                    break

            if used >= self.budget:
                break

            offspring_X = np.array(offspring_X)
            offspring_F = np.array(offspring_F)

            # Combine and select
            combined_X = np.vstack([X, offspring_X])
            combined_F = np.vstack([F, offspring_F])
            n_combined = len(combined_F)

            # Non-dominated sort on combined
            combined_fronts = self._non_dominated_sort(combined_F)

            # Select next population
            new_X = []
            new_F = []
            for front in combined_fronts:
                if len(new_X) + len(front) <= pop_size:
                    new_X.extend(combined_X[front])
                    new_F.extend(combined_F[front])
                else:
                    # Need to sort this front by crowding distance
                    cd = self._crowding_distance(combined_F, front)
                    sorted_front = sorted(front, key=lambda i: cd[i - (front[0] if front[0] < 0 else 0)], reverse=True)
                    # Properly compute crowding with global indices
                    cd_global = self._crowding_distance(combined_F, front)
                    # Sort by crowding distance descending
                    order = np.argsort(-cd_global)
                    remaining = pop_size - len(new_X)
                    for idx in order[:remaining]:
                        global_idx = front[idx]
                        new_X.append(combined_X[global_idx])
                        new_F.append(combined_F[global_idx])
                    break

            X = np.array(new_X)
            F = np.array(new_F)

        return X
# EVOLVE-BLOCK-END
