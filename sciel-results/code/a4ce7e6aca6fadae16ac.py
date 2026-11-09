"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """DE-based multi-objective optimizer with non-dominated sorting and crowding distance."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def _non_dominated_sort(self, F):
        """Perform non-dominated sorting on F (n x m array). Returns list of fronts (indices)."""
        n = len(F)
        S = [[] for _ in range(n)]  # S[i] = set of indices dominated by i
        n_dom = np.zeros(n, dtype=int)  # n_dom[i] = number of individuals dominating i
        fronts = []
        front_idx = 0
        current_front = []

        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                # Check if i dominates j
                if np.all(F[i] <= F[j]) and np.any(F[i] < F[j]):
                    S[i].append(j)
                    n_dom[j] += 1
                elif np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                    S[j].append(i)
                    n_dom[i] += 1

        # First front: all with n_dom == 0
        for i in range(n):
            if n_dom[i] == 0:
                current_front.append(i)

        if current_front:
            fronts.append(current_front)
            front_idx += 1

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
            else:
                break

        return fronts

    def _crowding_distance(self, F, front_indices):
        """Compute crowding distance for individuals in a front. Returns array of distances."""
        n = len(front_indices)
        if n <= 2:
            return np.full(n, np.inf)

        m = self.n_obj
        dist = np.zeros(n)
        F_front = F[front_indices]

        for k in range(m):
            sorted_idx = np.argsort(F_front[:, k])
            dist[sorted_idx[0]] = np.inf
            dist[sorted_idx[-1]] = np.inf
            range_val = F_front[sorted_idx[-1], k] - F_front[sorted_idx[0], k]
            if range_val == 0:
                continue
            for i in range(1, n - 1):
                dist[sorted_idx[i]] += (F_front[sorted_idx[i + 1], k] - F_front[sorted_idx[i - 1], k]) / range_val

        return dist

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = 310.0  # Safety margin before 330s kill

        rng = np.random.default_rng(0)
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        span = hi - lo

        # Initial population
        pop_size = min(50, self.budget // 4)
        X = lo + span * rng.random((pop_size, self.dim))
        F = np.array([problem(x) for x in X])
        used = pop_size

        # Keep all evaluated points for final hypervolume
        all_F = [F.copy()]

        while used < self.budget:
            if time.time() - start_time > deadline:
                break

            # Non-dominated sorting
            fronts = self._non_dominated_sort(F)

            # Select parents from first front using crowding distance
            front0 = fronts[0]
            if len(front0) > 1:
                cd = self._crowding_distance(F, front0)
                # Select 3 parents with highest crowding distance
                top_idx = np.argsort(-cd)[:min(3, len(front0))]
                parent_indices = [front0[i] for i in top_idx]
            else:
                parent_indices = front0

            # Generate offspring using DE
            n_offspring = min(pop_size, self.budget - used)
            for _ in range(n_offspring):
                if used >= self.budget:
                    break

                # Select 3 distinct parents
                if len(parent_indices) >= 3:
                    p_idx = rng.choice(len(parent_indices), size=3, replace=False)
                    p1, p2, p3 = parent_indices[p_idx[0]], parent_indices[p_idx[1]], parent_indices[p_idx[2]]
                else:
                    p1 = p2 = p3 = rng.choice(len(parent_indices))

                # DE mutation: x_trial = x_p1 + F * (x_p2 - x_p3)
                F_factor = 0.5 + 0.5 * rng.random()
                x_trial = X[p1] + F_factor * (X[p2] - X[p3])
                x_trial = np.clip(x_trial, lo, hi)

                f_trial = problem(x_trial)
                used += 1

                # Replacement: replace the worst individual in the last front
                # Find the last front and replace the one with lowest crowding distance
                if len(fronts) > 1:
                    last_front = fronts[-1]
                    if len(last_front) > 1:
                        cd_last = self._crowding_distance(F, last_front)
                        worst_local = np.argmin(cd_last)
                        worst_global = last_front[worst_local]
                    else:
                        worst_global = last_front[0]
                else:
                    # Only one front: replace the one with lowest crowding distance
                    cd_all = self._crowding_distance(F, list(range(len(F))))
                    worst_global = np.argmin(cd_all)

                X[worst_global] = x_trial
                F[worst_global] = f_trial

                # Re-sort after replacement
                fronts = self._non_dominated_sort(F)

            all_F.append(F.copy())

        # Combine all evaluated points
        all_points = np.vstack(all_F)
        return all_points
# EVOLVE-BLOCK-END
