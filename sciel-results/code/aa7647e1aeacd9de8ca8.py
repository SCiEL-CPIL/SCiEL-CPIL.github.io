"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """NSGA-II style multi-objective evolutionary optimizer for WFG7."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def _fast_non_dominated_sort(self, F):
        """Return list of fronts (list of indices) and domination counts."""
        n = len(F)
        S = [[] for _ in range(n)]  # S[i] = indices dominated by i
        n_dom = np.zeros(n, dtype=int)  # number of points that dominate i
        fronts = []

        for i in range(n):
            for j in range(i + 1, n):
                if np.all(F[i] <= F[j]) and np.any(F[i] < F[j]):
                    S[i].append(j)
                    n_dom[j] += 1
                elif np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                    S[j].append(i)
                    n_dom[i] += 1

        current_front = [i for i in range(n) if n_dom[i] == 0]
        rank = np.zeros(n, dtype=int)

        while current_front:
            fronts.append(current_front)
            next_front = []
            for i in current_front:
                rank[i] = len(fronts) - 1
                for j in S[i]:
                    n_dom[j] -= 1
                    if n_dom[j] == 0:
                        next_front.append(j)
            current_front = next_front

        # Assign remaining (shouldn't happen in practice)
        for i in range(n):
            if rank[i] == 0 and i not in [idx for front in fronts for idx in front]:
                rank[i] = len(fronts)
                fronts.append([i])

        return fronts, rank

    def _crowding_distance(self, F, front_indices):
        """Compute crowding distance for indices in a front."""
        n = len(front_indices)
        if n <= 2:
            return np.full(n, np.inf)

        dist = np.zeros(n)
        m = self.n_obj

        for k in range(m):
            obj = F[front_indices, k]
            min_val = obj.min()
            max_val = obj.max()
            if max_val - min_val < 1e-15:
                dist[:] = np.inf
                continue
            order = np.argsort(obj)
            dist[order[0]] = np.inf
            dist[order[-1]] = np.inf
            for i in range(1, n - 1):
                dist[order[i]] += (obj[order[i + 1]] - obj[order[i - 1]]) / (max_val - min_val)

        return dist

    def _binary_tournament(self, rank, crowd_dist, idx1, idx2):
        """Pick better of two candidates based on rank then crowding."""
        if rank[idx1] < rank[idx2]:
            return idx1
        elif rank[idx2] < rank[idx1]:
            return idx2
        else:
            return idx1 if crowd_dist[idx1] > crowd_dist[idx2] else idx2

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280  # safety margin

        rng = np.random.RandomState()  # uses np.random state set by evaluator
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        span = hi - lo

        pop_size = min(100, max(20, self.budget // 2))
        # Initialize population
        X = lo + span * rng.rand(pop_size, self.dim)
        F = np.array([problem(x) for x in X])
        used = pop_size

        # Initial non-dominated sort and crowding
        fronts, rank = self._fast_non_dominated_sort(F)
        crowd_dist = np.zeros(pop_size)
        for front in fronts:
            cd = self._crowding_distance(F, front)
            for i, idx in enumerate(front):
                crowd_dist[idx] = cd[i]

        all_X = [X.copy()]
        all_F = [F.copy()]

        while used < self.budget:
            if time.time() > deadline:
                break

            # Create offspring population
            offspring = []
            for _ in range(pop_size):
                # Binary tournament selection
                i1 = rng.randint(pop_size)
                i2 = rng.randint(pop_size)
                p1 = self._binary_tournament(rank, crowd_dist, i1, i2)

                i1 = rng.randint(pop_size)
                i2 = rng.randint(pop_size)
                p2 = self._binary_tournament(rank, crowd_dist, i1, i2)

                # Simulated binary crossover (SBX)
                child = np.zeros(self.dim)
                for d in range(self.dim):
                    if rng.rand() < 0.5:  # crossover probability
                        x1, x2 = X[p1, d], X[p2, d]
                        if abs(x1 - x2) < 1e-15:
                            child[d] = x1
                        else:
                            u = rng.rand()
                            if u <= 0.5:
                                beta = (2 * u) ** (1 / 11)  # eta=10
                            else:
                                beta = (1 / (2 * (1 - u))) ** (1 / 11)
                            c1 = 0.5 * ((1 + beta) * x1 + (1 - beta) * x2)
                            child[d] = c1
                    else:
                        child[d] = X[p1, d] if rng.rand() < 0.5 else X[p2, d]

                # Polynomial mutation
                for d in range(self.dim):
                    if rng.rand() < 0.5:  # mutation probability
                        u = rng.rand()
                        if u < 0.5:
                            delta = (2 * u) ** (1 / 11) - 1
                        else:
                            delta = 1 - (2 * (1 - u)) ** (1 / 11)
                        child[d] += delta * (hi[d] - lo[d])

                child = np.clip(child, lo, hi)
                offspring.append(child)

            offspring = np.array(offspring)
            F_off = np.array([problem(x) for x in offspring])
            used += pop_size

            # Combine parent and offspring
            X_all = np.vstack([X, offspring])
            F_all = np.vstack([F, F_off])
            n_all = len(X_all)

            # Non-dominated sort on combined
            fronts_all, rank_all = self._fast_non_dominated_sort(F_all)

            # Build new population
            new_X = []
            new_F = []
            new_rank = np.zeros(n_all, dtype=int)
            new_crowd = np.zeros(n_all)

            for front in fronts_all:
                if len(new_X) + len(front) <= pop_size:
                    for idx in front:
                        new_X.append(X_all[idx])
                        new_F.append(F_all[idx])
                        new_rank.append(rank_all[idx])
                else:
                    # Need to use crowding distance to select remaining
                    cd = self._crowding_distance(F_all, front)
                    order = np.argsort(-cd)
                    remaining = pop_size - len(new_X)
                    for i in range(min(remaining, len(order))):
                        idx = front[order[i]]
                        new_X.append(X_all[idx])
                        new_F.append(F_all[idx])
                        new_rank.append(rank_all[idx])
                        new_crowd.append(cd[order[i]])
                    break
                for idx in front:
                    new_crowd.append(0)  # placeholder, will recompute

            X = np.array(new_X)
            F = np.array(new_F)
            rank = np.array(new_rank)

            # Recompute crowding distance for the new population
            fronts_new, _ = self._fast_non_dominated_sort(F)
            crowd_dist = np.zeros(len(X))
            for front in fronts_new:
                cd = self._crowding_distance(F, front)
                for i, idx in enumerate(front):
                    crowd_dist[idx] = cd[i]

            all_X.append(X.copy())
            all_F.append(F.copy())

        # Collect all non-dominated points from all generations
        all_points = []
        for Xg, Fg in zip(all_X, all_F):
            all_points.append((Xg, Fg))

        # Find overall non-dominated set
        all_F_flat = np.vstack([Fg for _, Fg in all_points])
        all_X_flat = np.vstack([Xg for Xg, _ in all_points])
        n_total = len(all_F_flat)

        # Filter points better than reference point [3.0, 5.0, 7.0]
        ref = np.array([3.0, 5.0, 7.0])
        mask = np.all(all_F_flat < ref, axis=1)
        all_F_flat = all_F_flat[mask]
        all_X_flat = all_X_flat[mask]

        # Return the best set (we return X, but the evaluator records all F)
        # Actually, the evaluator records every returned objective vector.
        # The return value is ignored, but we need to make sure all evaluations
        # are recorded. Since problem() is called and its result used in F,
        # all are recorded. We just return something.
        return all_X_flat
# EVOLVE-BLOCK-END
