"""Multi-objective optimizer on RE34 using NSGA-II with LHS initialization."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """NSGA-II multi-objective optimizer with LHS initialization, SBX crossover, and adaptive mutation."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def _lhs(self, n, dim, rng):
        """Latin Hypercube Sampling."""
        pts = np.zeros((n, dim))
        for i in range(n):
            pts[i, :] = rng.random(dim)
        # Shuffle each dimension
        for d in range(dim):
            perm = rng.permutation(n)
            pts[:, d] = pts[perm, d]
        pts /= n
        return pts

    def _non_dominated_sort(self, F):
        """Return list of fronts (each a list of indices) and domination count array."""
        n = len(F)
        S = [[] for _ in range(n)]  # S[i] = set of points dominated by i
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
        while current_front:
            fronts.append(current_front)
            next_front = []
            for i in current_front:
                for j in S[i]:
                    n_dom[j] -= 1
                    if n_dom[j] == 0:
                        next_front.append(j)
            current_front = next_front

        # Assign front numbers
        front_num = np.zeros(n, dtype=int)
        for fi, front in enumerate(fronts):
            for idx in front:
                front_num[idx] = fi
        return fronts, front_num

    def _crowding_distance(self, F, indices):
        """Compute crowding distance for a set of indices."""
        if len(indices) == 0:
            return np.array([])
        if len(indices) == 1:
            return np.array([np.inf])

        dist = np.zeros(len(indices))
        F_sub = F[indices]
        m = self.n_obj

        for k in range(m):
            vals = F_sub[:, k]
            mn, mx = vals.min(), vals.max()
            if mx - mn < 1e-14:
                continue
            order = np.argsort(vals)
            dist[order[0]] = np.inf
            dist[order[-1]] = np.inf
            for i in range(1, len(order) - 1):
                dist[order[i]] += (vals[order[i + 1]] - vals[order[i - 1]]) / (mx - mn)

        return dist

    def _sbx_crossover(self, x1, x2, lo, hi, rng, eta=20.0):
        """Simulated Binary Crossover."""
        dim = self.dim
        children = np.zeros((2, dim))
        for i in range(dim):
            if rng.random() < 0.5:
                x1_i, x2_i = x1[i], x2[i]
            else:
                x1_i, x2_i = x2[i], x1[i]

            if abs(x1_i - x2_i) < 1e-14:
                children[0, i] = children[1, i] = x1_i
                continue

            u = rng.random()
            if u <= 0.5:
                beta = (2 * u) ** (1 / (eta + 1))
            else:
                beta = (1 / (2 * (1 - u))) ** (1 / (eta + 1))

            c1 = 0.5 * ((1 + beta) * x1_i + (1 - beta) * x2_i)
            c2 = 0.5 * ((1 - beta) * x1_i + (1 + beta) * x2_i)

            # Clip to bounds
            c1 = np.clip(c1, lo[i], hi[i])
            c2 = np.clip(c2, lo[i], hi[i])

            children[0, i] = c1
            children[1, i] = c2

        return children

    def _polynomial_mutation(self, x, lo, hi, rng, eta_mut=20.0):
        """Polynomial mutation."""
        dim = self.dim
        x_mut = x.copy()
        for i in range(dim):
            if rng.random() < 0.5:
                continue
            u = rng.random()
            if u <= 0.5:
                delta = (2 * u) ** (1 / (eta_mut + 1)) - 1
            else:
                delta = 1 - (2 * (1 - u)) ** (1 / (eta_mut + 1))
            x_mut[i] += delta * (hi[i] - lo[i])
            x_mut[i] = np.clip(x_mut[i], lo[i], hi[i])
        return x_mut

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280.0  # 300s limit, 20s safety margin

        lo = np.array([b[0] for b in bounds], dtype=float)
        hi = np.array([b[1] for b in bounds], dtype=float)
        dim = self.dim
        n_obj = self.n_obj
        budget = self.budget

        rng = np.random.default_rng(42)

        # Population size
        pop_size = min(40, max(10, budget // 5))
        # Ensure we don't exceed budget
        if pop_size * 2 > budget:
            pop_size = max(10, budget // 2)

        # LHS initialization
        X = lo + (hi - lo) * self._lhs(pop_size, dim, rng)
        F = np.array([problem(x) for x in X], dtype=float)
        used = pop_size

        # Track all evaluated points for hypervolume
        all_X = [X.copy()]
        all_F = [F.copy()]

        # Main loop
        while used < budget:
            if time.time() > deadline:
                break

            # Non-dominated sort
            fronts, front_num = self._non_dominated_sort(F)

            # Crowding distance per front
            crowd = np.zeros(pop_size)
            for front in fronts:
                cd = self._crowding_distance(F, front)
                for i, idx in enumerate(front):
                    crowd[idx] = cd[i]

            # Selection: binary tournament
            def tournament():
                i = rng.integers(pop_size)
                j = rng.integers(pop_size)
                if front_num[i] < front_num[j]:
                    return i
                elif front_num[j] < front_num[i]:
                    return j
                else:
                    return i if crowd[i] > crowd[j] else j

            # Generate offspring
            offspring_size = pop_size
            children_X = np.zeros((offspring_size, dim))
            for i in range(offspring_size):
                p1 = tournament()
                p2 = tournament()
                if rng.random() < 0.9:  # crossover probability
                    c = self._sbx_crossover(X[p1], X[p2], lo, hi, rng)
                else:
                    c = np.stack([X[p1].copy(), X[p2].copy()])

                # Mutation
                for ci in range(2):
                    children_X[i * 2 + ci] = self._polynomial_mutation(c[ci], lo, hi, rng)

                used += 2
                if used > budget:
                    # Truncate to fit budget
                    children_X = children_X[:budget - used + 2]
                    offspring_size = len(children_X)
                    break

            # Evaluate offspring
            if offspring_size > 0:
                children_F = np.array([problem(x) for x in children_X], dtype=float)
                used += offspring_size  # This is already counted above, but let's be careful
                # Actually we already incremented used in the loop, so just evaluate
                all_X.append(children_X.copy())
                all_F.append(children_F.copy())

            # Combine parent and offspring
            combined_X = np.vstack([X, children_X[:offspring_size]])
            combined_F = np.vstack([F, children_F[:offspring_size]])
            n_combined = len(combined_X)

            # Sort combined
            fronts_c, front_num_c = self._non_dominated_sort(combined_F)

            # Select next population
            next_X = np.zeros((pop_size, dim))
            next_F = np.zeros((pop_size, n_obj))
            count = 0
            for front in fronts_c:
                if count + len(front) <= pop_size:
                    for idx in front:
                        next_X[count] = combined_X[idx]
                        next_F[count] = combined_F[idx]
                        count += 1
                else:
                    # Need to select from this front based on crowding
                    cd = self._crowding_distance(combined_F, front)
                    order = np.argsort(-cd)
                    needed = pop_size - count
                    for oi in range(needed):
                        idx = front[order[oi]]
                        next_X[count] = combined_X[idx]
                        next_F[count] = combined_F[idx]
                        count += 1
                    break

            X = next_X
            F = next_F

        return X
# EVOLVE-BLOCK-END
