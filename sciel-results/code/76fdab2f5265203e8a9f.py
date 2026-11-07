# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Vectorized NSGA-II for WFG9 (3 objectives, 12 dims) with explicit deadline guard."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def _dominates(self, F):
        """Return boolean matrix D where D[i,j] is True if row i dominates row j."""
        # F shape: (n, m)
        # i dominates j if F[i] <= F[j] for all objectives AND F[i] < F[j] for at least one
        leq = (F[:, None, :] <= F[None, :, :])  # (n, n, m)
        lt = (F[:, None, :] < F[None, :, :])    # (n, n, m)
        return leq.all(axis=2) & lt.any(axis=2)

    def _fast_non_dominated_sort(self, F):
        """Return list of fronts (each a list of indices), sorted by rank."""
        n = F.shape[0]
        S = [[] for _ in range(n)]  # S[i] = indices dominated by i
        n_count = np.zeros(n, dtype=int)  # number of solutions that dominate i
        rank = np.zeros(n, dtype=int)

        D = self._dominates(F)  # D[i,j] = i dominates j

        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                if D[i, j]:
                    S[i].append(j)
                    n_count[j] += 1
                elif D[j, i]:
                    pass  # j dominates i, already counted when j was processed
                else:
                    pass

        # Actually, let me redo this more carefully
        S = [[] for _ in range(n)]
        n_count = np.zeros(n, dtype=int)

        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                if D[i, j]:
                    S[i].append(j)
                    n_count[j] += 1

        front = []
        for i in range(n):
            if n_count[i] == 0:
                front.append(i)
                rank[i] = 0

        fronts = []
        current_front = front
        fronts.append(current_front)

        while current_front:
            next_front = []
            for i in current_front:
                for j in S[i]:
                    n_count[j] -= 1
                    if n_count[j] == 0:
                        rank[j] = rank[i] + 1
                        next_front.append(j)
            if next_front:
                fronts.append(next_front)
            current_front = next_front

        return fronts

    def _crowding_distance(self, F, indices):
        """Compute crowding distance for a set of indices in F."""
        n = len(indices)
        if n <= 2:
            return np.full(n, np.inf)

        cd = np.zeros(n)
        m = self.n_obj
        for k in range(m):
            vals = F[indices, k]
            order = np.argsort(vals)
            cd[order[0]] = np.inf
            cd[order[-1]] = np.inf
            val_range = vals[order[-1]] - vals[order[0]]
            if val_range > 1e-15:
                for i in range(1, n - 1):
                    cd[order[i]] += (vals[order[i + 1]] - vals[order[i - 1]]) / val_range
        return cd

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 290.0  # 30s safety margin before 320s kill

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        rng = np.random.default_rng(0)

        budget = self.budget
        dim = self.dim
        n_obj = self.n_obj

        # Population size
        pop_size = 40
        # Ensure we don't exceed budget
        if pop_size >= budget:
            pop_size = max(2, budget // 2)

        # Initialize population
        X = lo + (hi - lo) * rng.random((pop_size, dim))
        F = np.array([problem(x) for x in X])
        used = pop_size

        # Track all evaluated points for hypervolume (the evaluator records every returned objective vector)
        # Actually, the evaluator records every objective vector returned by problem(x).
        # We need to be careful: the problem function is called, and its return is recorded.
        # So we just need to call problem() at most budget times.

        # NSGA-II loop
        while used < budget:
            if time.time() > deadline:
                break

            # Create offspring
            offspring_size = pop_size
            X_off = np.empty((offspring_size, dim))
            F_off = np.empty((offspring_size, n_obj))

            for i in range(offspring_size):
                if time.time() > deadline:
                    # Fill remaining with copies of best to avoid partial
                    X_off[i] = X[0]
                    F_off[i] = F[0]
                    continue
                if used + 1 > budget:
                    break

                # Binary tournament selection
                idx1 = rng.integers(pop_size)
                idx2 = rng.integers(pop_size)

                # We need ranks. Let's compute them once per generation
                # Actually, let's compute ranks before the offspring loop

            # Recompute ranks for parent selection
            fronts = self._fast_non_dominated_sort(F)
            ranks = np.zeros(pop_size, dtype=int)
            for r, front in enumerate(fronts):
                for idx in front:
                    ranks[idx] = r

            # Crowding distance per front
            cd = np.zeros(pop_size)
            for front in fronts:
                if len(front) > 2:
                    cd[front] = self._crowding_distance(F, np.array(front))

            # Selection
            def select():
                # Tournament
                idx1 = rng.integers(pop_size)
                idx2 = rng.integers(pop_size)
                # Compare by rank then crowding distance
                if ranks[idx1] < ranks[idx2]:
                    return idx1
                elif ranks[idx2] < ranks[idx1]:
                    return idx2
                else:
                    if cd[idx1] > cd[idx2]:
                        return idx1
                    else:
                        return idx2

            for i in range(offspring_size):
                if used + 1 > budget:
                    break
                if time.time() > deadline:
                    break

                p1 = select()
                p2 = select()

                # Simulated binary crossover (SBX)
                if rng.random() < 0.9:
                    eta = 20.0
                    beta1 = 2.0 ** (-1.0 / (eta + 1.0))
                    beta2 = 2.0 ** (-1.0 / (eta + 1.0))

                    u1 = rng.random()
                    if u1 <= 0.5:
                        alpha = 1.0 + 2.0 * (p1 - p2) / (hi - lo + 1e-15)
                        beta = (2.0 - alpha * beta1) ** (-1.0 / (eta + 1.0)) if alpha * beta1 < 2.0 else 1.0
                        child1 = 0.5 * ((1.0 + beta) * p1 + (1.0 - beta) * p2)
                    else:
                        alpha = 1.0 + 2.0 * (p2 - p1) / (hi - lo + 1e-15)
                        beta = (2.0 - alpha * beta1) ** (-1.0 / (eta + 1.0)) if alpha * beta1 < 2.0 else 1.0
                        child1 = 0.5 * ((1.0 - beta) * p1 + (1.0 + beta) * p2)

                    u2 = rng.random()
                    if u2 <= 0.5:
                        alpha = 1.0 + 2.0 * (p1 - p2) / (hi - lo + 1e-15)
                        beta = (2.0 - alpha * beta2) ** (-1.0 / (eta + 1.0)) if alpha * beta2 < 2.0 else 1.0
                        child2 = 0.5 * ((1.0 + beta) * p2 + (1.0 - beta) * p1)
                    else:
                        alpha = 1.0 + 2.0 * (p2 - p1) / (hi - lo + 1e-15)
                        beta = (2.0 - alpha * beta2) ** (-1.0 / (eta + 1.0)) if alpha * beta2 < 2.0 else 1.0
                        child2 = 0.5 * ((1.0 - beta) * p2 + (1.0 + beta) * p1)
                else:
                    child1 = p1.copy()
                    child2 = p2.copy()

                # Polynomial mutation
                eta_m = 20.0
                for d in range(dim):
                    if rng.random() < 1.0 / dim:
                        u = rng.random()
                        if u < 0.5:
                            delta1 = (2.0 * u) ** (1.0 / (eta_m + 1.0)) - 1.0
                        else:
                            delta1 = 1.0 - (2.0 * (1.0 - u)) ** (1.0 / (eta_m + 1.0))
                        child1[d] += delta1 * (hi[d] - lo[d])
                    if rng.random() < 1.0 / dim:
                        u = rng.random()
                        if u < 0.5:
                            delta2 = (2.0 * u) ** (1.0 / (eta_m + 1.0)) - 1.0
                        else:
                            delta2 = 1.0 - (2.0 * (1.0 - u)) ** (1.0 / (eta_m + 1.0))
                        child2[d] += delta2 * (hi[d] - lo[d])

                child1 = np.clip(child1, lo, hi)
                child2 = np.clip(child2, lo, hi)

                # Evaluate
                if used + 1 <= budget:
                    F_off[i] = problem(child1)
                    X_off[i] = child1
                    used += 1
                if used + 1 <= budget:
                    # Wait, we need to evaluate both children? That would use 2 evaluations per offspring slot.
                    # Let's just evaluate one child per iteration to stay within budget more easily.
                    pass

            # Actually, let me simplify: evaluate one child per loop iteration
            # Redo the offspring generation

            # Let me restart the offspring generation with a simpler approach
            X_off = np.empty((min(offspring_size, budget - used), dim))
            F_off = np.empty((min(offspring_size, budget - used), n_obj))
            n_off = 0

            for i in range(offspring_size):
                if used >= budget:
                    break
                if time.time() > deadline:
                    break

                p1 = select()
                p2 = select()

                # SBX crossover
                if rng.random() < 0.9:
                    eta = 20.0
                    u = rng.random()
                    if u <= 0.5:
                        alpha = 1.0 + 2.0 * (p1 - p2) / (hi - lo + 1e-15)
                        beta = (2.0 - alpha * 2.0 ** (-1.0 / (eta + 1.0))) ** (-1.0 / (eta + 1.0)) if alpha * 2.0 ** (-1.0 / (eta + 1.0)) < 2.0 else 1.0
                        child = 0.5 * ((1.0 + beta) * p1 + (1.0 - beta) * p2)
                    else:
                        alpha = 1.0 + 2.0 * (p2 - p1) / (hi - lo + 1e-15)
                        beta = (2.0 - alpha * 2.0 ** (-1.0 / (eta + 1.0))) ** (-1.0 / (eta + 1.0)) if alpha * 2.0 ** (-1.0 / (eta + 1.0)) < 2.0 else 1.0
                        child = 0.5 * ((1.0 - beta) * p1 + (1.0 + beta) * p2)
                else:
                    child = p1.copy()

                # Mutation
                eta_m = 20.0
                for d in range(dim):
                    if rng.random() < 1.0 / dim:
                        u = rng.random()
                        if u < 0.5:
                            delta = (2.0 * u) ** (1.0 / (eta_m + 1.0)) - 1.0
                        else:
                            delta = 1.0 - (2.0 * (1.0 - u)) ** (1.0 / (eta_m + 1.0))
                        child[d] += delta * (hi[d] - lo[d])

                child = np.clip(child, lo, hi)
                X_off[n_off] = child
                F_off[n_off] = problem(child)
                used += 1
                n_off += 1

            X_off = X_off[:n_off]
            F_off = F_off[:n_off]

            if n_off == 0:
                break

            # Combine parent and offspring
            X_all = np.vstack([X, X_off])
            F_all = np.vstack([F, F_off])

            # Non-dominated sort
            fronts = self._fast_non_dominated_sort(F_all)

            # Select next population
            new_X = np.empty((pop_size, dim))
            new_F = np.empty((pop_size, n_obj))
            n_selected = 0

            for front in fronts:
                if n_selected + len(front) <= pop_size:
                    for idx in front:
                        new_X[n_selected] = X_all[idx]
                        new_F[n_selected] = F_all[idx]
                        n_selected += 1
                else:
                    # Need to select from this front based on crowding distance
                    remaining = pop_size - n_selected
                    cd = self._crowding_distance(F_all, np.array(front))
                    order = np.argsort(-cd)
                    for i in range(min(remaining, len(front))):
                        idx = front[order[i]]
                        new_X[n_selected] = X_all[idx]
                        new_F[n_selected] = F_all[idx]
                        n_selected += 1
                    break

            X = new_X[:n_selected]
            F = new_F[:n_selected]
            pop_size = n_selected

        return X
# EVOLVE-BLOCK-END
