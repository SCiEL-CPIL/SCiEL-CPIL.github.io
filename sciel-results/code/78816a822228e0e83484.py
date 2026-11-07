# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-objective optimizer for RE34: NSGA-II with adaptive crossover."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def _fast_non_dominated_sort(self, F):
        """Fast non-dominated sorting. Returns list of lists of indices."""
        n = len(F)
        if n == 0:
            return []

        S = [[] for _ in range(n)]
        n_dom = np.zeros(n, dtype=int)
        fronts = []

        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                # Check if i dominates j
                if np.all(F[i] <= F[j]) and np.any(F[i] < F[j]):
                    S[i].append(j)
                    n_dom[i] += 1
                elif np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                    S[j].append(i)
                    n_dom[j] += 1

        current_front = [i for i in range(n) if n_dom[i] == 0]
        if not current_front:
            return [list(range(n))]

        front_idx = 0
        while current_front:
            fronts.append(current_front)
            next_front = []
            for i in current_front:
                for j in S[i]:
                    n_dom[j] -= 1
                    if n_dom[j] == 0:
                        next_front.append(j)
            current_front = next_front
            front_idx += 1

        return fronts

    def _crowding_distance(self, F, indices):
        """Compute crowding distances for a set of indices."""
        n = len(indices)
        if n <= 2:
            return np.full(n, np.inf)

        dist = np.zeros(n)
        F_sub = F[indices]

        for m in range(self.n_obj):
            order = np.argsort(F_sub[:, m])
            dist[order[0]] = np.inf
            dist[order[-1]] = np.inf
            f_range = F_sub[order[-1], m] - F_sub[order[0], m]
            if f_range == 0:
                continue
            for i in range(1, n - 1):
                dist[order[i]] += (F_sub[order[i + 1], m] - F_sub[order[i - 1], m]) / f_range

        return dist

    def _hv_2d(self, pts, ref):
        """2D hypervolume. pts: (m,2) array, all < ref. ref: (2,) array."""
        m = len(pts)
        if m == 0:
            return 0.0
        if m == 1:
            return float((ref[0] - pts[0, 0]) * (ref[1] - pts[0, 1]))
        idx = np.argsort(pts[:, 0])
        pts = pts[idx]
        hv = 0.0
        for i in range(m):
            x_cur = pts[i, 0]
            x_next = pts[i + 1, 0] if i + 1 < m else ref[0]
            y_min = np.min(pts[:i + 1, 1])
            hv += (x_next - x_cur) * (ref[1] - y_min)
        return hv

    def _hv_3d(self, pts, ref):
        """3D hypervolume. pts: (n,3) array, all < ref. ref: (3,) array."""
        n = len(pts)
        if n == 0:
            return 0.0
        if n == 1:
            return float(np.prod(ref - pts[0]))
        idx = np.argsort(pts[:, 0])
        pts = pts[idx]
        hv = 0.0
        for i in range(n):
            x_cur = pts[i, 0]
            x_next = pts[i + 1, 0] if i + 1 < n else ref[0]
            yz = pts[:i + 1, 1:3]
            hv += (x_next - x_cur) * self._hv_2d(yz, ref[1:3])
        return hv

    def _is_pareto(self, F):
        """Return boolean mask of Pareto-optimal points in F (n x 3)."""
        n = len(F)
        if n == 0:
            return np.array([], dtype=bool)
        pareto = np.ones(n, dtype=bool)
        for i in range(n):
            if not pareto[i]:
                continue
            self_mask = np.ones(n, dtype=bool)
            self_mask[i] = False
            dominated = np.any(np.all(F[self_mask] <= F[i], axis=1) & np.any(F[self_mask] < F[i], axis=1))
            if dominated:
                pareto[i] = False
        return pareto

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 300  # 30s safety margin before 330s limit

        lo = np.array([b[0] for b in bounds], dtype=np.float64)
        hi = np.array([b[1] for b in bounds], dtype=np.float64)
        dim = self.dim
        n_obj = self.n_obj
        budget = self.budget
        ref = np.array([1705.0, 12.0, 0.3])

        rng = np.random  # Use the global seeded RNG

        # --- Phase 1: Initial population using LHS ---
        pop_size = min(40, budget)
        lhs_X = np.zeros((pop_size, dim))
        for d in range(dim):
            perm = np.random.permutation(pop_size)
            lhs_X[:, d] = lo[d] + (hi[d] - lo[d]) * (perm + np.random.uniform(size=pop_size)) / pop_size
        lhs_X = np.clip(lhs_X, lo, hi)

        F = np.array([problem(x) for x in lhs_X])
        used = pop_size

        # Store all evaluated points
        all_X = lhs_X.copy()
        all_F = F.copy()

        # --- Phase 2: Evolutionary loop ---
        # Adaptive population size
        base_pop_size = 30
        while used < budget:
            if time.time() > deadline:
                break

            progress = used / budget
            # Population size decreases slightly with progress
            pop_size = max(15, int(base_pop_size * (1.0 - 0.3 * progress)))

            # Get current non-dominated front from all_F
            valid = np.all(all_F < ref, axis=1)
            if np.any(valid):
                valid_F = all_F[valid]
                valid_X = all_X[valid]
                p_mask = self._is_pareto(valid_F)
                if np.any(p_mask):
                    pareto_F = valid_F[p_mask]
                    pareto_X = valid_X[p_mask]
                else:
                    pareto_F = valid_F[:1]
                    pareto_X = valid_X[:1]
            else:
                # No valid points, use best available
                best_idx = np.argmin(np.sum(all_F, axis=1))
                pareto_F = all_F[best_idx:best_idx+1]
                pareto_X = all_X[best_idx:best_idx+1]

            # Generate offspring
            offspring_X = []
            for _ in range(pop_size):
                if time.time() > deadline:
                    break

                # Select two parents from Pareto front
                n_par = len(pareto_F)
                if n_par >= 2:
                    # Tournament selection based on HV contribution
                    idx1 = np.random.randint(n_par)
                    idx2 = np.random.randint(n_par)
                    # Simple selection: pick randomly from Pareto
                    p1 = pareto_X[idx1]
                    p2 = pareto_X[idx2]
                else:
                    p1 = pareto_X[0]
                    p2 = pareto_X[0]

                # Crossover: SBX-like with adaptive eta
                eta = 20.0 * (1.0 - progress) + 5.0
                if np.random.uniform() < 0.9:  # 90% crossover
                    child = np.zeros(dim)
                    for d in range(dim):
                        if np.random.uniform() < 0.5:
                            beta = 1.0 + 2.0 * np.abs(p1[d] - p2[d])
                            if abs(p1[d] - p2[d]) < 1e-10:
                                beta = 1.0
                            rand = np.random.uniform()
                            if rand <= 1.0 / (beta + 1.0):
                                alpha = (rand * (beta + 1.0)) ** (1.0 / (eta + 1.0))
                            else:
                                alpha = (1.0 / (2.0 - rand * (beta + 1.0))) ** (1.0 / (eta + 1.0))
                            child[d] = 0.5 * ((1.0 + alpha) * p1[d] + (1.0 - alpha) * p2[d])
                        else:
                            beta = 1.0 + 2.0 * np.abs(p2[d] - p1[d])
                            if abs(p2[d] - p1[d]) < 1e-10:
                                beta = 1.0
                            rand = np.random.uniform()
                            if rand <= 1.0 / (beta + 1.0):
                                alpha = (rand * (beta + 1.0)) ** (1.0 / (eta + 1.0))
                            else:
                                alpha = (1.0 / (2.0 - rand * (beta + 1.0))) ** (1.0 / (eta + 1.0))
                            child[d] = 0.5 * ((1.0 + alpha) * p2[d] + (1.0 - alpha) * p1[d])
                else:
                    child = p1.copy()

                # Mutation
                mut_prob = 1.0 / dim
                for d in range(dim):
                    if np.random.uniform() < mut_prob:
                        # Polynomial mutation
                        rand = np.random.uniform()
                        if rand < 0.5:
                            mut_power = 1.0 / (eta_m + 1.0) if (eta_m := 20.0 * (1.0 - progress) + 5.0) else 0.1
                            delta1 = (2.0 * rand) ** mut_power - 1.0
                        else:
                            mut_power = 1.0 / (eta_m + 1.0) if (eta_m := 20.0 * (1.0 - progress) + 5.0) else 0.1
                            delta2 = 1.0 - (2.0 * (1.0 - rand)) ** mut_power
                            delta1 = -delta2
                        child[d] += delta1 * (hi[d] - lo[d])

                child = np.clip(child, lo, hi)
                offspring_X.append(child)

            if len(offspring_X) == 0:
                break

            offspring_X = np.array(offspring_X)
            offspring_F = np.array([problem(x) for x in offspring_X])
            used += len(offspring_X)

            all_X = np.vstack([all_X, offspring_X])
            all_F = np.vstack([all_F, offspring_F])

            # Update Pareto front
            valid = np.all(all_F < ref, axis=1)
            if np.any(valid):
                valid_F = all_F[valid]
                valid_X = all_X[valid]
                p_mask = self._is_pareto(valid_F)
                if np.any(p_mask):
                    pareto_F = valid_F[p_mask]
                    pareto_X = valid_X[p_mask]
                else:
                    pareto_F = valid_F[:1]
                    pareto_X = valid_X[:1]

            # Prune archive if too large
            if len(pareto_F) > 100:
                # Keep top 100 by HV contribution approximation
                # Use simple random subsampling
                if len(pareto_F) > 100:
                    keep_idx = np.random.choice(len(pareto_F), size=100, replace=False)
                    pareto_F = pareto_F[keep_idx]
                    pareto_X = pareto_X[keep_idx]

        return all_X

# EVOLVE-BLOCK-END
