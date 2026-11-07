# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


class Optimizer:
    """Multi-objective optimizer for RE34 using deterministic space-filling + local refinement."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def _halton_sequence(self, n, dim):
        """Generate n points in [0,1]^dim using Halton sequence."""
        points = np.zeros((n, dim))
        for d in range(dim):
            base = d + 2
            for i in range(n):
                f = 1.0
                r = 0.0
                j = i + 1
                while j > 0:
                    f /= base
                    r += f * (j % base)
                    j //= base
                points[i, d] = r
        return points

    def _non_dominated_sort(self, F):
        """Return list of fronts (lists of indices)."""
        n = len(F)
        if n == 0:
            return []
        S = [[] for _ in range(n)]
        nd = np.zeros(n, dtype=int)
        for p in range(n):
            for q in range(n):
                if p == q:
                    continue
                if np.all(F[p] <= F[q]) and np.any(F[p] < F[q]):
                    S[p].append(q)
                    nd[p] += 1
        front0 = [p for p in range(n) if nd[p] == 0]
        fronts = [front0]
        i = 0
        while len(fronts[i]) > 0:
            next_front = []
            for p in fronts[i]:
                for q in S[p]:
                    nd[q] -= 1
                    if nd[q] == 0:
                        next_front.append(q)
            if next_front:
                fronts.append(next_front)
            i += 1
        return fronts

    def _crowding_distance(self, F, indices):
        """Compute crowding distance for given indices."""
        n = len(indices)
        if n <= 2:
            return np.full(n, np.inf)
        cd = np.zeros(n)
        F_sub = F[indices]
        for m in range(self.n_obj):
            order = np.argsort(F_sub[:, m])
            cd[order[0]] = np.inf
            cd[order[-1]] = np.inf
            f_range = F_sub[order[-1], m] - F_sub[order[0], m]
            if f_range == 0:
                continue
            for i in range(1, n - 1):
                cd[order[i]] += (F_sub[order[i + 1], m] - F_sub[order[i - 1], m]) / f_range
        return cd

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280  # safety margin before 300s limit

        lo = np.array([b[0] for b in bounds], dtype=np.float64)
        hi = np.array([b[1] for b in bounds], dtype=np.float64)
        dim = self.dim
        n_obj = self.n_obj
        budget = self.budget
        ref = np.array([1705.0, 12.0, 0.3])

        used = 0

        # Phase 1: Deterministic space-filling exploration
        # Use Halton sequence for uniform coverage
        n_fill = min(100, budget)
        halton = self._halton_sequence(n_fill, dim)
        X_fill = lo + halton * (hi - lo)

        F_fill = np.array([problem(x) for x in X_fill])
        used += n_fill

        # Keep track of all evaluated points
        all_X = X_fill.copy()
        all_F = F_fill.copy()

        if used >= budget:
            return all_X

        # Phase 2: Identify top-k non-dominated points for local refinement
        # Filter points that are strictly better than reference
        valid_mask = np.all(all_F < ref, axis=1)
        valid_indices = np.where(valid_mask)[0]

        if len(valid_indices) == 0:
            # No valid points, just return what we have
            return all_X

        F_valid = all_F[valid_indices]
        X_valid = all_X[valid_indices]

        # Get non-dominated sort
        fronts = self._non_dominated_sort(F_valid)

        # Select top-k points from first few fronts based on crowding distance
        k = min(15, len(valid_indices))
        selected = []
        for front in fronts[:3]:
            if len(selected) + len(front) <= k:
                selected.extend(front)
            else:
                cd = self._crowding_distance(F_valid, front)
                order = np.argsort(-cd)
                remaining = k - len(selected)
                selected.extend(front[order[:remaining]])
                break

        if len(selected) == 0:
            selected = list(range(min(k, len(F_valid))))

        # Phase 3: Local refinement using Nelder-Mead on each selected point
        # For each selected point, minimize a weighted sum of objectives
        # We'll try several weight vectors to explore different parts of the Pareto front
        weight_vectors = self._generate_weight_vectors(n_obj, k)

        n_local_budget = budget - used
        points_per_local = max(1, n_local_budget // (len(selected) * len(weight_vectors)))

        for sel_idx in selected:
            if used >= budget or time.time() > deadline:
                break

            x0 = X_valid[sel_idx]

            for w in weight_vectors:
                if used >= budget or time.time() > deadline:
                    break

                def objective(x, w=w):
                    fx = problem(x)
                    return np.dot(w, fx)

                try:
                    result = minimize(
                        objective, x0, method='Nelder-Mead',
                        bounds=list(zip(lo, hi)),
                        options={'maxiter': points_per_local, 'xatol': 1e-4, 'fatol': 1e-4}
                    )
                    used += points_per_local  # approximate; actual count may differ
                except Exception:
                    continue

                x_new = np.clip(result.x, lo, hi)
                f_new = problem(x_new)

                # Add to our archive if it's valid
                if np.all(f_new < ref):
                    all_X = np.vstack([all_X, [x_new]])
                    all_F = np.vstack([all_F, [f_new]])

        # Final: return all evaluated points
        return all_X

    def _generate_weight_vectors(self, n_obj, k):
        """Generate k weight vectors for weighted sum minimization."""
        if k <= 1:
            return [np.ones(n_obj) / n_obj]

        weights = []
        # Include uniform weights
        weights.append(np.ones(n_obj) / n_obj)

        # Include extreme weights (focus on each objective)
        for i in range(n_obj):
            w = np.ones(n_obj) * 0.1
            w[i] = 1.0
            weights.append(w / np.sum(w))

        # Add some random weights to fill up to k
        while len(weights) < k:
            w = np.random.dirichlet(np.ones(n_obj))
            weights.append(w)

        return weights[:k]


# EVOLVE-BLOCK-END
