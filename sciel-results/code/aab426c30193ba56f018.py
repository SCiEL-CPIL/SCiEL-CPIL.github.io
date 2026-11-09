# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-objective optimizer for DTLZ1 using a Sobol-based initial sample
    plus a fast evolutionary refinement with deadline guard."""

    def __init__(self, budget, dim, n_obj):
        self.budget, self.dim, self.n_obj = budget, dim, n_obj

    def __call__(self, problem, bounds):
        t0 = time.time()
        deadline = t0 + 280.0  # safety margin before 300s limit

        lo = np.array([b[0] for b in bounds], dtype=np.float64)
        hi = np.array([b[1] for b in bounds], dtype=np.float64)
        dim = self.dim
        n_obj = self.n_obj
        budget = self.budget
        ref = np.array([400.0] * n_obj)

        # Sobol sequence for initial population
        n_init = min(64, budget // 3)
        X_init = self._sobol(n_init, dim)
        X_init = lo + (hi - lo) * X_init

        # Evaluate initial population
        F_init = np.array([problem(x) for x in X_init])
        used = n_init

        # Keep track of all evaluated points
        all_X = list(X_init)
        all_F = list(F_init)

        # Non-dominated sort
        def non_dominated_mask(F):
            n = len(F)
            mask = np.ones(n, dtype=bool)
            for i in range(n):
                if not mask[i]:
                    continue
                for j in range(n):
                    if i == j or not mask[j]:
                        continue
                    if np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                        mask[i] = False
                        break
            return mask

        # Crowding distance
        def crowding_dist(F, idx):
            n = len(idx)
            cd = np.zeros(n)
            if n <= 2:
                return cd
            for m in range(n_obj):
                vals = F[idx, m]
                min_v, max_v = vals.min(), vals.max()
                if max_v - min_v < 1e-12:
                    continue
                order = np.argsort(vals)
                cd[order[0]] = np.inf
                cd[order[-1]] = np.inf
                for k in range(1, n - 1):
                    cd[order[k]] += (vals[order[k + 1]] - vals[order[k - 1]]) / (max_v - min_v)
            return cd

        # Main evolutionary loop
        pop_size = min(30, budget // 4)
        # Select initial population from non-dominated set
        nd_mask = non_dominated_mask(F_init)
        if np.sum(nd_mask) >= pop_size:
            # Use crowding to select
            idx_nd = np.where(nd_mask)[0]
            cd = crowding_dist(F_init, idx_nd)
            sel_idx = idx_nd[np.argsort(cd)[::-1][:pop_size]]
        else:
            sel_idx = np.arange(n_init)[:pop_size]

        X = np.array([all_X[i] for i in sel_idx])
        F = np.array([all_F[i] for i in sel_idx])

        rng = np.random.default_rng(42)

        while used < budget:
            # Check deadline
            if time.time() > deadline:
                break

            # Non-dominated sort
            nd_mask = non_dominated_mask(F)
            if np.any(nd_mask):
                parents_idx = np.where(nd_mask)[0]
            else:
                parents_idx = np.arange(len(F))

            # Tournament selection
            p1 = parents_idx[rng.integers(len(parents_idx))]
            p2 = parents_idx[rng.integers(len(parents_idx))]

            # Crossover (SBX-like)
            alpha = 0.5
            child = alpha * X[p1] + (1 - alpha) * X[p2]

            # Mutation
            sigma = 0.05 * (hi - lo)
            child = child + sigma * rng.standard_normal(dim)
            child = np.clip(child, lo, hi)

            fc = problem(child)
            used += 1

            # Replace worst point in population
            # Use a simple dominance-based replacement
            worst_idx = self._find_replacement(F, fc)
            X[worst_idx] = child
            F[worst_idx] = fc

            all_X.append(child)
            all_F.append(fc)

        # Final: collect all non-dominated points from all evaluations
        all_F_arr = np.array(all_F)
        # Filter out points dominated by reference
        valid = np.all(all_F_arr < ref, axis=1)
        all_F_valid = all_F_arr[valid]
        all_X_valid = np.array(all_X)[valid]

        if len(all_F_valid) == 0:
            return all_X

        nd_final = non_dominated_mask(all_F_valid)
        return all_X_valid[nd_final]

    def _find_replacement(self, F, fc):
        """Find the index to replace in population F with new point fc."""
        n = len(F)
        # If fc dominates some points, replace the first dominated one
        for i in range(n):
            if np.all(fc <= F[i]) and np.any(fc < F[i]):
                return i
        # If fc is non-dominated, replace the point with worst crowding
        # Simple heuristic: replace point with largest sum of objectives
        sums = np.sum(F, axis=1)
        return int(np.argmax(sums))

    @staticmethod
    def _sobol(n, dim):
        """Generate n points in [0,1]^dim using a simple Sobol-like sequence.
        Uses a basic van der Corput / Halton-style low-discrepancy sequence."""
        points = np.zeros((n, dim))
        for d in range(dim):
            base = d + 2  # primes: 2, 3, 5, 7, 11, 13, 17
            for i in range(n):
                val = 0.0
                denom = 1.0
                k = i + 1
                while k > 0:
                    k, r = divmod(k, base)
                    val += r / denom
                    denom *= base
                points[i, d] = val
        return points
# EVOLVE-BLOCK-END
