"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Hypervolume-guided adaptive differential evolution for RE21."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def _hypervolume_2d(self, F, ref):
        """Compute exact 2D hypervolume of points dominated by F with reference point ref."""
        if len(F) == 0:
            return 0.0
        # Filter points strictly better than ref in all objectives
        mask = np.all(F < ref, axis=1)
        F = F[mask]
        if len(F) == 0:
            return 0.0
        # Sort by first objective (minimizing)
        order = np.argsort(F[:, 0])
        F = F[order]
        hv = 0.0
        prev_y = ref[1]
        for i in range(len(F)):
            f0, f1 = F[i]
            if f1 >= ref[1]:
                continue
            # Width in obj0 direction: from f0 to next point's f0 (or ref[0])
            if i + 1 < len(F):
                next_f0 = F[i + 1, 0]
            else:
                next_f0 = ref[0]
            width = next_f0 - f0
            if width > 0:
                height = prev_y - f1
                if height > 0:
                    hv += width * height
            prev_y = min(prev_y, f1)
        return hv

    def _non_dominated_mask(self, F):
        """Return boolean mask for non-dominated points."""
        n = len(F)
        nd = np.ones(n, dtype=bool)
        for i in range(n):
            if not nd[i]:
                continue
            for j in range(n):
                if i == j:
                    continue
                # j dominates i if j <= i in all and j < i in at least one
                if np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                    nd[i] = False
                    break
        return nd

    def _crowding_distance(self, F, nd_mask):
        """Compute crowding distances for non-dominated points."""
        n = len(F)
        dist = np.zeros(n)
        if not np.any(nd_mask):
            return dist
        idx = np.where(nd_mask)[0]
        k = len(idx)
        if k <= 2:
            dist[idx] = np.inf
            return dist
        for m in range(self.n_obj):
            order = np.argsort(F[idx, m])
            min_val = F[idx[order[0]], m]
            max_val = F[idx[order[-1]], m]
            if max_val - min_val < 1e-15:
                continue
            dist[idx[order[0]]] = np.inf
            dist[idx[order[-1]]] = np.inf
            for i in range(1, k - 1):
                dist[idx[order[i]]] += (F[idx[order[i + 1]], m] - F[idx[order[i - 1]], m]) / (max_val - min_val)
        return dist

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 320.0  # Safety margin before 330s kill

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        rng = np.random.default_rng(42)

        budget = self.budget
        dim = self.dim
        n_obj = self.n_obj

        # Reference point for hypervolume computation
        ref = np.array([3000.0, 0.06])

        # Deterministic grid initialization to saturate asymmetric space
        # RE21: obj1 ~ range [0, ~2000+], obj2 ~ range [0, ~0.05]
        # Grid in objective space, mapped back via random sampling
        pop_size = 50
        X = np.zeros((pop_size, dim))
        F = np.zeros((pop_size, n_obj))

        # Phase 1: Deterministic spread initialization
        # Use a mix of corner points and grid points
        corners = np.array([
            lo,
            hi,
            np.array([lo[0], lo[1], hi[2], hi[3]]),
            np.array([hi[0], hi[1], lo[2], lo[3]]),
            np.array([lo[0], hi[1], lo[2], hi[3]]),
            np.array([hi[0], lo[1], hi[2], lo[3]]),
            np.array([lo[0], hi[1], hi[2], lo[3]]),
            np.array([hi[0], lo[1], lo[2], hi[3]]),
        ])
        n_corners = min(len(corners), pop_size)
        for i in range(n_corners):
            X[i] = corners[i]
            F[i] = problem(X[i])

        # Fill rest with stratified random
        remaining = pop_size - n_corners
        if remaining > 0:
            X[n_corners:] = lo + (hi - lo) * rng.random((remaining, dim))
            for i in range(n_corners, pop_size):
                F[i] = problem(X[i])

        used = pop_size

        # Adaptive DE loop
        # Dynamically shrinking mutation factor
        f_start = 0.8
        f_end = 0.3
        gen = 0
        max_gen = budget // pop_size + 10

        # Track best hypervolume
        best_hv = self._hypervolume_2d(F, ref)

        while used < budget and gen < max_gen:
            if time.time() > deadline:
                break

            # Compute non-dominated front
            nd_mask = self._non_dominated_mask(F)
            n_nd = np.sum(nd_mask)

            # Adaptive mutation factor
            progress = gen / max_gen
            f_mut = f_start - (f_start - f_end) * progress

            # Generate offspring
            n_off = pop_size
            for _ in range(n_off):
                if used >= budget:
                    break
                if time.time() > deadline:
                    break

                # Select parents from non-dominated set if possible
                if n_nd >= 3:
                    idx = np.where(nd_mask)[0]
                    i1, i2, i3 = rng.choice(idx, size=3, replace=False)
                else:
                    i1, i2, i3 = rng.choice(pop_size, size=3, replace=False)

                # Differential evolution mutation
                v = X[i1] + f_mut * (X[i2] - X[i3])
                # Crossover
                cr = 0.9
                mask = rng.random(dim) < cr
                if not np.any(mask):
                    mask[rng.integers(dim)] = True
                trial = np.where(mask, v, X[i1])
                trial = np.clip(trial, lo, hi)

                ft = problem(trial)
                used += 1

                # Replacement: compare with worst in population
                # Use hypervolume contribution heuristic
                # Replace the point with lowest hypervolume marginal contribution
                # Approximate: replace worst by sum of objectives normalized
                norm_f = F / (hi - lo + 1e-10)
                scores = np.sum(norm_f, axis=1)
                worst_idx = np.argmax(scores)

                # Only replace if trial is better or non-dominated
                if np.all(ft <= F[worst_idx]) and np.any(ft < F[worst_idx]):
                    X[worst_idx] = trial
                    F[worst_idx] = ft
                elif not np.all(F[worst_idx] <= ft) and np.any(F[worst_idx] < ft):
                    # trial is not dominated by worst, replace
                    X[worst_idx] = trial
                    F[worst_idx] = ft
                elif np.all(ft < F[worst_idx]):
                    X[worst_idx] = trial
                    F[worst_idx] = ft

            gen += 1

            # Update best hypervolume
            current_hv = self._hypervolume_2d(F, ref)
            if current_hv > best_hv:
                best_hv = current_hv

        return X
# EVOLVE-BLOCK-END
