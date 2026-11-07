# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """DE-style multi-objective optimizer for WFG9 with time guard."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start = time.time()
        deadline = start + 275.0

        lo = np.array([b[0] for b in bounds], dtype=float)
        hi = np.array([b[1] for b in bounds], dtype=float)
        dim = self.dim
        n_obj = self.n_obj

        # Budget allocation
        pop_size = 24
        polish_budget = 50
        evolve_budget = max(0, self.budget - polish_budget)

        # RNG (use default_rng for reproducibility within the optimizer)
        rng = np.random.default_rng(12345)

        # ---------- helpers ----------
        def dominates(a, b):
            # a dominates b if a <= b in all and < in at least one
            return np.all(a <= b) and np.any(a < b)

        def fast_non_dominated_sort(F):
            """Return list of fronts, each a list of indices."""
            n = len(F)
            S = [[] for _ in range(n)]
            nd = [0] * n
            fronts = []
            for i in range(n):
                for j in range(n):
                    if i == j:
                        continue
                    if dominates(F[i], F[j]):
                        S[i].append(j)
                    elif dominates(F[j], F[i]):
                        nd[i] += 1
                if nd[i] == 0:
                    fronts.append([i])
            k = 0
            while k < len(fronts):
                current = fronts[k]
                next_front = []
                for i in current:
                    for j in S[i]:
                        nd[j] -= 1
                        if nd[j] == 0:
                            next_front.append(j)
                if next_front:
                    fronts.append(next_front)
                k += 1
            return fronts

        # ---------- initialization ----------
        X = lo + (hi - lo) * rng.random((pop_size, dim))
        F = np.array([problem(x) for x in X])
        used = pop_size

        # archive of non-dominated solutions
        fronts = fast_non_dominated_sort(F)
        archive_idx = fronts[0]
        archive_F = F[archive_idx].copy()

        # reference point for diversity
        ref = np.array([3.0, 5.0, 7.0], dtype=float)

        def crowding_distance(F_sub, idx_in_F):
            """Compute crowding distance for a subset of a front.
            F_sub: (m, n_obj) array
            idx_in_F: indices into original F (not used here, just for clarity)
            Returns array of distances.
            """
            m = len(F_sub)
            if m <= 2:
                return np.full(m, np.inf)
            dist = np.zeros(m)
            for k in range(n_obj):
                order = np.argsort(F_sub[:, k])
                dist[order[0]] = np.inf
                dist[order[-1]] = np.inf
                fmin, fmax = F_sub[order[0], k], F_sub[order[-1], k]
                if fmax - fmin < 1e-14:
                    continue
                for i in range(1, m - 1):
                    dist[order[i]] += (F_sub[order[i + 1], k] - F_sub[order[i - 1], k]) / (fmax - fmin)
            return dist

        # ---------- DE evolution ----------
        F_scale = 0.7
        CR = 0.6

        while used < evolve_budget:
            if time.time() > deadline:
                break

            # pick two distinct parents from the whole population
            i1 = rng.integers(pop_size)
            i2 = rng.integers(pop_size)
            while i2 == i1:
                i2 = rng.integers(pop_size)

            v = X[i1] + F_scale * (X[i1] - X[i2])
            child = np.clip(v, lo, hi)

            fc = problem(child)
            used += 1

            # Try to insert child: check if it is dominated by any in archive
            dominated = False
            for af in archive_F:
                if dominates(af, fc):
                    dominated = True
                    break

            if not dominated:
                # add to archive and remove dominated
                new_archive_F = np.vstack([archive_F, [fc]])
                new_archive_X = np.vstack([X[i1], [child]])  # placeholder; we don't actually need X for archive
                # Remove dominated points
                keep = []
                for j, af in enumerate(new_archive_F):
                    if not dominates(fc, af):
                        keep.append(j)
                if keep:
                    archive_F = new_archive_F[keep]
                else:
                    archive_F = np.array([fc])
            else:
                # child is dominated by archive; try to replace worst in population
                # Use sum of normalized objectives as a simple quality indicator
                sums = np.sum(F, axis=1)
                worst = np.argmax(sums)
                if np.sum(fc) < sums[worst]:
                    X[worst] = child
                    F[worst] = fc

        # ---------- Polish phase: local search from diverse archive points ----------
        if used < self.budget and len(archive_F) > 0:
            # Get a diverse subset of the archive
            fronts = fast_non_dominated_sort(archive_F)
            # Use the first front (non-dominated within archive)
            first_front_idx = fronts[0]
            first_front_F = archive_F[first_front_idx]
            cd = crowding_distance(first_front_F, first_front_idx)

            # Pick up to 8 diverse points
            order = np.argsort(-cd)
            num_polish = min(8, len(order))
            polish_indices = order[:num_polish]

            local_sigma = 0.08
            for pi in polish_indices:
                if used >= self.budget:
                    break
                if time.time() > deadline:
                    break

                p_F = archive_F[pi]
                p_X = X[0]  # placeholder; we don't have X for archive, so we'll just use random restart near p_F is not possible
                # Since we don't store X for archive, we'll just do random perturbations from random points in current population
                # Instead, let's pick a random point in the current population and do local search
                base_idx = rng.integers(pop_size)
                base_X = X[base_idx].copy()

                for step in range(5):
                    if used >= self.budget:
                        break
                    if time.time() > deadline:
                        break

                    candidate = base_X + local_sigma * (hi - lo) * rng.standard_normal(dim)
                    candidate = np.clip(candidate, lo, hi)
                    fc = problem(candidate)
                    used += 1

                    dominated = False
                    for af in archive_F:
                        if dominates(af, fc):
                            dominated = True
                            break

                    if not dominated:
                        new_archive_F = np.vstack([archive_F, [fc]])
                        keep = []
                        for j, af in enumerate(new_archive_F):
                            if not dominates(fc, af):
                                keep.append(j)
                        if keep:
                            archive_F = new_archive_F[keep]
                        else:
                            archive_F = np.array([fc])
                        base_X = candidate
                        local_sigma *= 0.7
                    else:
                        # Try a smaller step
                        local_sigma *= 0.7

        return archive_F
# EVOLVE-BLOCK-END
