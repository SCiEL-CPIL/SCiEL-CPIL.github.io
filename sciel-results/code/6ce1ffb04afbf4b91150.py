"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """DE/rand/1 + crowding-distance-based archive for multi-objective optimization."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        t_start = time.time()
        deadline = t_start + 280.0  # safety margin before 300s limit

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        rng = np.random.default_rng(42)

        # Initial population
        init_pop = min(self.budget, 40)
        X = lo + (hi - lo) * rng.random((init_pop, self.dim))
        F = np.array([problem(x) for x in X])
        used = init_pop

        # Archive: keep non-dominated points with crowding distance
        archive_X = []
        archive_F = []

        def _dominates(a, b):
            """True if a dominates b (a <= b in all, a < b in at least one)."""
            return np.all(a <= b) and np.any(a < b)

        def _non_dominated_indices(Fs):
            """Return indices of non-dominated points in list of vectors."""
            n = len(Fs)
            nd = list(range(n))
            i = 0
            while i < len(nd):
                j = i + 1
                while j < len(nd):
                    if _dominates(Fs[nd[i]], Fs[nd[j]]):
                        nd.pop(j)
                    elif _dominates(Fs[nd[j]], Fs[nd[i]]):
                        nd.pop(i)
                        break
                    else:
                        j += 1
                i += 1
            return nd

        def _crowding_distance(Fs, idxs):
            """Compute crowding distance for non-dominated points."""
            if len(idxs) <= 2:
                return np.full(len(idxs), np.inf)
            n = len(idxs)
            dist = np.zeros(n)
            for m in range(self.n_obj):
                vals = np.array([Fs[i][m] for i in idxs])
                order = np.argsort(vals)
                dist[order[0]] = np.inf
                dist[order[-1]] = np.inf
                span = vals[order[-1]] - vals[order[0]]
                if span > 1e-15:
                    for k in range(1, n - 1):
                        dist[order[k]] += (vals[order[k + 1]] - vals[order[k - 1]]) / span
            return dist

        # Update archive with new points
        def _update_archive(new_X, new_F):
            nonlocal archive_X, archive_F
            for x, f in zip(new_X, new_F):
                # Check if dominated by existing archive
                dominated = False
                for af in archive_F:
                    if _dominates(af, f):
                        dominated = True
                        break
                if dominated:
                    continue
                # Remove archive points dominated by new point
                new_archive_X = []
                new_archive_F = []
                for ax, af in zip(archive_X, archive_F):
                    if not _dominates(f, af):
                        new_archive_X.append(ax)
                        new_archive_F.append(af)
                new_archive_X.append(x)
                new_archive_F.append(f)
                archive_X, archive_F = new_archive_X, new_archive_F

                # If archive too large, trim by crowding distance
                max_archive = 100
                if len(archive_X) > max_archive:
                    nd_idx = _non_dominated_indices(archive_F)
                    if len(nd_idx) > max_archive:
                        cd = _crowding_distance(archive_F, nd_idx)
                        # Remove point with smallest crowding distance
                        min_cd_idx = nd_idx[np.argmin(cd)]
                        archive_X.pop(min_cd_idx)
                        archive_F.pop(min_cd_idx)

        _update_archive(X, F)

        # Differential Evolution parameters
        CR = 0.9
        F_mut = 0.5

        while used < self.budget:
            if time.time() > deadline:
                break

            # Select base and two distinct parents from archive
            n_arch = len(archive_X)
            if n_arch < 3:
                # Not enough diversity; just random perturbation
                base_idx = rng.integers(n_arch)
                v = archive_X[base_idx] + 0.1 * (hi - lo) * rng.standard_normal(self.dim)
                v = np.clip(v, lo, hi)
                fv = problem(v)
                used += 1
                _update_archive([v], [fv])
                continue

            i1 = rng.integers(n_arch)
            i2 = rng.integers(n_arch)
            i3 = rng.integers(n_arch)
            # Ensure distinct
            if i1 == i2 or i1 == i3 or i2 == i3:
                i2 = (i2 + 1) % n_arch
                if i1 == i2 or i1 == i3 or i2 == i3:
                    i3 = (i3 + 1) % n_arch

            base = archive_X[i1]
            v = base + F_mut * (archive_X[i2] - archive_X[i3])
            v = np.clip(v, lo, hi)
            fv = problem(v)
            used += 1
            _update_archive([v], [fv])

        return archive_X
# EVOLVE-BLOCK-END
