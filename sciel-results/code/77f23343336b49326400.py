"""Multi-objective optimizer for RE34 (3 objectives, 5 variables)."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """DE-MOPSO: differential-evolution particle swarm with grid archive."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        DEADLINE = 315.0

        dim = self.dim
        budget = self.budget
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        span = hi - lo
        mid = (lo + hi) / 2.0

        rng = np.random

        # --- Grid-based archive ---
        N_GRID = 8  # cells per objective dimension
        MAX_ARCHIVE = 200

        def grid_key(f):
            """Map objective vector to grid cell index."""
            keys = []
            for k in range(self.n_obj):
                fmin = np.min(archive_F[:, k]) if archive_F else 0.0
                fmax = np.max(archive_F[:, k]) if archive_F else 1.0
                rng_k = fmax - fmin if fmax > fmin else 1.0
                idx = int(np.clip((f[k] - fmin) / rng_k, 0, 1.0) * N_GRID)
                idx = min(idx, N_GRID - 1)
                keys.append(idx)
            return tuple(keys)

        archive_X = []
        archive_F = []
        grid = {}  # grid_key -> index into archive

        def dominates(a, b):
            """True if a dominates b (a <= b in all, a < b in at least one)."""
            return np.all(a <= b) and np.any(a < b)

        def crowding_distance(idx):
            """Approximate crowding distance for archive point idx."""
            if len(archive_F) <= 2:
                return 1.0
            f = archive_F[idx]
            total = 0.0
            for k in range(self.n_obj):
                # Find nearest neighbours in this objective
                dists = np.abs(archive_F[:, k] - f[k])
                dists[idx] = np.inf
                # Two nearest
                sorted_d = np.sort(dists)
                total += sorted_d[0] + sorted_d[1]
            return total

        def update_archive(x, f):
            nonlocal archive_X, archive_F, grid
            # Check if dominated by existing archive
            for i, af in enumerate(archive_F):
                if dominates(af, f):
                    return
            # Remove dominated points
            new_X = []
            new_F = []
            new_grid = {}
            for i, (ax, af) in enumerate(zip(archive_X, archive_F)):
                if not dominates(f, af):
                    new_X.append(ax)
                    new_F.append(af)
                    gk = grid_key(af)
                    if gk not in new_grid:
                        new_grid[gk] = len(new_F) - 1
            # Check if this point is in an occupied grid cell
            gk = grid_key(f)
            if gk in new_grid:
                # Keep the one with better crowding distance
                existing_idx = new_grid[gk]
                cd_new = crowding_distance(len(new_F))  # approximate
                cd_old = crowding_distance(existing_idx)
                if cd_new > cd_old:
                    # Replace existing
                    new_X[existing_idx] = x.copy()
                    new_F[existing_idx] = f.copy()
                else:
                    return
            else:
                new_X.append(x.copy())
                new_F.append(f.copy())
                new_grid[gk] = len(new_F) - 1

            # Enforce max archive size
            if len(new_F) > MAX_ARCHIVE:
                # Remove point with lowest crowding distance
                cds = np.array([crowding_distance(i) for i in range(len(new_F))])
                remove_idx = int(np.argmin(cds))
                removed_f = new_F[remove_idx]
                rgk = grid_key(removed_f)
                new_X.pop(remove_idx)
                new_F.pop(remove_idx)
                # Rebuild grid
                new_grid = {}
                for i, af in enumerate(new_F):
                    gk2 = grid_key(af)
                    if gk2 not in new_grid:
                        new_grid[gk2] = i

            archive_X = new_X
            archive_F = new_F
            grid = new_grid

        def pick_global_best():
            """Pick a global best from archive using grid-based selection."""
            if not archive_F:
                return None
            if len(archive_F) == 1:
                return 0
            # Pick random grid cell that has a point, then pick from that cell
            occupied = list(grid.keys())
            if not occupied:
                return rng.integers(len(archive_F))
            gk = occupied[rng.integers(len(occupied))]
            return grid[gk]

        # --- Initialization ---
        pop_size = min(30, max(12, budget // 7))

        X = np.zeros((pop_size, dim))
        V = np.zeros((pop_size, dim))
        F = np.zeros((pop_size, self.n_obj))
        pbest_X = np.zeros((pop_size, dim))
        pbest_F = np.zeros((pop_size, self.n_obj))

        for i in range(pop_size):
            X[i] = lo + span * rng.random(dim)

        for i in range(pop_size):
            F[i] = problem(X[i])
            pbest_X[i] = X[i].copy()
            pbest_F[i] = F[i].copy()
            update_archive(X[i], F[i])

        used = pop_size

        # --- PSO parameters ---
        w_max = 0.9
        w_min = 0.4
        c1 = 1.5
        c2 = 1.5
        v_max_frac = 0.5  # max velocity as fraction of span

        # --- Main loop ---
        gen = 0
        while used < budget:
            if time.time() - start_time > DEADLINE:
                break

            gen += 1
            # Adaptive inertia
            progress = min(1.0, used / budget)
            w = w_max - (w_max - w_min) * progress

            for i in range(pop_size):
                if used >= budget:
                    break
                if time.time() - start_time > DEADLINE:
                    break

                # Pick global best
                gb_idx = pick_global_best()
                if gb_idx is None or not archive_X:
                    gb_X = mid.copy()
                else:
                    gb_X = archive_X[gb_idx]

                r1 = rng.random(dim)
                r2 = rng.random(dim)

                # Velocity update
                V[i] = (w * V[i]
                        + c1 * r1 * (pbest_X[i] - X[i])
                        + c2 * r2 * (gb_X - X[i]))

                # Clamp velocity
                V[i] = np.clip(V[i], -v_max_frac * span, v_max_frac * span)

                # Position update
                X[i] = X[i] + V[i]
                X[i] = np.clip(X[i], lo, hi)

                # Evaluate
                f_new = np.array(problem(X[i]))
                used += 1

                # Update personal best
                if dominates(f_new, pbest_F[i]):
                    pbest_X[i] = X[i].copy()
                    pbest_F[i] = f_new.copy()

                # Update archive
                update_archive(X[i], f_new)

            # Periodic diversity injection (every 5 generations)
            if gen % 5 == 0 and used < budget:
                if time.time() - start_time > DEADLINE:
                    break
                # Replace worst particle
                sums = np.sum(pbest_F, axis=1)
                worst_i = int(np.argmax(sums))
                X[worst_i] = lo + span * rng.random(dim)
                V[worst_i] = np.zeros(dim)
                f_new = np.array(problem(X[worst_i]))
                used += 1
                pbest_X[worst_i] = X[worst_i].copy()
                pbest_F[worst_i] = f_new.copy()
                update_archive(X[worst_i], f_new)

        # --- Polish phase: local search around best archive points ---
        if used < budget and archive_X:
            if time.time() - start_time > DEADLINE:
                return X

            # Sort by sum of objectives, take top
            sums = np.array([np.sum(af) for af in archive_F])
            top_n = min(8, len(archive_X))
            top_idx = np.argsort(sums)[:top_n]

            for idx in top_idx:
                if used >= budget:
                    break
                if time.time() - start_time > DEADLINE:
                    break

                center = archive_X[idx].copy()
                center_f = archive_F[idx].copy()

                # Multi-scale local search
                for scale in [0.03, 0.08, 0.15]:
                    if used >= budget:
                        break
                    if time.time() - start_time > DEADLINE:
                        break
                    n_local = min(3, budget - used)
                    for _ in range(n_local):
                        if time.time() - start_time > DEADLINE:
                            break
                        perturbed = center + rng.normal(0, scale * span, dim)
                        perturbed = np.clip(perturbed, lo, hi)
                        f_pert = np.array(problem(perturbed))
                        used += 1
                        if np.any(f_pert < center_f) and not np.all(center_f <= f_pert):
                            update_archive(perturbed, f_pert)
                            center = perturbed
                            center_f = f_pert

        return X
# EVOLVE-BLOCK-END
