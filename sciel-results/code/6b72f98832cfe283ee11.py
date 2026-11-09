"""Multi-objective optimizer for RE34 using iterative DE search with Pareto archive."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Iterative differential-evolution search with Pareto archive."""

    def __init__(self, budget, dim, n_obj):
        self.budget, self.dim, self.n_obj = budget, dim, n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = 310.0  # Safety margin before 330s kill

        dim = self.dim
        n_obj = self.n_obj
        budget = self.budget

        lo = np.array([b[0] for b in bounds], dtype=float)
        hi = np.array([b[1] for b in bounds], dtype=float)

        rng = np.random.default_rng(0)

        # ---------- Initialization: structured diversity ----------
        # Use a mix of corners, midpoints, and LHS for good initial coverage
        pop_size = 40
        
        # Key structural points: corners of the hypercube (subset) + center + face centers
        init_points = []
        # 8 most spread-out corners (Hamming distance pattern)
        corner_indices = [0, 1, 2, 4, 8, 16, 3, 12]
        for ci in corner_indices:
            x = np.array([hi[j] if (ci >> j) & 1 else lo[j] for j in range(dim)])
            init_points.append(x)
        
        # Center point
        init_points.append((lo + hi) / 2.0)
        
        # Face centers (midpoint of each edge direction)
        for d in range(dim):
            x = (lo + hi) / 2.0
            x[d] = lo[d]
            init_points.append(x.copy())
            x[d] = hi[d]
            init_points.append(x.copy())
        
        # Fill remaining with LHS
        n_remaining = pop_size - len(init_points)
        if n_remaining > 0:
            # Latin Hypercube: stratified random
            for d in range(dim):
                strata = np.linspace(lo[d], hi[d], n_remaining + 1)
                offsets = (np.arange(n_remaining) + rng.random(n_remaining)) / n_remaining
                vals = strata[:-1] + offsets * (strata[1] - strata[0])
                if d == 0:
                    lhs_points = vals[:, np.newaxis]
                else:
                    lhs_points = np.hstack([lhs_points, vals[:, np.newaxis]])
            # Shuffle each dimension independently
            for d in range(dim):
                idx = rng.permutation(n_remaining)
                lhs_points[:, d] = lhs_points[idx, d]
            init_points.extend(lhs_points.tolist())

        X = np.array(init_points[:pop_size])
        F = np.array([problem(x) for x in X])
        used = pop_size

        # ---------- Pareto archive ----------
        archive_X = []
        archive_F = []

        def dominates(a, b):
            """Return True if a dominates b (a <= b in all, a < b in at least one)."""
            return np.all(a <= b) and np.any(a < b)

        def is_dominated(f, archive_F):
            """Vectorized domination check against archive."""
            if len(archive_F) == 0:
                return False
            af_arr = np.array(archive_F)
            return np.any(np.all(af_arr <= f, axis=1) & np.any(af_arr < f, axis=1))

        def crowding_distance(archive_F):
            """Compute crowding distance for each member of the archive."""
            n = len(archive_F)
            if n <= 2:
                return np.full(n, np.inf)
            cd = np.zeros(n)
            af_arr = np.array(archive_F)
            for m in range(n_obj):
                obj = af_arr[:, m]
                order = np.argsort(obj)
                cd[order[0]] = np.inf
                cd[order[-1]] = np.inf
                span = obj[order[-1]] - obj[order[0]]
                if span > 0:
                    for i in range(1, n - 1):
                        cd[order[i]] += (obj[order[i + 1]] - obj[order[i - 1]]) / span
            return cd

        def update_archive(x, f, archive_X, archive_F):
            """Add x/f to archive if non-dominated, removing dominated members."""
            if is_dominated(f, archive_F):
                return archive_X, archive_F
            # Check for exact duplicate
            if len(archive_F) > 0:
                af_arr = np.array(archive_F)
                if np.any(np.allclose(f, af_arr, atol=1e-12, axis=1)):
                    return archive_X, archive_F
            new_archive_X = []
            new_archive_F = []
            for ax, af in zip(archive_X, archive_F):
                if not dominates(f, af):
                    new_archive_X.append(ax)
                    new_archive_F.append(af)
            new_archive_X.append(x.copy())
            new_archive_F.append(f.copy())
            return new_archive_X, new_archive_F

        # Add initial population to archive
        for i in range(len(X)):
            archive_X, archive_F = update_archive(X[i], F[i], archive_X, archive_F)

        # ---------- Main search loop ----------
        F_factor = 0.6
        CR = 0.85
        pop_size = len(X)

        while used < budget:
            if time.time() - start_time > deadline:
                break

            # Select parent from archive using crowding distance
            if len(archive_X) > 1:
                cd = crowding_distance(archive_F)
                # Tournament selection: pick 2 random, choose one with higher cd
                i1 = rng.integers(len(archive_X))
                i2 = rng.integers(len(archive_X))
                parent_idx = i1 if cd[i1] >= cd[i2] else i2
                parent = archive_X[parent_idx]
            elif len(archive_X) == 1:
                parent = archive_X[0]
            else:
                parent = X[rng.integers(pop_size)]

            # DE mutation: select 3 distinct random indices from population
            idx1, idx2, idx3 = rng.choice(pop_size, size=3, replace=False)

            # Create trial vector
            trial = X[idx1] + F_factor * (X[idx2] - X[idx3])
            # Binary crossover with parent
            mask = rng.random(dim) < CR
            trial = np.where(mask, trial, parent)
            trial = np.clip(trial, lo, hi)

            ft = problem(trial)
            used += 1

            # Update archive
            archive_X, archive_F = update_archive(trial, ft, archive_X, archive_F)

            # Update population: replace worst member only if trial is non-dominated by it
            worst_idx = np.argmax(np.sum(F, axis=1))
            if not dominates(F[worst_idx], ft):
                X[worst_idx] = trial
                F[worst_idx] = ft

        # ---------- Final polish: local refinement on archive members ----------
        if time.time() - start_time < deadline - 5:
            # For each archive member, do a small local search
            step_size = 0.1 * (hi - lo)
            for i in range(len(archive_X)):
                if time.time() - start_time > deadline - 3:
                    break
                base_x = archive_X[i].copy()
                base_f = archive_F[i].copy()
                # Try small perturbations in each dimension
                for d in range(dim):
                    for direction in [-1, 1]:
                        trial_x = base_x.copy()
                        trial_x[d] += direction * step_size[d]
                        trial_x = np.clip(trial_x, lo, hi)
                        if np.allclose(trial_x, base_x, atol=1e-10):
                            continue
                        trial_f = problem(trial_x)
                        used += 1
                        if used >= budget:
                            break
                        # Accept if it improves any objective without worsening others too much
                        if np.all(trial_f <= base_f) and np.any(trial_f < base_f):
                            base_x = trial_x
                            base_f = trial_f.copy()
                        elif used >= budget:
                            break
                    if used >= budget:
                        break
                # Update archive with potentially improved point
                if not np.allclose(base_f, archive_F[i], atol=1e-10):
                    archive_X, archive_F = update_archive(base_x, base_f, archive_X, archive_F)

        return X
# EVOLVE-BLOCK-END
