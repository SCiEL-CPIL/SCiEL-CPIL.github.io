"""Multi-objective optimizer on WFG4 using efficient archive-based approach."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Efficient multi-objective optimizer for WFG4."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 310.0  # Safety margin for 330s limit

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        rng = np.random.default_rng(42)

        MAX_ARCHIVE = 120
        INIT_POP = 40

        # Initialize with Latin Hypercube sampling for better spread
        def latin_hypercube(n, dim):
            """Generate n points using Latin Hypercube sampling in [0,1]^dim."""
            points = np.zeros((n, dim))
            for d in range(dim):
                # For each dimension, divide [0,1] into n intervals and pick one point per interval
                intervals = np.linspace(0, 1, n + 1)
                points[:, d] = (intervals[:-1] + rng.random(n) * (intervals[1] - intervals[0]))
            # Shuffle rows to decorrelate dimensions
            perm = rng.permutation(n)
            return points[perm]

        lhc = latin_hypercube(INIT_POP, self.dim)
        X_init = lo + (hi - lo) * lhc
        F_init = np.array([problem(x) for x in X_init])
        evaluations = INIT_POP

        # Build initial archive using efficient non-dominated sort
        archive_X = []
        archive_F = []

        def is_dominated(f_new, archive_F):
            """Check if f_new is dominated by any point in archive."""
            if not archive_F:
                return False
            # Vectorized check
            leq = np.all(archive_F <= f_new, axis=1)
            lt = np.any(archive_F < f_new, axis=1)
            return np.any(leq & lt)

        def remove_dominated(new_f, archive_X, archive_F):
            """Remove points in archive dominated by new_f. Returns modified lists."""
            if not archive_F:
                return archive_X, archive_F
            leq = np.all(new_f <= archive_F, axis=1)
            lt = np.any(new_f < archive_F, axis=1)
            dominated_mask = leq & lt
            if not np.any(dominated_mask):
                return archive_X, archive_F
            new_X = [archive_X[i] for i in range(len(archive_X)) if not dominated_mask[i]]
            new_F = [archive_F[i] for i in range(len(archive_F)) if not dominated_mask[i]]
            return new_X, new_F

        def crowding_distance(F_arr):
            """Compute crowding distance for all points in F_arr."""
            n = len(F_arr)
            if n <= 2:
                return np.full(n, float('inf'))
            
            distances = np.zeros(n)
            for m in range(self.n_obj):
                values = F_arr[:, m]
                min_val = np.min(values)
                max_val = np.max(values)
                if max_val - min_val < 1e-12:
                    continue
                sorted_idx = np.argsort(values)
                distances[sorted_idx[0]] = float('inf')
                distances[sorted_idx[-1]] = float('inf')
                for k in range(1, len(sorted_idx) - 1):
                    idx = sorted_idx[k]
                    prev_idx = sorted_idx[k - 1]
                    next_idx = sorted_idx[k + 1]
                    distances[idx] += (F_arr[next_idx, m] - F_arr[prev_idx, m]) / (max_val - min_val)
            return distances

        def prune_archive(archive_X, archive_F, max_size):
            """Prune archive to max_size using crowding distance."""
            if len(archive_X) <= max_size:
                return archive_X, archive_F
            
            F_arr = np.array(archive_F)
            cd = crowding_distance(F_arr)
            # Sort by crowding distance descending
            sorted_indices = np.argsort(-cd)
            kept_indices = sorted_indices[:max_size]
            kept_X = [archive_X[i] for i in kept_indices]
            kept_F = [archive_F[i] for i in kept_indices]
            return kept_X, kept_F

        def differential_evolution_select(archive_X, rng):
            """Select parents for Differential Evolution."""
            n = len(archive_X)
            if n < 3:
                # If not enough points, fall back to random or single point perturbation
                if n == 1:
                    return archive_X[0], None, None
                elif n == 2:
                    i = rng.integers(2)
                    j = 1 - i
                    return archive_X[i], archive_X[j], None
                else:
                    return None, None, None
            
            # Select 3 distinct random indices
            indices = rng.choice(n, size=3, replace=False)
            r1 = indices[0]
            r2 = indices[1]
            r3 = indices[2]
            
            return archive_X[r1], archive_X[r2], archive_X[r3]

        # Populate initial archive
        for i in range(INIT_POP):
            if not is_dominated(F_init[i], archive_F):
                archive_X, archive_F = remove_dominated(F_init[i], archive_X, archive_F)
                archive_X.append(X_init[i])
                archive_F.append(F_init[i])
        
        archive_X, archive_F = prune_archive(archive_X, archive_F, MAX_ARCHIVE)

        # Main Loop
        while evaluations < self.budget:
            if time.time() > deadline:
                break
            
            # Adaptive parameters based on progress
            progress = evaluations / self.budget
            
            # Differential Evolution Variation
            if len(archive_X) < 3:
                if len(archive_X) == 0:
                    x_off = lo + (hi - lo) * rng.random(self.dim)
                elif len(archive_X) == 1:
                    x1 = archive_X[0]
                    sigma = 0.2
                    x_off = x1 + sigma * (hi - lo) * rng.standard_normal(self.dim)
                else:
                    # Use available points for a simple blend
                    i = rng.integers(len(archive_X))
                    j = (i + 1) % len(archive_X)
                    x1 = archive_X[i]
                    x2 = archive_X[j]
                    x_off = x1 + 0.5 * (x2 - x1)
                    sigma = 0.1
                    x_off = x_off + sigma * (hi - lo) * rng.standard_normal(self.dim)
            else:
                x_r1, x_r2, x_r3 = differential_evolution_select(archive_X, rng)
                
                # F factor: start high for exploration, decrease slightly
                F = 0.8 - 0.2 * progress
                
                # DE/rand/1
                x_off = x_r1 + F * (x_r2 - x_r3)
                
                # Add small Gaussian noise to escape local optima
                sigma = 0.05
                x_off = x_off + sigma * (hi - lo) * rng.standard_normal(self.dim)
            
            x_off = np.clip(x_off, lo, hi)
            
            # Evaluate
            f_off = problem(x_off)
            evaluations += 1
            
            # Update Archive
            if not is_dominated(f_off, archive_F):
                archive_X, archive_F = remove_dominated(f_off, archive_X, archive_F)
                archive_X.append(x_off)
                archive_F.append(f_off)
                
                if len(archive_X) > MAX_ARCHIVE:
                    archive_X, archive_F = prune_archive(archive_X, archive_F, MAX_ARCHIVE)

        return archive_X
# EVOLVE-BLOCK-END
