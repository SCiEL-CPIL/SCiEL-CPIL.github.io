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

        def binary_tournament_select(archive_F, rng):
            """Select one parent using binary tournament on crowding distance."""
            n = len(archive_F)
            if n <= 1:
                return 0
            F_arr = np.array(archive_F)
            cd = crowding_distance(F_arr)
            i = rng.integers(n)
            j = rng.integers(n)
            if cd[i] >= cd[j]:
                return i
            else:
                return j

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
            
            # Explore-exploit balance
            if len(archive_X) < 2:
                # Perturb the single point
                x1 = archive_X[0]
                sigma = 0.15
                x_off = x1 + sigma * (hi - lo) * rng.standard_normal(self.dim)
            else:
                # Select two parents using binary tournament for diversity
                idx1 = binary_tournament_select(archive_F, rng)
                idx2 = binary_tournament_select(archive_F, rng)
                x1 = archive_X[idx1]
                x2 = archive_X[idx2]
                
                # SBX crossover
                eta = 20.0
                x_off = np.zeros(self.dim)
                for d in range(self.dim):
                    u = rng.random()
                    if u <= 0.5:
                        beta = (2 * u) ** (1.0 / (eta + 1))
                    else:
                        beta = (1 / (2 * (1 - u))) ** (1.0 / (eta + 1))
                    x_off[d] = 0.5 * ((1 + beta) * x1[d] + (1 - beta) * x2[d])
                
                # Mutation: Gaussian with adaptive sigma (larger early, smaller late)
                sigma = 0.10 * (1.0 - 0.5 * progress)
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
