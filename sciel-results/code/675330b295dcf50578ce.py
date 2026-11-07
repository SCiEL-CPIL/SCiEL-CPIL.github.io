"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import math


class Optimizer:
    """
    Hybrid Multi-Objective Optimizer:
    1. Deterministic Tchebycheff grid to establish initial Pareto front.
    2. Stochastic local search (Simulated Annealing) to refine points.
    """

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 290  # 300s limit, 10s safety margin

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        rng = np.random.default_rng(42)

        # Phase 1: Deterministic Tchebycheff grid
        # For 3 objectives, we use a simplex grid of weight vectors
        n_weights = 50
        weights = []
        # Generate points on the 2-simplex (sum to 1, all >= 0)
        # We use a stratified approach for better coverage
        for i in range(n_weights):
            # Use quasi-random numbers for better spread
            u1 = (i + 0.5) / n_weights
            # Use a golden ratio based spacing for 2D
            # Actually, let's just use a simple grid on the simplex
            # For 3 objectives, we can parameterize by two angles or two coordinates
            pass

        # Simpler: use a fixed set of weight vectors that cover the simplex
        # We'll generate 50 points on the 2-simplex
        weight_points = []
        # Add the 3 vertices
        weight_points.append([1.0, 0.0, 0.0])
        weight_points.append([0.0, 1.0, 0.0])
        weight_points.append([0.0, 0.0, 1.0])
        # Add midpoints of edges
        weight_points.append([0.5, 0.5, 0.0])
        weight_points.append([0.5, 0.0, 0.5])
        weight_points.append([0.0, 0.5, 0.5])
        # Add center
        weight_points.append([1/3, 1/3, 1/3])
        # Fill remaining with stratified sampling
        remaining = n_weights - len(weight_points)
        for _ in range(remaining):
            u = rng.random(3)
            u /= u.sum()
            weight_points.append(u)

        weight_points = np.array(weight_points)

        # Evaluate each weight point using Tchebycheff scalarization
        # Find initial points by minimizing max_i w_i * f_i(x)
        # We do a quick local search for each weight vector
        X_archive = []
        F_archive = []
        used = 0

        # First, do a broad random sampling to get good starting points
        n_initial = min(50, self.budget // 4)
        X_init = lo + (hi - lo) * rng.random((n_initial, self.dim))
        F_init = np.array([problem(x) for x in X_init])
        used += n_initial

        # For each weight vector, find the best point in F_init by Tchebycheff measure
        # Then do local search to refine
        for w in weight_points:
            if used >= self.budget or time.time() > deadline:
                break

            # Find best point in current archive for this weight
            if len(F_archive) > 0:
                F_arr = np.array(F_archive)
                # Tchebycheff measure: max_i w_i * f_i
                # Normalize objectives to [0,1] range for fair weighting
                f_min = F_arr.min(axis=0)
                f_max = F_arr.max(axis=0)
                f_range = f_max - f_min
                f_range[f_range < 1e-10] = 1e-10
                F_norm = (F_arr - f_min) / f_range
                tcheb = np.max(w * F_norm, axis=1)
                best_idx = np.argmin(tcheb)
                x_start = X_archive[best_idx].copy()
            else:
                x_start = X_init[0].copy()

            # Simulated Annealing to refine
            x_best = x_start.copy()
            f_best = problem(x_start)
            used += 1

            T_init = 0.1 * (hi - lo).max()
            T_current = T_init
            cooling = 0.95
            n_sa_steps = min(20, self.budget - used)

            for step in range(n_sa_steps):
                if used >= self.budget or time.time() > deadline:
                    break

                # Generate neighbor
                x_new = x_best + T_current * (hi - lo) * rng.standard_normal(self.dim)
                x_new = np.clip(x_new, lo, hi)
                f_new = problem(x_new)
                used += 1

                # Accept if better or with probability exp(-delta/T)
                # Use Tchebycheff measure for comparison
                f_min_cur = np.minimum(f_best, f_new)
                f_max_cur = np.maximum(f_best, f_new)
                f_range_cur = f_max_cur - f_min_cur
                f_range_cur[f_range_cur < 1e-10] = 1e-10
                f_best_norm = (f_best - f_min_cur) / f_range_cur
                f_new_norm = (f_new - f_min_cur) / f_range_cur
                tcheb_best = np.max(w * f_best_norm)
                tcheb_new = np.max(w * f_new_norm)
                delta = tcheb_new - tcheb_best

                if delta < 0 or rng.random() < math.exp(-delta / max(T_current, 1e-10)):
                    x_best = x_new
                    f_best = f_new

                T_current *= cooling

            # Add to archive if not dominated
            if len(F_archive) == 0 or not self._is_dominated(f_best, np.array(F_archive)):
                X_archive.append(x_best.copy())
                F_archive.append(f_best.copy())

        # Phase 2: Use remaining budget for stochastic local search on random points
        while used < self.budget and time.time() < deadline:
            # Pick a random starting point from archive or new random
            if len(X_archive) > 0 and rng.random() < 0.5:
                idx = rng.integers(len(X_archive))
                x = X_archive[idx].copy()
            else:
                x = lo + (hi - lo) * rng.random(self.dim)

            # Local search with random restarts
            n_ls_steps = min(10, self.budget - used)
            f_curr = problem(x)
            used += 1

            for _ in range(n_ls_steps):
                if used >= self.budget or time.time() > deadline:
                    break

                x_new = x + 0.1 * (hi - lo) * rng.standard_normal(self.dim)
                x_new = np.clip(x_new, lo, hi)
                f_new = problem(x_new)
                used += 1

                # Accept if not dominated by current best in archive
                if len(F_archive) == 0 or not self._is_dominated(f_new, np.array(F_archive)):
                    x = x_new
                    f_curr = f_new

            # Add to archive if not dominated
            if len(F_archive) == 0 or not self._is_dominated(f_curr, np.array(F_archive)):
                X_archive.append(x.copy())
                F_archive.append(f_curr.copy())

        # Return the archive (the return value is ignored, but we return something)
        if len(X_archive) > 0:
            return np.array(X_archive)
        return np.array([lo])

    def _is_dominated(self, f, F_archive):
        """Check if f is dominated by any point in F_archive."""
        for f_other in F_archive:
            if np.all(f_other <= f) and np.any(f_other < f):
                return True
        return False
# EVOLVE-BLOCK-END
