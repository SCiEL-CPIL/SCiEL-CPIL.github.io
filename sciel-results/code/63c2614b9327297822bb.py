"""Multi-objective optimizer combining DE with decomposition-based selection for uniform Pareto front coverage."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-objective optimizer: DE-based search with decomposition-guided archive."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280  # safety margin before 300s kill

        lo = np.array([b[0] for b in bounds], dtype=np.float64)
        hi = np.array([b[1] for b in bounds], dtype=np.float64)
        rng = np.random.default_rng(0)

        dim = self.dim
        n_obj = self.n_obj
        budget = self.budget

        # Reference point for hypervolume
        ref = np.array([1705.0, 12.0, 0.3])

        # Initialize with stratified sampling for good spread
        n_init = min(30, max(10, budget // 7))
        # Use a grid-like initialization with jitter
        per_dim = max(2, int(round(n_init ** (1.0 / dim))))
        grids = [np.linspace(0, 1, per_dim) for _ in range(dim)]
        grid = np.array(np.meshgrid(*grids, indexing='ij'))
        grid = np.stack(grid, axis=-1).reshape(-1, dim)
        # Add jitter
        grid = grid + rng.random((len(grid), dim)) * 0.1
        grid = np.clip(grid, 0, 1)

        # If grid is too large, subsample
        if len(grid) > n_init:
            idx = rng.choice(len(grid), size=n_init, replace=False)
            grid = grid[idx]
        # If grid is too small, add random points
        while len(grid) < n_init:
            grid = np.vstack([grid, rng.random((1, dim))])

        X = lo + (hi - lo) * grid[:n_init]
        F = np.array([problem(x) for x in X])
        used = n_init

        # Archive: maintain a diverse set of non-dominated points
        archive_X = [X[i].copy() for i in range(n_init)]
        archive_F = [F[i].copy() for i in range(n_init)]

        def dominates(a, b):
            """Check if a dominates b (a <= b in all, a < b in at least one)."""
            return np.all(a <= b) and np.any(a < b)

        def is_non_dominated(point, front):
            """Check if point is non-dominated with respect to front."""
            for f in front:
                if dominates(f, point):
                    return False
            return True

        def remove_dominated(point, front):
            """Remove points from front that are dominated by point."""
            return [f for f in front if not dominates(point, f)]

        def update_archive(point, fval, front):
            """Update the Pareto front with a new point. Returns new front."""
            if not is_non_dominated(fval, front):
                return front
            new_front = remove_dominated(fval, front)
            new_front.append(fval)
            return new_front

        def rebuild_archive(archive_X, archive_F):
            """Rebuild archive to remove any dominated points."""
            new_x = []
            new_f = []
            for i in range(len(archive_X)):
                if is_non_dominated(archive_F[i], archive_F):
                    new_x.append(archive_X[i])
                    new_f.append(archive_F[i])
            return new_x, new_f

        # DE parameters
        CR = 0.9  # crossover rate
        F_mut = 0.8  # mutation scale factor
        pop_size = 20  # population size for DE

        # Generate reference directions for decomposition-based selection
        # For 3 objectives, use simplex-lattice style directions
        def generate_reference_directions(n_obj, n_dirs=50):
            """Generate reference directions on the simplex."""
            dirs = []
            # For 3 objectives, we can use a simple approach
            # Generate points on the simplex {x1+x2+x3=1, xi>=0}
            for i in range(n_dirs):
                # Random point on simplex using Dirichlet
                u = rng.dirichlet([1, 1, 1])
                dirs.append(u)
            dirs = np.array(dirs)
            return dirs

        ref_dirs = generate_reference_directions(n_obj, 50)

        def assign_to_nearest_direction(f, ref_dirs):
            """Assign a point to its nearest reference direction."""
            # Normalize f to sum to 1 (or handle zero sum)
            f_sum = np.sum(f)
            if f_sum < 1e-10:
                f_norm = np.ones(n_obj) / n_obj
            else:
                f_norm = f / f_sum
            # Find nearest direction
            dists = np.linalg.norm(ref_dirs - f_norm, axis=1)
            return np.argmin(dists)

        # Main optimization loop
        while used < budget:
            if time.time() > deadline:
                break

            # Build current population from archive (sample if too large)
            if len(archive_F) <= pop_size:
                pop_F = archive_F
                pop_X = archive_X
            else:
                # Use decomposition-based selection: sample from archive
                # with bias toward under-represented directions
                dir_assignments = [assign_to_nearest_direction(f, ref_dirs) for f in archive_F]
                dir_counts = np.bincount(dir_assignments, minlength=len(ref_dirs))
                
                # Points in under-represented directions get higher probability
                # Inverse count with smoothing
                smooth_counts = dir_counts + 1.0
                inv_counts = 1.0 / smooth_counts
                inv_counts /= np.sum(inv_counts)
                
                # Create per-point probabilities
                point_probs = np.array([inv_counts[da] for da in dir_assignments])
                point_probs /= np.sum(point_probs)
                
                # Sample population
                if len(archive_F) > pop_size:
                    idx = rng.choice(len(archive_F), size=pop_size, replace=False, p=point_probs)
                else:
                    idx = np.arange(len(archive_F))
                pop_F = [archive_F[i] for i in idx]
                pop_X = [archive_X[i] for i in idx]

            # Differential evolution step
            n_pop = len(pop_X)
            if n_pop < 3:
                # Too few points, just do random perturbation
                base = rng.choice(n_pop)
                v = pop_X[base] + F_mut * (hi - lo) * rng.standard_normal(dim)
                v = np.clip(v, lo, hi)
            else:
                # Standard DE/rand/1
                a = rng.randint(0, n_pop)
                while True:
                    b = rng.randint(0, n_pop)
                    if b != a:
                        break
                while True:
                    c = rng.randint(0, n_pop)
                    if c != a and c != b:
                        break
                v = pop_X[a] + F_mut * (pop_X[b] - pop_X[c])
                # Binomial crossover
                trial = v.copy()
                j_rand = rng.randint(0, dim)
                for j in range(dim):
                    if rng.random() <= CR or j == j_rand:
                        trial[j] = v[j]
                    else:
                        trial[j] = pop_X[a][j]
                trial = np.clip(trial, lo, hi)

            # Evaluate
            f_trial = problem(trial)
            used += 1

            # Update archive
            if is_non_dominated(f_trial, archive_F):
                # Add new point and remove dominated points
                new_archive_F = update_archive(trial, f_trial, archive_F)
                archive_F = new_archive_F
                archive_X.append(trial.copy())

                # Rebuild to ensure consistency
                archive_X, archive_F = rebuild_archive(archive_X, archive_F)

        return X
# EVOLVE-BLOCK-END
