"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


class Optimizer:
    """Multi-objective optimizer using adaptive Tchebycheff scalarization with L-BFGS-B."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280.0  # safety margin before 330s limit

        lo = np.array([b[0] for b in bounds], dtype=np.float64)
        hi = np.array([b[1] for b in bounds], dtype=np.float64)
        dim = self.dim
        n_obj = self.n_obj
        budget = self.budget

        # Reference point for hypervolume (from task)
        ref_point = np.array([1705.0, 12.0, 0.3])

        # Evaluation counter
        eval_count = 0

        def count_eval():
            nonlocal eval_count
            eval_count += 1

        # Archive of non-dominated points
        archive_X = []
        archive_F = []

        def dominates(f1, f2):
            """Check if f1 dominates f2 (all <= and at least one <)."""
            return np.all(f1 <= f2) and np.any(f1 < f2)

        def is_dominated(f, archive_F):
            """Check if f is dominated by any point in archive_F."""
            if len(archive_F) == 0:
                return False
            for af in archive_F:
                if dominates(af, f):
                    return True
            return False

        def add_to_archive(x, f):
            """Add point to archive if not dominated, remove dominated points."""
            nonlocal archive_X, archive_F
            if is_dominated(f, archive_F):
                return False
            # Remove points dominated by f
            new_archive_X = []
            new_archive_F = []
            for i in range(len(archive_F)):
                if not dominates(f, archive_F[i]):
                    new_archive_X.append(archive_X[i])
                    new_archive_F.append(archive_F[i])
            new_archive_X.append(x)
            new_archive_F.append(f)
            archive_X = new_archive_X
            archive_F = new_archive_F
            return True

        def tchebycheff_obj(x, weights, zmin, zmax):
            """Tchebycheff scalarization objective."""
            f = problem(x)
            count_eval()
            # Normalize objectives
            zrange = zmax - zmin
            zrange[zrange < 1e-10] = 1.0
            normalized = (f - zmin) / zrange
            # Tchebycheff: max_i weights[i] * normalized[i]
            return np.max(weights * normalized), f

        # Initial random sampling
        rng = np.random.default_rng(42)
        n_init = min(30, budget // 5)
        for _ in range(n_init):
            if time.time() > deadline:
                break
            x = lo + (hi - lo) * rng.random(dim)
            f = problem(x)
            count_eval()
            add_to_archive(x.copy(), f.copy())

        # Get current bounds of objectives
        if len(archive_F) > 0:
            zmin = np.min(archive_F, axis=0)
            zmax = np.max(archive_F, axis=0)
        else:
            zmin = np.zeros(n_obj)
            zmax = np.ones(n_obj)

        # Generate weight vectors using a grid in the simplex
        def generate_weights(n_weights):
            """Generate weight vectors on the simplex."""
            weights_list = []
            # Uniform points on the simplex
            for _ in range(n_weights):
                w = rng.random(n_obj)
                w = w / w.sum()
                weights_list.append(w)
            # Also add axis-aligned weights
            for i in range(n_obj):
                w = np.zeros(n_obj)
                w[i] = 1.0
                weights_list.append(w)
            return weights_list

        # Main optimization loop
        weights = generate_weights(10)
        weight_idx = 0

        while eval_count < budget and time.time() < deadline:
            # Pick a weight vector
            w = weights[weight_idx % len(weights)]
            weight_idx += 1

            # Find best point in archive for this weight (closest to utopia)
            if len(archive_F) > 0:
                # Compute Tchebycheff distance for each archive point
                zrange = zmax - zmin
                zrange[zrange < 1e-10] = 1.0
                normalized = (np.array(archive_F) - zmin) / zrange
                dists = np.max(w * normalized, axis=1)
                best_idx = np.argmin(dists)
                x0 = archive_X[best_idx].copy()
            else:
                x0 = (lo + hi) / 2.0

            # Also try starting from a random point for diversity
            if rng.random() < 0.3:
                x0 = lo + (hi - lo) * rng.random(dim)

            # Run L-BFGS-B on Tchebycheff objective
            def objective(x):
                if time.time() > deadline:
                    return 1e10
                val, _ = tchebycheff_obj(x, w, zmin, zmax)
                return val

            try:
                res = minimize(
                    objective,
                    x0,
                    method='L-BFGS-B',
                    bounds=list(zip(lo, hi)),
                    options={'maxiter': 50, 'ftol': 1e-10, 'gtol': 1e-8}
                )
            except Exception:
                continue

            if res.success or res.fun < 1e9:
                x_opt = np.clip(res.x, lo, hi)
                f_opt = problem(x_opt)
                count_eval()

                # Update archive
                added = add_to_archive(x_opt.copy(), f_opt.copy())
                if added:
                    # Update zmin, zmax
                    zmin = np.minimum(zmin, f_opt)
                    zmax = np.maximum(zmax, f_opt)

                # If we found a new point, we might want to refine weights
                # by adding more weights near the new point
                if added and len(weights) < 30:
                    # Add a weight vector that emphasizes the direction of the new point
                    new_w = w + 0.1 * rng.standard_normal(n_obj)
                    new_w = np.abs(new_w)
                    new_w = new_w / new_w.sum()
                    weights.append(new_w)

            # Periodically regenerate weights based on current front
            if weight_idx % 15 == 0 and len(archive_F) > 2:
                weights = generate_weights(10)
                weight_idx = 0

        return archive_X
# EVOLVE-BLOCK-END
