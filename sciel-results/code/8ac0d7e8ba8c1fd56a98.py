"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize


class Optimizer:
    """
    Multi-objective optimizer for RE34 using adaptive Tchebycheff scalarization
    with multi-start L-BFGS-B local search initialized at the boundaries of the
    current Pareto front to fill hypervolume gaps.
    """

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        dim = self.dim
        n_obj = self.n_obj

        # Reference point for scoring (used to filter dominated points)
        ref_point = np.array([1705.0, 12.0, 0.3])

        # Archive of non-dominated points
        archive_X = []
        archive_F = []
        evaluations = 0

        def evaluate(x):
            nonlocal evaluations
            x_clipped = np.clip(x, lo, hi)
            f = np.array(problem(x_clipped))
            evaluations += 1
            return x_clipped, f

        def is_dominated(f, archive_F):
            """Check if f is dominated by any point in archive_F."""
            for af in archive_F:
                if np.all(af <= f) and np.any(af < f):
                    return True
            return False

        def is_dominating(f, archive_F):
            """Check if f dominates any point in archive_F."""
            for af in archive_F:
                if np.all(f <= af) and np.any(f < af):
                    return True
            return False

        def add_to_archive(x, f):
            """Add point to archive if non-dominated, remove dominated points."""
            if is_dominated(f, archive_F):
                return False
            # Remove points dominated by f
            new_archive_X = []
            new_archive_F = []
            for ax, af in zip(archive_X, archive_F):
                if not (np.all(f <= af) and np.any(f < af)):
                    new_archive_X.append(ax)
                    new_archive_F.append(af)
            new_archive_X.append(x)
            new_archive_F.append(f)
            archive_X.clear()
            archive_F.clear()
            archive_X.extend(new_archive_X)
            archive_F.extend(new_archive_F)
            return True

        def tchebycheff_objective(x, weights, ideal):
            """Tchebycheff scalarization objective."""
            x_clipped = np.clip(x, lo, hi)
            f = np.array(problem(x_clipped))
            # Normalize objectives
            f_norm = (f - ideal) / (self.range if self.range > 1e-10 else 1.0)
            return np.max(weights * f_norm)

        # Initialize with random points
        rng = np.random.default_rng(0)
        n_init = min(10, self.budget)
        for i in range(n_init):
            if evaluations >= self.budget:
                break
            x = lo + (hi - lo) * rng.random(dim)
            x_c, f = evaluate(x)
            add_to_archive(x_c, f)

        # Track ideal point and range for normalization
        self.ideal = np.min(archive_F, axis=0) if archive_F else np.zeros(n_obj)
        self.range = np.max(archive_F, axis=0) - self.ideal if archive_F else np.ones(n_obj)

        # Generate weight vectors using a grid
        n_weights = 5
        weight_vectors = []
        for i in range(n_weights):
            for j in range(n_weights):
                for k in range(n_weights):
                    w = np.array([i, j, k], dtype=float)
                    if np.sum(w) > 0:
                        w = w / np.sum(w)
                        weight_vectors.append(w)
        # Add extreme weights
        for i in range(n_obj):
            w = np.zeros(n_obj)
            w[i] = 1.0
            weight_vectors.append(w)

        # Local search using L-BFGS-B with Tchebycheff scalarization
        def local_search(start_x, weights, ideal):
            """Run L-BFGS-B minimization of Tchebycheff scalarization."""
            def objective(x):
                x_clipped = np.clip(x, lo, hi)
                f = np.array(problem(x_clipped))
                f_norm = (f - ideal) / (self.range if self.range > 1e-10 else 1.0)
                return np.max(weights * f_norm)

            bounds_list = list(zip(lo, hi))
            try:
                result = minimize(
                    objective,
                    start_x,
                    method='L-BFGS-B',
                    bounds=bounds_list,
                    options={'maxiter': 50}
                )
                return result.x
            except:
                return start_x

        # Iteratively fill gaps in the Pareto front
        while evaluations < self.budget:
            # Update ideal point
            if archive_F:
                self.ideal = np.min(archive_F, axis=0)
                self.range = np.max(archive_F, axis=0) - self.ideal
                self.range[self.range < 1e-10] = 1.0

            # For each weight vector, find the point on the Pareto front
            # that is closest to the weight direction, and start local search from there
            for w in weight_vectors:
                if evaluations >= self.budget:
                    break

                # Find the best point in the archive for this weight
                if archive_F:
                    scores = [np.max(w * (af - self.ideal) / self.range) for af in archive_F]
                    best_idx = np.argmin(scores)
                    start_x = archive_X[best_idx].copy()
                else:
                    start_x = (lo + hi) / 2

                # Run local search
                if evaluations >= self.budget:
                    break
                result_x = local_search(start_x, w, self.ideal)

                # Evaluate the result
                x_c, f = evaluate(result_x)
                add_to_archive(x_c, f)

                # Also try perturbing the result for diversity
                if evaluations < self.budget:
                    perturbed = x_c + 0.05 * (hi - lo) * rng.standard_normal(dim)
                    perturbed = np.clip(perturbed, lo, hi)
                    x_c2, f2 = evaluate(perturbed)
                    add_to_archive(x_c2, f2)

        # Final cleanup: ensure we have a good set of points
        # If budget allows, do a few more random explorations
        while evaluations < self.budget:
            x = lo + (hi - lo) * rng.random(dim)
            x_c, f = evaluate(x)
            add_to_archive(x_c, f)

        return archive_X
# EVOLVE-BLOCK-END
