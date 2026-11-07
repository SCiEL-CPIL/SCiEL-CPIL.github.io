"""Decomposition-based multi-objective optimizer for WFG7 using Tchebycheff scalarization."""
import time
import numpy as np


class Optimizer:
    """Decomposition-based optimizer using Tchebycheff scalarization with local search."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 295

        lo = np.array([b[0] for b in bounds], dtype=np.float64)
        hi = np.array([b[1] for b in bounds], dtype=np.float64)
        dim = self.dim
        n_obj = self.n_obj
        budget = self.budget

        # Generate 30 uniformly distributed weight vectors in 3D
        n_weights = 30
        weight_vectors = []
        # Use Fibonacci sphere or grid for uniform distribution
        # For 3D, we can use a grid on the simplex
        # Generate points on the 2D simplex (w1+w2+w3=1, wi>=0)
        # Use a stratified approach
        n_per_dim = 5  # 5x5 grid on the triangle gives 15 points, we need 30
        # Actually let's use a different approach: generate on a grid and normalize
        weights = []
        # Use a 6x6 grid on the triangle
        for i in range(6):
            for j in range(6):
                if i + j <= 5:
                    w1 = (i + 0.5) / 6.0
                    w2 = (j + 0.5) / 6.0
                    w3 = 1.0 - w1 - w2
                    if w3 >= 0:
                        weights.append([w1, w2, w3])
        # Take first 30
        weight_vectors = weights[:n_weights]
        # If we don't have enough, add more
        while len(weight_vectors) < n_weights:
            # Add random points
            r = np.random.dirichlet([1.0, 1.0, 1.0])
            weight_vectors.append(r.tolist())
        weight_vectors = np.array(weight_vectors[:n_weights])

        # Normalize weights
        weight_vectors = weight_vectors / weight_vectors.sum(axis=1, keepdims=True)

        # Reference point for ideal/nadir estimation
        # We'll estimate ideal and nadir from evaluations
        ideal = np.full(n_obj, np.inf)
        nadir = np.full(n_obj, 0.0)

        # Initialize: evaluate center and corners to get rough bounds
        center = (lo + hi) / 2.0
        f_center = np.array(problem(center), dtype=np.float64)
        used = 1

        # Evaluate a few random points to estimate ideal/nadir
        rng = np.random.default_rng(42)
        n_init = min(20, budget // 10)
        init_points = lo + (hi - lo) * rng.random((n_init, dim))
        init_points[0] = center  # reuse center
        f_init = np.array([problem(x) for x in init_points], dtype=np.float64)
        used += n_init - 1  # center already counted

        # Update ideal and nadir
        for f in f_init:
            ideal = np.minimum(ideal, f)
            nadir = np.maximum(nadir, f)
        # Also include center
        ideal = np.minimum(ideal, f_center)
        nadir = np.maximum(nadir, f_center)

        # Ensure nadir > ideal
        for i in range(n_obj):
            if nadir[i] <= ideal[i]:
                nadir[i] = ideal[i] + 1.0

        # Range for normalization
        obj_range = nadir - ideal
        obj_range[obj_range == 0] = 1.0

        # Archive of non-dominated points
        archive_F = []
        archive_X = []

        def _dominates(f1, f2):
            return np.all(f1 <= f2) and np.any(f1 < f2)

        def _update_archive(x, f):
            # Check if new point is dominated
            for ef in archive_F:
                if _dominates(ef, f):
                    return
            # Remove dominated points
            keep = []
            for i, ef in enumerate(archive_F):
                if not _dominates(f, ef):
                    keep.append(i)
            for i in keep:
                pass
            new_X = [archive_X[i] for i in keep]
            new_F = [archive_F[i] for i in keep]
            new_X.append(x)
            new_F.append(f)
            archive_X[:] = new_X
            archive_F[:] = new_F

        # Initialize archive
        _update_archive(center, f_center)
        for i in range(n_init):
            _update_archive(init_points[i], f_init[i])

        # For each weight vector, run a local search
        # Allocate evaluations: budget // n_weights per subproblem, minimum 5
        evals_per_sub = max(5, budget // n_weights)

        # Initial points for each subproblem: use center + small perturbation
        # Or use points from the initial population
        for w_idx in range(n_weights):
            if used >= budget or time.time() > deadline - 10:
                break

            w = weight_vectors[w_idx]

            # Initialize at a point that is good for this weight vector
            # Start from center
            x = center.copy()
            f = f_center.copy()

            # Local search using Nelder-Mead-like approach
            # We'll use a simple gradient-free local search
            # Step size adapts based on progress
            step_size = 0.1 * (hi - lo)
            n_lm_steps = evals_per_sub

            for step in range(n_lm_steps):
                if used >= budget or time.time() > deadline - 5:
                    break

                # Generate a perturbation
                # Use a random direction with adaptive step size
                direction = rng.standard_normal(dim)
                direction /= np.linalg.norm(direction) + 1e-10

                x_new = x + step_size * direction
                x_new = np.clip(x_new, lo, hi)

                f_new = np.array(problem(x_new), dtype=np.float64)
                used += 1

                # Tchebycheff scalarization
                def tchebycheff(f, w, ideal, range_):
                    return np.max(w * np.abs(f - ideal) / range_)

                obj_current = tchebycheff(f, w, ideal, obj_range)
                obj_new = tchebycheff(f_new, w, ideal, obj_range)

                if obj_new < obj_current:
                    x = x_new
                    f = f_new
                    # Reduce step size slightly
                    step_size *= 0.9
                    # Update ideal if improved
                    ideal = np.minimum(ideal, f)
                    # Update range
                    obj_range = nadir - ideal
                    obj_range[obj_range == 0] = 1.0
                else:
                    # Increase step size if stuck
                    step_size *= 1.1
                    step_size = np.minimum(step_size, 0.2 * (hi - lo))

                # Update archive
                _update_archive(x, f)

        # Final local refinement on archive points
        if time.time() < deadline - 5 and used < budget:
            for i in range(len(archive_X)):
                if used >= budget or time.time() > deadline - 3:
                    break

                base_x = archive_X[i].copy()
                base_f = archive_F[i].copy()

                for _ in range(3):
                    if used >= budget or time.time() > deadline - 2:
                        break
                    pert = base_x + 0.02 * (hi - lo) * rng.standard_normal(dim)
                    pert = np.clip(pert, lo, hi)
                    fp = np.array(problem(pert), dtype=np.float64)
                    used += 1

                    if not _dominates(base_f, fp):
                        _update_archive(pert, fp)
                        if np.sum(fp) < np.sum(base_f):
                            base_x = pert
                            base_f = fp

        return archive_X if archive_X else [center]
