# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize


class Optimizer:
    """
    Multi-objective optimizer using reference-point-based adaptive scalarization
    (Tchebycheff) with multi-start L-BFGS-B, dynamically targeting hypervolume gaps.
    """

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        lo = np.array([b[0] for b in bounds], dtype=float)
        hi = np.array([b[1] for b in bounds], dtype=float)
        dim = self.dim
        n_obj = self.n_obj
        budget = self.budget

        # Reference point for hypervolume calculation (from task)
        ref_point = np.array([1705.0, 12.0, 0.3])

        # Archive of non-dominated points
        archive_X = []
        archive_F = []
        used = 0

        def is_dominated(f_new, archive_F):
            """Check if f_new is dominated by any point in archive_F."""
            for f in archive_F:
                if np.all(f <= f_new) and np.any(f < f_new):
                    return True
            return False

        def dominates(f_new, f_old):
            """Check if f_new dominates f_old."""
            return np.all(f_new <= f_old) and np.any(f_new < f_old)

        def update_archive(x, f):
            """Update the non-dominated archive."""
            nonlocal archive_X, archive_F
            # Check if new point is dominated by existing archive
            for f_old in archive_F:
                if dominates(f_old, f):
                    return
            # Remove points dominated by new point
            new_X = []
            new_F = []
            for x_old, f_old in zip(archive_X, archive_F):
                if not dominates(f, f_old):
                    new_X.append(x_old)
                    new_F.append(f_old)
            new_X.append(x)
            new_F.append(f)
            archive_X = new_X
            archive_F = new_F

        def compute_hypervolume_contribution(f, ref):
            """
            Approximate the hypervolume contribution of a single point in 3D.
            This is a simplified contribution metric for ranking.
            """
            if np.any(f >= ref):
                return 0.0
            # In 3D, the contribution of a point depends on its neighbors in the Pareto front.
            # For simplicity, use the volume of the box from f to ref, adjusted for dominance.
            # This is not exact but gives a reasonable ranking.
            return np.prod(ref - f)

        def get_weight_vector(target_F, ref):
            """
            Generate a Tchebycheff weight vector that points toward the target region.
            """
            # Normalize target relative to the range of the archive
            if len(archive_F) > 1:
                F_arr = np.array(archive_F)
                f_min = np.min(F_arr, axis=0)
                f_max = np.max(F_arr, axis=0)
                f_range = np.maximum(f_max - f_min, 1e-10)
            else:
                f_min = target_F
                f_range = np.ones(n_obj) * 10.0

            # Weight inversely proportional to distance from target to current best
            # Points far from the target should have higher weight to pull search there
            dist = np.abs(target_F - f_min)
            w = 1.0 / (dist + 1e-10)
            w = w / np.sum(w)
            return w

        def tchebycheff_objective(x, w, z, ref):
            """
            Tchebycheff scalarization: min max_i w_i * |f_i(x) - z_i|
            z is the ideal point (or reference).
            """
            f = problem(x)
            # Use the current best known values as reference for Tchebycheff
            if len(archive_F) > 0:
                F_arr = np.array(archive_F)
                z = np.min(F_arr, axis=0)
            else:
                z = np.array([1.0, 1.0, 0.01])  # reasonable lower bounds

            # Tchebycheff: max of weighted absolute deviations
            vals = w * np.abs(f - z)
            return np.max(vals)

        def tchebycheff_grad(x, w, z):
            """
            Approximate gradient for Tchebycheff objective.
            Since problem is a black-box, we use finite differences.
            """
            eps = 1e-6
            grad = np.zeros(dim)
            f0 = tchebycheff_objective(x, w, z, ref_point)
            for i in range(dim):
                x_plus = x.copy()
                x_plus[i] += eps
                x_plus = np.clip(x_plus, lo, hi)
                f_plus = tchebycheff_objective(x_plus, w, z, ref_point)
                grad[i] = (f_plus - f0) / eps
            return grad

        # Phase 1: Initial random sampling to get a rough sense of the Pareto front
        n_init = min(30, budget // 5)
        rng = np.random.default_rng(42)
        X_init = lo + (hi - lo) * rng.random((n_init, dim))
        for x in X_init:
            if used >= budget:
                break
            f = problem(x)
            used += 1
            if np.all(f < ref_point):  # Only keep points better than reference
                update_archive(x, f)

        # Phase 2: Adaptive scalarization with L-BFGS-B
        # Target points to fill hypervolume gaps
        n_targets = min(15, (budget - used) // 10)  # Leave some budget for each optimization
        if n_targets > 0 and used < budget:
            # Generate target points on the "hypersphere" or in the objective space
            # We want to target regions of the Pareto front that are poorly covered
            if len(archive_F) > 0:
                F_arr = np.array(archive_F)
                f_min = np.min(F_arr, axis=0)
                f_max = np.max(F_arr, axis=0)
            else:
                f_min = np.array([1.0, 1.0, 0.01])
                f_max = np.array([1705.0, 12.0, 0.3])

            # Create target points that spread across the objective space
            targets = []
            for i in range(n_targets):
                # Random point in the box [f_min, f_max] but biased toward extremes
                t = rng.random(n_obj)
                # Bias toward the boundaries to explore the full front
                t = t ** 2  # Square to bias toward 0 and 1
                target = f_min + t * (f_max - f_min)
                # Clip to be strictly better than reference
                target = np.minimum(target, ref_point - 1e-6)
                targets.append(target)

            # For each target, run L-BFGS-B from multiple starting points
            starts_per_target = max(2, (budget - used) // (n_targets * 3))
            for target in targets:
                if used >= budget:
                    break

                # Get weight vector for this target
                w = get_weight_vector(target, ref_point)

                # Get z (ideal point)
                if len(archive_F) > 0:
                    F_arr = np.array(archive_F)
                    z = np.min(F_arr, axis=0)
                else:
                    z = np.array([1.0, 1.0, 0.01])

                # Multiple starts: from archive points and random points
                starts = []
                if archive_X:
                    # Pick a few archive points as starts
                    idx = rng.choice(len(archive_X), size=min(3, len(archive_X)), replace=False)
                    for i in idx:
                        starts.append(archive_X[i].copy())
                # Add random starts
                for _ in range(min(2, starts_per_target)):
                    starts.append(lo + (hi - lo) * rng.random(dim))

                for x0 in starts:
                    if used >= budget:
                        break

                    # Run L-BFGS-B with a small budget of evaluations
                    # Since we can't count evaluations inside minimize easily,
                    # we limit maxiter and use a simple approach
                    try:
                        # Use a simple objective that calls problem
                        # We need to track evaluations, so we'll use a wrapper
                        eval_count = [0]
                        max_evals = min(20, budget - used)

                        def obj_wrapper(x, w=w, z=z):
                            if eval_count[0] >= max_evals:
                                return 1e10
                            eval_count[0] += 1
                            f = problem(x)
                            used_local = eval_count[0]
                            vals = w * np.abs(f - z)
                            return np.max(vals)

                        # Run optimization
                        result = minimize(
                            obj_wrapper,
                            x0,
                            method='L-BFGS-B',
                            bounds=list(zip(lo, hi)),
                            options={'maxiter': 10, 'maxfun': max_evals}
                        )

                        # Update used count
                        used += eval_count[0]

                        # Check result
                        if result.success and eval_count[0] > 0:
                            x_opt = np.clip(result.x, lo, hi)
                            f_opt = problem(x_opt)
                            # Note: f_opt was already evaluated in obj_wrapper, but we call again
                            # to be safe. This uses one extra evaluation.
                            used += 1
                            if used <= budget and np.all(f_opt < ref_point):
                                update_archive(x_opt, f_opt)

                    except Exception:
                        continue

        # Phase 3: Final polish - if budget remains, do a few more random evaluations
        # to catch any missed improvements
        while used < budget:
            x = lo + (hi - lo) * rng.random(dim)
            f = problem(x)
            used += 1
            if np.all(f < ref_point):
                update_archive(x, f)

        return archive_X
# EVOLVE-BLOCK-END
