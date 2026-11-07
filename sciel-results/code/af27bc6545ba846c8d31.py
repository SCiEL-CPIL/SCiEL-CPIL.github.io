"""Multi-objective optimizer for WFG4 using a hypervolume-guided coordinate descent."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Hypervolume-guided coordinate descent for multi-objective optimization."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280.0  # safety margin before 300s limit

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        dim = self.dim
        n_obj = self.n_obj
        budget = self.budget
        ref = np.array([3.0, 5.0, 7.0])

        # Track all evaluated points and their objective values
        all_points = []
        all_fvals = []
        eval_count = 0

        def evaluate(x):
            nonlocal eval_count
            f = np.array(problem(x), dtype=float)
            eval_count += 1
            return f

        def hv_contribution(f, archive):
            """Compute the hypervolume contribution of point f to the archive.
            Returns 0 if f is dominated by any point in archive or worse than ref."""
            # Check if f is better than reference
            if np.any(f >= ref):
                return 0.0
            # Check if f is dominated by any point in archive
            for af in archive:
                if np.all(af <= f) and np.any(af < f):
                    return 0.0
            # Compute contribution: volume of region dominated by f but not by archive
            # In 3D, contribution = product of (ref[i] - f[i]) minus overlaps
            # Simple approximation: just the box volume (crude but fast)
            vol = 1.0
            for i in range(n_obj):
                vol *= (ref[i] - f[i])
            # Subtract volumes dominated by archive points that are "below" f
            # This is an approximation; exact HV contribution is complex
            return vol

        def fast_hv_contribution(f, archive):
            """Faster approximate HV contribution."""
            if np.any(f >= ref):
                return 0.0
            # Quick dominance check
            for af in archive:
                if np.all(af <= f) and np.any(af < f):
                    return 0.0
            # Box volume
            vol = 1.0
            for i in range(n_obj):
                d = ref[i] - f[i]
                if d <= 0:
                    return 0.0
                vol *= d
            return vol

        # Initialize with random points
        rng = np.random.default_rng(42)
        n_init = min(20, budget // 10)
        for _ in range(n_init):
            x = lo + (hi - lo) * rng.random(dim)
            f = evaluate(x)
            all_points.append(x.copy())
            all_fvals.append(f.copy())

        # Build initial non-dominated archive
        def get_non_dominated(points, fvals):
            """Return indices of non-dominated points."""
            n = len(fvals)
            if n == 0:
                return []
            nd = list(range(n))
            for i in range(n):
                for j in range(n):
                    if i != j and i in nd:
                        fi = fvals[i]
                        fj = fvals[j]
                        if np.all(fj <= fi) and np.any(fj < fi):
                            nd.remove(i)
                            break
            return nd

        # Coordinate descent with ternary search on each dimension
        # Use a small archive of best points to guide the search
        max_archive_size = 50

        # Main optimization loop
        while eval_count < budget:
            # Check time
            if time.time() > deadline - 5.0:
                break

            # Get non-dominated set
            nd_idx = get_non_dominated(all_points, all_fvals)
            if not nd_idx:
                break

            # Pick a random non-dominated point to refine
            base_idx = rng.choice(nd_idx)
            base_x = all_points[base_idx].copy()
            base_f = all_fvals[base_idx].copy()

            # For each dimension, try to improve via local search
            for d in range(dim):
                if eval_count >= budget:
                    break
                if time.time() > deadline - 5.0:
                    break

                # Try several step sizes in both directions
                best_x = base_x.copy()
                best_f = base_f.copy()
                best_score = fast_hv_contribution(base_f, [all_fvals[i] for i in nd_idx if i != base_idx])

                # Adaptive step sizes
                span = hi[d] - lo[d]
                for step_frac in [0.5, 0.25, 0.1, 0.05, 0.02, 0.01]:
                    for direction in [-1, 1]:
                        if eval_count >= budget:
                            break
                        new_x = base_x.copy()
                        new_x[d] = base_x[d] + direction * step_frac * span
                        new_x[d] = np.clip(new_x[d], lo[d], hi[d])

                        if np.allclose(new_x, base_x):
                            continue

                        new_f = evaluate(new_x)

                        # Check if this improves our local best
                        # Simple criterion: is it non-dominated w.r.t. base_f?
                        if np.any(new_f < best_f) and not np.all(new_f <= best_f):
                            # Check if it's better than base in at least one obj
                            # and not worse in all
                            if not np.all(new_f >= best_f):
                                best_x = new_x.copy()
                                best_f = new_f.copy()
                                best_score = fast_hv_contribution(
                                    new_f,
                                    [all_fvals[i] for i in nd_idx if i != base_idx]
                                )

                # If we found a better point, add it
                if not np.allclose(best_x, base_x):
                    all_points.append(best_x.copy())
                    all_fvals.append(best_f.copy())

            # Occasionally add a random restart to escape local optima
            if eval_count < budget and rng.random() < 0.1:
                if time.time() > deadline - 5.0:
                    break
                x = lo + (hi - lo) * rng.random(dim)
                f = evaluate(x)
                all_points.append(x.copy())
                all_fvals.append(f.copy())

        return all_points
# EVOLVE-BLOCK-END
