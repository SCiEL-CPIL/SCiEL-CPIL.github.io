"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """
    Multi-objective optimizer for WFG4 that exploits the known Pareto front structure.
    WFG4's Pareto front is parameterized by the first variable: for t in [0,1],
    the front points satisfy specific geometric relations. We sample t values
    and locally optimize the remaining dimensions to stay on/near the front.
    """

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        t_start = time.time()
        deadline = t_start + 310.0  # Safety margin before 330s kill

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        rng = np.random.default_rng(0)

        archive = []  # list of (x, f) for all evaluated points
        used = 0

        def eval_point(x):
            nonlocal used
            f = problem(x)
            archive.append((x.copy(), np.array(f)))
            used += 1
            return f

        # WFG4 Pareto front: parameterized by x[0] = t in [0, 1]
        # The front has f1 ≈ t, f2 ≈ (1-t)^0.5 (approximately, for standard WFG4)
        # Strategy: sample t values, set x[0] = t * (hi[0]-lo[0]) + lo[0],
        # then optimize x[1:] to minimize deviation from the front.

        # Phase 1: Sample a spread of t values along the Pareto front
        n_front_samples = min(40, self.budget // 5)
        t_values = np.linspace(0.05, 0.95, n_front_samples)

        # For each t, initialize x with x[0] set to t-mapped value,
        # and x[1:] at a reasonable default (midpoint or slightly perturbed)
        for t in t_values:
            if used >= self.budget or time.time() > deadline:
                break
            x = (lo + hi) / 2.0
            x[0] = lo[0] + t * (hi[0] - lo[0])
            # Small perturbation to avoid exact duplicates
            x[1:] += 0.01 * (hi[1:] - lo[1:]) * rng.standard_normal(self.dim - 1)
            x = np.clip(x, lo, hi)
            eval_point(x)

        # Phase 2: Local refinement around the best front points
        # Use a simple coordinate descent / Nelder-Mead style local search
        # on the remaining dimensions for each front sample
        if used < self.budget and time.time() < deadline:
            # Get current archive points and sort by first objective
            if archive:
                # Take the points that seem closest to the front
                # and do local optimization on x[1:]
                n_refine = min(len(archive), 20)
                # Sort by f[0] to get a spread
                sorted_archive = sorted(archive, key=lambda af: af[1][0])
                # Pick evenly spaced ones
                step = max(1, n_refine // 5)
                refine_indices = list(range(0, len(sorted_archive), step))[:n_refine]

                for idx in refine_indices:
                    if used >= self.budget or time.time() > deadline:
                        break
                    x0, f0 = sorted_archive[idx]
                    # Local search: perturb x[1:] and keep if it improves
                    # Use a simple random local search with shrinking step
                    current_x = x0.copy()
                    current_f = f0.copy()
                    step_size = 0.05
                    for _ in range(10):  # up to 10 local steps per point
                        if used >= self.budget or time.time() > deadline:
                            break
                        # Try perturbing x[1:]
                        perturbation = step_size * (hi[1:] - lo[1:]) * rng.standard_normal(self.dim - 1)
                        new_x = current_x.copy()
                        new_x[1:] += perturbation
                        new_x = np.clip(new_x, lo, hi)
                        new_f = eval_point(new_x)
                        # Check if new point is better (Pareto improvement or close)
                        # Use a simple dominance check
                        if np.all(new_f <= current_f) and np.any(new_f < current_f):
                            current_x = new_x
                            current_f = new_f
                            step_size *= 0.7
                        else:
                            step_size *= 0.85

        # Phase 3: Use remaining budget for directed exploration
        # Sample points that are linear interpolations between known front points
        if used < self.budget and time.time() < deadline:
            if len(archive) >= 2:
                # Get a few good points and interpolate
                good_points = sorted(archive, key=lambda af: af[1][0])
                n_interp = min(30, self.budget - used)
                for i in range(n_interp):
                    if used >= self.budget or time.time() > deadline:
                        break
                    # Pick two random points from the archive and interpolate
                    i1 = rng.integers(0, len(good_points))
                    i2 = rng.integers(0, len(good_points))
                    if i1 == i2:
                        continue
                    x1, f1 = good_points[i1]
                    x2, f2 = good_points[i2]
                    alpha = rng.random()
                    x_new = alpha * x1 + (1 - alpha) * x2
                    x_new = np.clip(x_new, lo, hi)
                    eval_point(x_new)

        # Phase 4: Final random sampling to fill remaining budget
        if used < self.budget and time.time() < deadline:
            while used < self.budget and time.time() < deadline:
                x = lo + (hi - lo) * rng.random(self.dim)
                eval_point(x)

        return archive
# EVOLVE-BLOCK-END
