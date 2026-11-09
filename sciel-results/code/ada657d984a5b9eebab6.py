"""Multi-objective optimizer on RE21 using HypE-style hypervolume contribution selection."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


class Optimizer:
    """HypE-style multi-objective optimizer with LHS init, efficient HV, and NM polish."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280.0  # safety margin before 330s kill

        lo = np.array([b[0] for b in bounds], dtype=np.float64)
        hi = np.array([b[1] for b in bounds], dtype=np.float64)
        span = hi - lo
        ref = np.array([3000.0, 0.06], dtype=np.float64)

        archive_x = []
        archive_f = []
        used = 0

        def is_valid(f):
            return np.all(f < ref) and np.all(np.isfinite(f))

        def is_dominated(f, archive_f_list):
            for af in archive_f_list:
                if np.all(af <= f) and np.any(af < f):
                    return True
            return False

        def dominates(f, af):
            return np.all(f <= af) and np.any(f < af)

        def add_to_archive(x, f):
            nonlocal archive_x, archive_f
            if not is_valid(f):
                return
            # Check if new point is dominated by any in archive
            for af in archive_f:
                if np.all(af <= f) and np.any(af < f):
                    return
            # Remove points dominated by new point
            new_x = []
            new_f = []
            for ax, af in zip(archive_x, archive_f):
                if not (np.all(f <= af) and np.any(f < af)):
                    new_x.append(ax)
                    new_f.append(af)
            new_x.append(x)
            new_f.append(f)
            archive_x = new_x
            archive_f = new_f

        def compute_hv_2d(points, ref_point):
            """Compute hypervolume of 2D points relative to ref_point efficiently."""
            if len(points) == 0:
                return 0.0
            pts = [p for p in points if p[0] < ref_point[0] and p[1] < ref_point[1]]
            if len(pts) == 0:
                return 0.0
            sorted_pts = sorted(pts, key=lambda p: p[0])
            hv = 0.0
            prev_y = ref_point[1]
            for p in sorted_pts:
                if p[1] < prev_y:
                    hv += (ref_point[0] - p[0]) * (prev_y - p[1])
                    prev_y = p[1]
            return hv

        def marginal_hv(point, archive_points, ref_point):
            """Compute marginal hypervolume contribution of a point."""
            hv_without = compute_hv_2d(archive_points, ref_point)
            hv_with = compute_hv_2d(archive_points + [point], ref_point)
            return max(hv_with - hv_without, 1e-15)

        # Phase 1: Latin Hypercube Sampling initialization
        n_init = min(40, self.budget)
        lhs_samples = np.zeros((n_init, self.dim))
        for d in range(self.dim):
            perm = np.random.permutation(n_init)
            u = np.random.random(n_init)
            lhs_samples[:, d] = (perm + u) / n_init

        for i in range(n_init):
            if used >= self.budget or time.time() > deadline:
                break
            x = lo + span * lhs_samples[i]
            f = np.array(problem(x), dtype=np.float64)
            used += 1
            add_to_archive(x.copy(), f)

        # Phase 2: HypE-style evolutionary loop with efficient HV computation
        sigma = 0.15  # initial relative step size
        min_sigma = 0.02
        max_sigma = 0.40

        while used < self.budget - 30 and time.time() < deadline:  # reserve 30 for polish
            if len(archive_x) == 0:
                x = lo + span * np.random.random(self.dim)
                f = np.array(problem(x), dtype=np.float64)
                used += 1
                add_to_archive(x.copy(), f)
                continue

            # Compute marginal hypervolume contributions
            archive_points = list(archive_f)
            contributions = []
            for i in range(len(archive_f)):
                contrib = marginal_hv(archive_f[i], archive_points, ref)
                contributions.append(contrib)

            # Select parent with probability proportional to contribution
            total_contrib = sum(contributions)
            if total_contrib <= 1e-15:
                parent_idx = np.random.randint(len(archive_x))
            else:
                probs = np.array(contributions) / total_contrib
                parent_idx = np.random.choice(len(archive_x), p=probs)

            parent_x = archive_x[parent_idx].copy()
            parent_f = archive_f[parent_idx]

            # Adaptive step size based on progress and success rate
            progress = used / self.budget
            base_step = sigma * (1.0 - 0.4 * progress)

            # Occasionally use larger or smaller steps for exploration
            if np.random.random() < 0.15:
                step_mult = np.random.choice([0.5, 2.0, 3.0])
            else:
                step_mult = 1.0

            step = base_step * step_mult
            noise = np.random.standard_normal(self.dim)
            x_off = parent_x + step * span * noise
            x_off = np.clip(x_off, lo, hi)

            f_off = np.array(problem(x_off), dtype=np.float64)
            used += 1

            was_dominated = is_dominated(f_off, archive_f)
            add_to_archive(x_off.copy(), f_off)

            # Adaptive sigma
            if not was_dominated and is_valid(f_off):
                sigma *= 0.96
            else:
                sigma *= 1.04

            sigma = np.clip(sigma, min_sigma, max_sigma)

        # Phase 3: Nelder-Mead polish on top hypervolume contributors
        if len(archive_x) > 0 and used < self.budget and time.time() < deadline:
            archive_points = list(archive_f)
            contributions = []
            for i in range(len(archive_f)):
                contrib = marginal_hv(archive_f[i], archive_points, ref)
                contributions.append(contrib)

            # Get top-8 by contribution
            top_k = min(8, len(archive_x))
            top_indices = np.argsort(contributions)[-top_k:]
            top_indices = top_indices[::-1]  # descending order

            for idx in top_indices:
                if used >= self.budget or time.time() > deadline:
                    break

                x0 = archive_x[idx].copy()
                f0 = archive_f[idx]

                # Use weighted sum scalarization with weights based on reference distance
                w0 = (ref[0] - f0[0]) / max(ref[0] - lo[0], 1e-10)
                w1 = (ref[1] - f0[1]) / max(ref[1] - hi[1] + (hi[1] - lo[1]), 1e-10)
                w = np.array([w0, w1])
                w = w / (np.sum(w) + 1e-10)

                eval_count = [0]

                def scalarized_objective(x, w=w):
                    if used >= self.budget or time.time() > deadline:
                        return 1e10
                    fx = np.array(problem(x), dtype=np.float64)
                    eval_count[0] += 1
                    return float(np.dot(w, fx))

                try:
                    result = minimize(
                        scalarized_objective,
                        x0,
                        method='Nelder-Mead',
                        options={
                            'xatol': 1e-8,
                            'fatol': 1e-10,
                            'maxiter': 80,
                            'maxfev': 25
                        }
                    )
                    x_best = np.clip(result.x, lo, hi)
                    f_best = np.array(problem(x_best), dtype=np.float64)
                    used += 1
                    add_to_archive(x_best.copy(), f_best)
                except Exception:
                    pass

        return archive_x
# EVOLVE-BLOCK-END
