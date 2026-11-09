"""Multi-objective optimizer on RE21 using HypE-style hypervolume contribution selection."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """HypE-style multi-objective optimizer with hypervolume contribution selection."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280.0  # safety margin before 330s kill

        lo = np.array([b[0] for b in bounds], dtype=np.float64)
        hi = np.array([b[1] for b in bounds], dtype=np.float64)
        ref = np.array([3000.0, 0.06], dtype=np.float64)

        def is_valid(f):
            return np.all(f < ref)

        # Archive of non-dominated points
        archive = []  # list of (x, f)
        used = 0

        def add_to_archive(x, f):
            nonlocal used
            if not is_valid(f):
                return
            # Check if dominated by any point in archive
            for ax, af in archive:
                if np.all(af <= f) and np.any(af < f):
                    return  # dominated, don't add
            # Remove dominated points from archive
            new_archive = []
            for ax, af in archive:
                if not (np.all(f <= af) and np.any(f < af)):
                    new_archive.append((ax, af))
            new_archive.append((x, f))
            archive[:] = new_archive

        def hv_2d(points, ref_pt):
            """Compute hypervolume in 2D for points strictly better than ref_pt."""
            if not points:
                return 0.0
            # Sort by first objective
            pts = sorted(points, key=lambda p: p[0])
            hv = 0.0
            prev_y = ref_pt[1]
            for p in pts:
                if p[1] >= ref_pt[1]:
                    continue
                # Width in x: from p[0] to ref_pt[0]
                # Height in y: from p[1] to prev_y
                width = ref_pt[0] - p[0]
                height = prev_y - p[1]
                if width > 0 and height > 0:
                    hv += width * height
                prev_y = min(prev_y, p[1])
            return hv

        def marginal_contributions(archive_pts, ref_pt):
            """Compute marginal hypervolume contribution of each point in archive."""
            n = len(archive_pts)
            if n == 0:
                return np.array([])
            total_hv = hv_2d(archive_pts, ref_pt)
            contribs = np.zeros(n)
            for i in range(n):
                remaining = [archive_pts[j] for j in range(n) if j != i]
                hv_without = hv_2d(remaining, ref_pt)
                contribs[i] = total_hv - hv_without
            return contribs

        def sample_uniform():
            return lo + (hi - lo) * np.random.random(self.dim)

        def sample_gaussian(parent, sigma_scale):
            sigma = sigma_scale * (hi - lo)
            child = parent + sigma * np.random.standard_normal(self.dim)
            return np.clip(child, lo, hi)

        # Phase 1: LHS-like initialization with spread
        n_init = min(30, self.budget // 3)
        # Use stratified sampling for better spread
        for i in range(n_init):
            if used >= self.budget:
                break
            if time.time() > deadline:
                break
            # Stratified: divide each dimension into n_init bins
            x = np.zeros(self.dim)
            for d in range(self.dim):
                bin_start = i / n_init
                bin_end = (i + 1) / n_init
                x[d] = lo[d] + (hi[d] - lo[d]) * (bin_start + np.random.random() * (bin_end - bin_start))
            f = problem(x)
            used += 1
            add_to_archive(x, f)

        # Phase 2: HypE-style selection and mutation
        # Adaptively choose between uniform and Gaussian mutation
        mutation_sigma = 0.3  # initial sigma scale
        min_sigma = 0.01
        max_sigma = 0.5

        while used < self.budget:
            if time.time() > deadline:
                break

            # Compute marginal contributions for selection
            if len(archive) >= 2:
                pts = [f for _, f in archive]
                contribs = marginal_contributions(pts, ref)
                # Normalize to probabilities
                contribs = np.maximum(contribs, 1e-12)
                probs = contribs / contribs.sum()
                idx = np.random.choice(len(archive), p=probs)
            elif len(archive) == 1:
                idx = 0
            else:
                # No valid points yet, just sample uniformly
                x = sample_uniform()
                f = problem(x)
                used += 1
                add_to_archive(x, f)
                continue

            parent_x, parent_f = archive[idx]

            # Decide mutation strategy
            r = np.random.random()
            if r < 0.3:
                # Uniform exploration
                x = sample_uniform()
            else:
                # Gaussian exploitation around parent
                x = sample_gaussian(parent_x, mutation_sigma)

            f = problem(x)
            used += 1
            add_to_archive(x, f)

            # Adapt sigma: decrease if archive is growing, keep if stable
            if len(archive) > 0:
                # Slowly decrease sigma to focus search
                mutation_sigma = max(min_sigma, mutation_sigma * 0.995)

        return archive
# EVOLVE-BLOCK-END
