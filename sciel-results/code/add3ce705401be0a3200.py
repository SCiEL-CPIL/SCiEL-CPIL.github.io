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

        archive = []  # list of (x, f) tuples
        used = 0

        def is_valid(f):
            return np.all(f < ref) and np.all(np.isfinite(f))

        def compute_hypervolume(points, ref_point):
            """Compute hypervolume of 2D points relative to ref_point."""
            if len(points) == 0:
                return 0.0
            # Sort by first objective ascending
            sorted_pts = sorted(points, key=lambda p: p[0])
            hv = 0.0
            prev_y = ref_point[1]
            for p in sorted_pts:
                if p[0] >= ref_point[0] or p[1] >= ref_point[1]:
                    continue
                if p[1] < prev_y:
                    hv += (ref_point[0] - p[0]) * (prev_y - p[1])
                    prev_y = p[1]
            return hv

        def marginal_hypervolume(point, archive_points, ref_point):
            """Compute marginal hypervolume contribution of a point."""
            hv_with = compute_hypervolume(archive_points + [point], ref_point)
            hv_without = compute_hypervolume(archive_points, ref_point)
            return hv_with - hv_without

        def add_to_archive(x, f):
            nonlocal used
            if not is_valid(f):
                return
            # Check if dominated by existing archive
            for _, af in archive:
                if np.all(af <= f) and np.any(af < f):
                    return
            # Remove dominated points
            new_archive = []
            for ax, af in archive:
                if not (np.all(f <= af) and np.any(f < af)):
                    new_archive.append((ax, af))
            new_archive.append((x, f))
            archive[:] = new_archive

        def evaluate(x):
            nonlocal used
            if used >= self.budget or time.time() > deadline:
                return None
            f = problem(x)
            used += 1
            if is_valid(f):
                add_to_archive(x, f)
            return f

        # Phase 1: Stratified initialization
        n_init = min(40, self.budget)
        for i in range(n_init):
            if used >= self.budget or time.time() > deadline:
                break
            x = np.zeros(self.dim)
            for d in range(self.dim):
                x[d] = lo[d] + (hi[d] - lo[d]) * ((i + np.random.random()) / n_init)
            evaluate(x)

        # Phase 2: HypE-style evolutionary loop
        # Adaptive sigma for mutation
        sigma = 0.15  # initial relative step size
        
        while used < self.budget and time.time() < deadline:
            if len(archive) == 0:
                # If archive is empty, do random search
                x = np.random.uniform(lo, hi)
                evaluate(x)
                continue
            
            # Compute marginal hypervolume contributions
            archive_points = [f for _, f in archive]
            contributions = []
            for i, (x_i, f_i) in enumerate(archive):
                contrib = marginal_hypervolume(f_i, archive_points, ref)
                contributions.append(contrib)
            
            # Select parent with probability proportional to contribution
            total_contrib = sum(contributions)
            if total_contrib <= 0:
                parent_idx = np.random.randint(len(archive))
            else:
                probs = np.array(contributions) / total_contrib
                parent_idx = np.random.choice(len(archive), p=probs)
            
            parent_x, parent_f = archive[parent_idx]
            
            # Generate offspring via Gaussian mutation
            # Use adaptive sigma
            x_off = parent_x.copy()
            for d in range(self.dim):
                step = sigma * (hi[d] - lo[d])
                x_off[d] += np.random.normal(0, step)
                x_off[d] = np.clip(x_off[d], lo[d], hi[d])
            
            f_off = evaluate(x_off)
            
            # Adaptive sigma decay
            if f_off is not None and is_valid(f_off):
                # Check if offspring is non-dominated
                dominated = False
                for _, af in archive:
                    if np.all(af <= f_off) and np.any(af < f_off):
                        dominated = True
                        break
                if not dominated:
                    sigma *= 0.95  # slight decay on success
                else:
                    sigma *= 1.05  # slight increase on failure
            else:
                sigma *= 1.05
            
            # Keep sigma in reasonable bounds
            sigma = np.clip(sigma, 0.01, 0.5)
        
        return archive

# EVOLVE-BLOCK-END
