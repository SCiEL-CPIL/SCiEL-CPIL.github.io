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
            """Compute marginal hypervolume contribution of a point in O(n) for 2D."""
            # For 2D, the marginal contribution of point p to a set S is:
            # (ref_x - p[0]) * (max(0, min_y_of_dominating_or_equal - p[1]))
            # where min_y is the minimum y among points in S that have x <= p[0]
            # and y <= p[1] (i.e., points that dominate or are dominated by p in a specific way)
            # Actually, for 2D: contribution = (ref_x - p[0]) * (gap_y)
            # where gap_y = max(0, min(y of points with x<=p[0] and y<=p[1], ref_y) - p[1])
            # But we need to be careful: the contribution is the area that p adds.
            # In 2D, if we sort by x, p contributes width*(ref_y - p[1]) if no other point
            # with smaller x has smaller y. More precisely:
            # contribution = (ref_x - p[0]) * max(0, min_y_of_points_with_x_le_p0_and_y_le_p1 - p[1])
            # where if no such point exists, min_y = ref_y.
            
            min_y = ref_point[1]
            for q in archive_points:
                if q[0] <= point[0] and q[1] <= point[1]:
                    if q[1] < min_y:
                        min_y = q[1]
            gap = min_y - point[1]
            if gap <= 0:
                return 0.0
            return (ref_point[0] - point[0]) * gap

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

        # Phase 1: Latin Hypercube initialization for better spread
        n_init = min(40, self.budget)
        for i in range(n_init):
            if used >= self.budget or time.time() > deadline:
                break
            x = np.zeros(self.dim)
            for d in range(self.dim):
                # LHS: stratify each dimension independently
                stratum = np.random.randint(0, n_init)
                frac = (stratum + np.random.random()) / n_init
                x[d] = lo[d] + (hi[d] - lo[d]) * frac
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
        
        # Phase 3: Local polish on top archive points using scipy
        if time.time() < deadline - 30.0 and len(archive) > 0:
            from scipy.optimize import minimize
            
            # Sort archive by hypervolume contribution (top points first)
            archive_points = [f for _, f in archive]
            contributions = []
            for i, (x_i, f_i) in enumerate(archive):
                contrib = marginal_hypervolume(f_i, archive_points, ref)
                contributions.append((contrib, i))
            
            # Take top 8 points for local optimization
            contributions.sort(reverse=True)
            n_polish = min(8, len(archive))
            polish_indices = [idx for _, idx in contributions[:n_polish]]
            
            for idx in polish_indices:
                if used >= self.budget or time.time() > deadline - 10.0:
                    break
                
                x_init, f_init = archive[idx]
                
                # Optimize a weighted sum (scalarization) to find new Pareto points
                # Use different weights to explore different parts of the front
                for w1 in [0.3, 0.5, 0.7]:
                    if used >= self.budget or time.time() > deadline - 10.0:
                        break
                    w2 = 1.0 - w1
                    
                    def objective(x):
                        f = problem(x)
                        return w1 * f[0] + w2 * f[1]
                    
                    try:
                        result = minimize(
                            objective,
                            x_init,
                            method='L-BFGS-B',
                            bounds=list(bounds),
                            options={'maxiter': 80, 'gtol': 1e-8}
                        )
                        if result.success and used < self.budget:
                            f_new = problem(result.x)
                            used += 1
                            if is_valid(f_new):
                                add_to_archive(result.x, f_new)
                    except Exception:
                        pass
        
        return archive

# EVOLVE-BLOCK-END
