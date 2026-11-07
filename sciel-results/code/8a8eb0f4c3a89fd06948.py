"""Multi-objective optimizer for RE21 with grid initialization and HV-guided adaptive DE."""
# EVOLVE-BLOCK-START
import numpy as np


class Optimizer:
    """Reference-point-aligned grid init + hypervolume-guided adaptive DE for RE21."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        lo = np.array([b[0] for b in bounds], dtype=float)
        hi = np.array([b[1] for b in bounds], dtype=float)
        dim = self.dim
        budget = self.budget
        ref = np.array([3000.0, 0.06])

        def dominates(a, b):
            return np.all(a <= b) and np.any(a < b)

        def hv_slice_contributions(points, ref_pt):
            """Exact 2D HV marginal contributions via slice decomposition."""
            n = len(points)
            if n == 0:
                return np.array([]), np.array([], dtype=int)
            order = np.argsort(points[:, 0])
            pts = points[order]
            contribs = np.zeros(n)
            prev_y = ref_pt[1]
            for i in range(n):
                x_i, y_i = pts[i]
                next_x = pts[i + 1, 0] if i + 1 < n else ref_pt[0]
                width = next_x - x_i
                height = prev_y - y_i
                if height < 0:
                    height = 0.0
                contribs[i] = width * height
                prev_y = y_i
            return contribs, order

        def update_archive(archive_X, archive_F, x_new, f_new, max_size=40):
            """Insert point, remove dominated, prune by HV contribution if over max_size."""
            # Check if dominated by any archive member
            for f in archive_F:
                if dominates(f, f_new):
                    return archive_X, archive_F
            # Remove points dominated by new point
            new_X = []
            new_F = []
            for x, f in zip(archive_X, archive_F):
                if not dominates(f_new, f):
                    new_X.append(x)
                    new_F.append(f)
            new_X.append(x_new)
            new_F.append(f_new)
            # Prune if over limit
            if len(new_F) > max_size:
                F_arr = np.array(new_F)
                contribs, order = hv_slice_contributions(F_arr, ref)
                min_idx = np.argmin(contribs)
                remove_idx = order[min_idx]
                new_X.pop(remove_idx)
                new_F.pop(remove_idx)
            return new_X, new_F

        # Phase 1: Deterministic grid initialization aligned to reference point
        # The reference point is [3000, 0.06], so f1 range is large, f2 range is small.
        # Use a grid that covers the decision space uniformly but denser in regions
        # likely to produce diverse objective values.
        n_grid = 60
        # Create a stratified grid in the 4D decision space
        # Use a 3x3x2x2 = 36 grid plus some extra points
        grid_points = []
        # 3 levels per dimension for first 2 dims, 2 for last 2
        levels = [3, 3, 2, 2]
        for d in range(dim):
            pass
        # Generate grid using meshgrid-like approach
        # 3 * 3 * 2 * 2 = 36 points
        grid_X = []
        for i1 in range(3):
            for i2 in range(3):
                for i3 in range(2):
                    for i4 in range(2):
                        x = np.zeros(dim)
                        x[0] = lo[0] + (hi[0] - lo[0]) * (i1 + 0.5) / 3
                        x[1] = lo[1] + (hi[1] - lo[1]) * (i2 + 0.5) / 3
                        x[2] = lo[2] + (hi[2] - lo[2]) * (i3 + 0.5) / 2
                        x[3] = lo[3] + (hi[3] - lo[3]) * (i4 + 0.5) / 2
                        grid_X.append(x)
        
        # Add center point
        grid_X.append((lo + hi) / 2)
        
        # Add a few random points to break symmetry
        n_rand_init = n_grid - len(grid_X)
        if n_rand_init > 0:
            rand_X = lo + (hi - lo) * np.random.rand(n_rand_init, dim)
            grid_X.extend(rand_X.tolist())
        
        grid_X = np.array(grid_X)
        n_init = len(grid_X)
        F_init = np.array([problem(x) for x in grid_X])
        used = n_init

        # Build initial archive
        archive_X = []
        archive_F = []
        for i in range(n_init):
            archive_X, archive_F = update_archive(archive_X, archive_F, grid_X[i], F_init[i], max_size=40)

        # Phase 2: Hypervolume-guided adaptive DE
        # Focus on the high-value region where f1 is small (dominates HV)
        # Dynamically shrink mutation factor to exploit promising regions
        while used < budget:
            progress = used / budget
            # Mutation factor: start wide (1.0) for exploration, shrink to 0.15 for exploitation
            F_mut = 1.0 - 0.85 * progress

            n_archive = len(archive_F)
            if n_archive == 0:
                x_child = lo + (hi - lo) * np.random.rand(dim)
            elif n_archive == 1:
                x_parent = archive_X[0]
                x_child = x_parent + 0.05 * (hi - lo) * np.random.standard_normal(dim)
            else:
                # Weighted selection: favor points with lower f1 (higher HV contribution)
                # Compute f1 values for weighting
                f1_vals = np.array([f[0] for f in archive_F])
                # Inverse weighting: lower f1 gets higher weight
                f1_min = np.min(f1_vals)
                f1_max = np.max(f1_vals)
                if f1_max > f1_min:
                    weights = 1.0 / (f1_vals - f1_min + 1e-8)
                else:
                    weights = np.ones(n_archive)
                weights = weights / np.sum(weights)

                # Pick 3 distinct parents with weighted sampling
                idxs = np.random.choice(n_archive, size=3, replace=False, p=weights)
                p1 = np.array(archive_X[idxs[0]])
                p2 = np.array(archive_X[idxs[1]])
                p3 = np.array(archive_X[idxs[2]])
                x_parent = p1 + F_mut * (p2 - p3)

                # Perturbation: shrink over time
                sigma = 0.08 * (1.0 - 0.7 * progress)
                x_child = x_parent + sigma * (hi - lo) * np.random.standard_normal(dim)

            x_child = np.clip(x_child, lo, hi)
            f_child = problem(x_child)
            used += 1

            archive_X, archive_F = update_archive(archive_X, archive_F, x_child, f_child, max_size=40)

        if archive_X:
            return np.array(archive_X)
        else:
            return np.array([lo + (hi - lo) * 0.5])

# EVOLVE-BLOCK-END
