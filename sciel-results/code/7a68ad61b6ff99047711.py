"""Rewritten optimizer for WFG9 using a differential-evolution based MOEA with adaptive mutation."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def _hv_3d(points, ref):
    """Compute exact hypervolume for a set of 3D points against a reference point."""
    if len(points) == 0:
        return 0.0
    mask = np.all(points < ref, axis=1)
    pts = points[mask]
    if len(pts) == 0:
        return 0.0
    pts = np.array(pts)
    idx = np.argsort(pts[:, 0])
    pts = pts[idx]
    n = len(pts)
    if n == 1:
        p = pts[0]
        return max(0.0, (ref[0] - p[0]) * (ref[1] - p[1]) * (ref[2] - p[2]))
    
    def hv_2d(pts2d, ref2d):
        if len(pts2d) == 0:
            return 0.0
        mask = np.all(pts2d < ref2d, axis=1)
        pts2d = pts2d[mask]
        if len(pts2d) == 0:
            return 0.0
        idx = np.argsort(pts2d[:, 0])
        pts2d = pts2d[idx]
        n2 = len(pts2d)
        y_min = np.full(n2, np.inf)
        y_min[-1] = pts2d[-1, 1]
        for i in range(n2 - 2, -1, -1):
            y_min[i] = min(pts2d[i, 1], y_min[i + 1])
        hv2 = 0.0
        for i in range(n2 - 1):
            if y_min[i] < ref2d[1]:
                hv2 += (pts2d[i + 1, 0] - pts2d[i, 0]) * (ref2d[1] - y_min[i])
        if y_min[-1] < ref2d[1]:
            hv2 += (ref2d[0] - pts2d[-1, 0]) * (ref2d[1] - y_min[-1])
        return hv2
    
    hv3 = 0.0
    for i in range(n - 1):
        suffix_yz = pts[i:, 1:]
        hv3 += (pts[i + 1, 0] - pts[i, 0]) * hv_2d(suffix_yz, ref[1:])
    if n > 0:
        suffix_yz = pts[-1:, 1:]
        hv3 += (ref[0] - pts[-1, 0]) * hv_2d(suffix_yz, ref[1:])
    return hv3


class Optimizer:
    """Multi-objective optimizer using adaptive differential evolution with hypervolume-guided selection."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 290.0

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        dim = self.dim
        n_obj = self.n_obj
        budget = self.budget

        rng = np.random.default_rng(42)

        # Population size
        pop_size = min(40, budget // 5)
        
        # Initialize population
        X = np.empty((pop_size, dim))
        F = np.empty((pop_size, n_obj))
        for i in range(pop_size):
            X[i] = lo + (hi - lo) * rng.random(dim)
            F[i] = problem(X[i])
        used = pop_size

        # Track all evaluated points
        all_X = X.copy()
        all_F = F.copy()

        # Personal bests (non-dominated in their own history)
        P_best_X = X.copy()
        P_best_F = F.copy()

        # Reference point
        ref = np.array([3.0, 5.0, 7.0])

        def is_dominated(f, archive):
            """Check if f is dominated by any point in archive."""
            for a in archive:
                if np.all(a <= f) and np.any(a < f):
                    return True
            return False

        def update_archive(archive_X, archive_F, x, f, max_size=200):
            """Update archive with new point, removing dominated points."""
            # Check if new point is dominated by any in archive
            dominated = False
            for af in archive_F:
                if np.all(af <= f) and np.any(af < f):
                    dominated = True
                    break
            if dominated:
                return archive_X, archive_F
            
            # Remove points dominated by new point
            new_archive_X = []
            new_archive_F = []
            for ax, af in zip(archive_X, archive_F):
                if not (np.all(f <= af) and np.any(f < af)):
                    new_archive_X.append(ax)
                    new_archive_F.append(af)
            new_archive_X.append(x)
            new_archive_F.append(f)
            
            # If too large, remove dominated points and then diverse points
            if len(new_archive_F) > max_size:
                # Find non-dominated points
                nd_idx = []
                for i, f in enumerate(new_archive_F):
                    if not is_dominated(f, [new_archive_F[j] for j in range(len(new_archive_F)) if j != i]):
                        nd_idx.append(i)
                if len(nd_idx) >= max_size:
                    # Keep max_size non-dominated points with best hypervolume contribution
                    # For simplicity, just keep the first max_size
                    keep_idx = nd_idx[:max_size]
                else:
                    # Keep all non-dominated + some dominated
                    dominated_idx = [i for i in range(len(new_archive_F)) if i not in nd_idx]
                    n_keep = max_size - len(nd_idx)
                    if n_keep > 0 and len(dominated_idx) > 0:
                        rand_dominated = rng.choice(dominated_idx, size=min(n_keep, len(dominated_idx)), replace=False)
                        keep_idx = nd_idx + list(rand_dominated)
                    else:
                        keep_idx = nd_idx
                
                new_archive_X = [new_archive_X[i] for i in keep_idx]
                new_archive_F = [new_archive_F[i] for i in keep_idx]
            
            return np.array(new_archive_X), np.array(new_archive_F)

        # Initialize archive
        archive_X = X.copy()
        archive_F = F.copy()
        # Remove dominated points from initial archive
        nd_idx = []
        for i in range(len(archive_F)):
            if not is_dominated(archive_F[i], [archive_F[j] for j in range(len(archive_F)) if j != i]):
                nd_idx.append(i)
        archive_X = archive_X[nd_idx]
        archive_F = archive_F[nd_idx]

        # Adaptive parameters
        F_cr = 0.8  # Crossover rate
        F_mut = 0.5  # Mutation scale
        adapt_counter = 0
        last_hv = 0.0

        while used < budget:
            if time.time() > deadline:
                break

            # Adaptive parameter adjustment
            adapt_counter += 1
            if adapt_counter % 20 == 0:
                # Compute current hypervolume
                current_hv = _hv_3d(archive_F, ref)
                if current_hv > last_hv:
                    # Improving: keep parameters or slightly increase exploration
                    F_cr = min(1.0, F_cr + 0.05)
                    F_mut = max(0.3, F_mut - 0.05)
                else:
                    # Stagnating: increase exploration
                    F_cr = max(0.5, F_cr - 0.1)
                    F_mut = min(1.0, F_mut + 0.1)
                last_hv = current_hv

            # Generate offspring using DE/rand/1/bin
            for i in range(pop_size):
                if used >= budget:
                    break

                # Select three distinct parents
                while True:
                    r1, r2, r3 = rng.choice(pop_size, size=3, replace=False)
                    if len(set([r1, r2, r3, i])) == 4:
                        break

                # Mutation
                v = X[r1] + F_mut * (X[r2] - X[r3])

                # Crossover
                j_rand = rng.integers(dim)
                for j in range(dim):
                    if rng.random() < F_cr or j == j_rand:
                        v[j] = X[i, j]

                # Clip to bounds
                v = np.clip(v, lo, hi)

                # Evaluate
                f_v = problem(v)
                used += 1

                # Record
                all_X = np.vstack([all_X, v])
                all_F = np.vstack([all_F, f_v])

                # Update personal best
                if not is_dominated(f_v, [P_best_F[i]]):
                    P_best_X[i] = v.copy()
                    P_best_F[i] = f_v.copy()

                # Selection: compare offspring with parent
                if not is_dominated(f_v, [F[i]]):
                    X[i] = v
                    F[i] = f_v
                    # Update archive
                    archive_X, archive_F = update_archive(archive_X, archive_F, v, f_v, max_size=150)

                # Occasionally update population member from personal best
                if rng.random() < 0.1:
                    X[i] = P_best_X[i]
                    F[i] = P_best_F[i]

        return all_F

# EVOLVE-BLOCK-END
