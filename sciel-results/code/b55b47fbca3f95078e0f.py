"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Hypervolume-driven multi-objective optimizer for RE21."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def _hv_2d(self, front, ref):
        """Compute exact 2D hypervolume for a list of points (Pareto front)."""
        if len(front) == 0:
            return 0.0
        pts = np.array(front, dtype=float)
        mask = np.all(pts < ref, axis=1)
        pts = pts[mask]
        if len(pts) == 0:
            return 0.0
        order = np.argsort(pts[:, 0])
        pts = pts[order]
        front = [pts[0]]
        for p in pts[1:]:
            if p[1] < front[-1][1]:
                front.append(p)
        if len(front) == 0:
            return 0.0
        hv = 0.0
        for i in range(len(front)):
            x_next = front[i + 1][0] if i + 1 < len(front) else ref[0]
            y_curr = front[i][1]
            if y_curr >= ref[1]:
                continue
            width = x_next - front[i][0]
            height = ref[1] - y_curr
            hv += width * height
        return hv

    def _hv_contribution_2d(self, point, front, ref):
        """Compute HV contribution of a single point in a 2D front."""
        if len(front) == 0:
            return 0.0
        p = np.array(point, dtype=float)
        if not np.all(p < ref):
            return 0.0
        
        pts = np.array(front, dtype=float)
        mask = np.all(pts < ref, axis=1)
        pts = pts[mask]
        if len(pts) == 0:
            # Point is the only one; contribution is its full rectangle
            return (ref[0] - p[0]) * (ref[1] - p[1])
        
        order = np.argsort(pts[:, 0])
        pts = pts[order]
        
        nd_front = [pts[0]]
        for pt in pts[1:]:
            if pt[1] < nd_front[-1][1]:
                nd_front.append(pt)
        
        if len(nd_front) == 0:
            return 0.0
        
        # Find where p would be inserted
        # p dominates points to its right if p[1] < their y
        # Points to left dominate p if their x < p[0] and y <= p[1]
        
        # Check if p is dominated by existing front
        for fp in nd_front:
            if fp[0] <= p[0] and fp[1] <= p[1] and (fp[0] < p[0] or fp[1] < p[1]):
                return 0.0
        
        # Compute HV with and without p
        all_pts = np.vstack([np.array(nd_front), p.reshape(1, -1)])
        order_all = np.argsort(all_pts[:, 0])
        all_sorted = all_pts[order_all]
        
        # Build front from all_sorted
        front_with = [all_sorted[0]]
        for pt in all_sorted[1:]:
            if pt[1] < front_with[-1][1]:
                front_with.append(pt)
        
        hv_with = 0.0
        for i in range(len(front_with)):
            x_next = front_with[i + 1][0] if i + 1 < len(front_with) else ref[0]
            y_curr = front_with[i][1]
            if y_curr >= ref[1]:
                continue
            width = x_next - front_with[i][0]
            height = ref[1] - y_curr
            hv_with += width * height
        
        hv_without = self._hv_2d(nd_front, ref)
        
        return max(0.0, hv_with - hv_without)

    def _non_dominated_front(self, F):
        """Return indices of non-dominated points in F (2D minimization)."""
        n = len(F)
        if n == 0:
            return np.array([], dtype=int)
        F = np.array(F, dtype=float)
        dominated = np.zeros(n, dtype=bool)
        for i in range(n):
            if dominated[i]:
                continue
            for j in range(n):
                if i == j:
                    continue
                if np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                    dominated[i] = True
                    break
        return np.where(~dominated)[0]

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 290.0  # 10s safety margin before 300s limit

        lo = np.array([b[0] for b in bounds], dtype=float)
        hi = np.array([b[1] for b in bounds], dtype=float)
        ref = np.array([3000.0, 0.06])
        rng = np.random

        # Archive-based approach: maintain Pareto front
        # Initialize with diverse points
        init_size = min(30, self.budget // 6)
        
        X_archive = []
        F_archive = []
        
        # Structured initialization
        X_archive.append(lo.copy())
        F_archive.append(problem(lo.copy()))
        
        X_archive.append(hi.copy())
        F_archive.append(problem(hi.copy()))
        
        X_archive.append((lo + hi) / 2.0)
        F_archive.append(problem((lo + hi) / 2.0))
        
        # Add mid-edge points
        for i in range(self.dim):
            for j in range(i + 1, self.dim):
                pt = (lo + hi) / 2.0
                pt[i] = lo[i]
                pt[j] = hi[j]
                X_archive.append(pt.copy())
                F_archive.append(problem(pt.copy()))
                
                pt2 = (lo + hi) / 2.0
                pt2[i] = hi[i]
                pt2[j] = lo[j]
                X_archive.append(pt2.copy())
                F_archive.append(problem(pt2.copy()))
        
        # Fill with random points
        while len(X_archive) < init_size:
            x = lo + (hi - lo) * rng.random(self.dim)
            X_archive.append(x)
            F_archive.append(problem(x))
        
        used = len(X_archive)
        
        # Track all evaluations for return
        all_X = list(X_archive)
        all_F = list(F_archive)
        
        # Main optimization loop
        while used < self.budget:
            if time.time() > deadline:
                break
            
            # Get current Pareto front
            front_idx = self._non_dominated_front(F_archive)
            if len(front_idx) == 0:
                # All dominated; use best point
                best_idx = np.argmin([np.sum(F_archive[i]) for i in range(len(F_archive))])
                front_idx = np.array([best_idx])
            
            front_points = [F_archive[i] for i in front_idx]
            front_X = [X_archive[i] for i in front_idx]
            
            # Compute HV contributions
            contributions = []
            for k in range(len(front_idx)):
                contrib = self._hv_contribution_2d(front_points[k], front_points, ref)
                contributions.append(contrib)
            
            contributions = np.array(contributions)
            total_contrib = np.sum(contributions)
            
            if total_contrib > 1e-10:
                probs = contributions / total_contrib
            else:
                probs = np.ones(len(front_idx)) / len(front_idx)
            
            # Select parent based on HV contribution
            if len(front_idx) >= 2:
                # Weighted selection
                parent_k = rng.choice(len(front_idx), p=probs)
                parent_X = front_X[parent_k].copy()
                parent_F = front_points[parent_k].copy()
                
                # Adaptive mutation scale based on front spread
                if len(front_points) >= 2:
                    x_spread = np.max([p[0] for p in front_points]) - np.min([p[0] for p in front_points])
                    y_spread = np.max([p[1] for p in front_points]) - np.min([p[1] for p in front_points])
                    # Scale mutation with front spread (normalized)
                    scale_factor = 0.1 + 0.05 * np.log1p(used / 10.0)
                else:
                    scale_factor = 0.15
                
                # Gaussian mutation
                mutation = rng.standard_normal(self.dim) * scale_factor * (hi - lo)
                child = parent_X + mutation
                child = np.clip(child, lo, hi)
            else:
                # Single point: explore around it
                parent_X = front_X[0].copy()
                scale_factor = 0.2
                mutation = rng.standard_normal(self.dim) * scale_factor * (hi - lo)
                child = parent_X + mutation
                child = np.clip(child, lo, hi)
            
            fc = problem(child)
            used += 1
            
            all_X.append(child)
            all_F.append(fc)
            
            # Add to archive if it improves the Pareto front
            F_arr = np.array(F_archive)
            # Check if child is dominated by any archive point
            dominated_by_archive = False
            for i in range(len(F_archive)):
                if np.all(F_archive[i] <= fc) and np.any(F_archive[i] < fc):
                    dominated_by_archive = True
                    break
            
            if not dominated_by_archive:
                # Add to archive
                X_archive.append(child)
                F_archive.append(fc)
                
                # Remove dominated points from archive
                new_archive_X = []
                new_archive_F = []
                for i in range(len(X_archive)):
                    # Check if archive point i is dominated by child or other archive points
                    is_dominated = False
                    for j in range(len(X_archive)):
                        if i == j:
                            continue
                        if np.all(F_archive[j] <= F_archive[i]) and np.any(F_archive[j] < F_archive[i]):
                            is_dominated = True
                            break
                    if not is_dominated:
                        new_archive_X.append(X_archive[i])
                        new_archive_F.append(F_archive[i])
                
                X_archive = new_archive_X
                F_archive = new_archive_F
            
            # Periodic random exploration to maintain diversity
            if used % 25 == 0 and used < self.budget - 3:
                for _ in range(2):
                    if used >= self.budget:
                        break
                    random_child = lo + (hi - lo) * rng.random(self.dim)
                    rc = problem(random_child)
                    used += 1
                    all_X.append(random_child)
                    all_F.append(rc)
                    
                    # Add to archive if not dominated
                    dominated_by_archive = False
                    for i in range(len(F_archive)):
                        if np.all(F_archive[i] <= rc) and np.any(F_archive[i] < rc):
                            dominated_by_archive = True
                            break
                    
                    if not dominated_by_archive:
                        X_archive.append(random_child)
                        F_archive.append(rc)
                        
                        # Remove dominated points
                        new_archive_X = []
                        new_archive_F = []
                        for i in range(len(X_archive)):
                            is_dominated = False
                            for j in range(len(X_archive)):
                                if i == j:
                                    continue
                                if np.all(F_archive[j] <= F_archive[i]) and np.any(F_archive[j] < F_archive[i]):
                                    is_dominated = True
                                    break
                            if not is_dominated:
                                new_archive_X.append(X_archive[i])
                                new_archive_F.append(F_archive[i])
                        
                        X_archive = new_archive_X
                        F_archive = new_archive_F

        return all_X
# EVOLVE-BLOCK-END
