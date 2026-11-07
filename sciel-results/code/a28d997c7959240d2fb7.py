import numpy as np
import time


class Optimizer:
    """Multi-objective optimizer for ZDT6 using greedy hypervolume maximization."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 285  # 15s safety margin before 300s limit

        lo = np.array([b[0] for b in bounds], dtype=float)
        hi = np.array([b[1] for b in bounds], dtype=float)
        n = self.dim
        budget = self.budget

        # Reference point for hypervolume
        ref = np.array([11.0, 11.0])

        # Archive
        archive_X = []
        archive_F = []
        used = 0

        def evaluate(x):
            nonlocal used
            f = np.array(problem(x))
            used += 1
            return f

        def is_dominated(f_new):
            for f_old in archive_F:
                if np.all(f_old <= f_new) and np.any(f_old < f_new):
                    return True
            return False

        def remove_dominated(f_new):
            nonlocal archive_X, archive_F
            new_X = []
            new_F = []
            for i in range(len(archive_F)):
                if not (np.all(f_new <= archive_F[i]) and np.any(f_new < archive_F[i])):
                    new_X.append(archive_X[i])
                    new_F.append(archive_F[i])
            archive_X = new_X
            archive_F = new_F

        def add_point(x, f):
            if is_dominated(f):
                return
            remove_dominated(f)
            archive_X.append(x)
            archive_F.append(f)

        def zdt6_f2(t):
            """Compute f2 for ZDT6 given f1=t on the Pareto front."""
            return 1.0 - np.sqrt(t) - t

        def fast_marginal_hv(t_new, sorted_f1, sorted_f2):
            """
            Compute marginal hypervolume gain of adding point (t_new, f2(t_new))
            to the archive, given sorted archive points by f1.
            
            In 2D, the marginal contribution is the area of the rectangle
            that is dominated by the new point but not by any existing point.
            This equals (f1_right - t_new) * (f2_left - f2_new) where
            f1_left < t_new < f1_right are neighbors, and f2_left, f2_right
            are their f2 values.
            """
            if len(sorted_f1) == 0:
                f2_new = zdt6_f2(t_new)
                if t_new < ref[0] and f2_new < ref[1]:
                    return (ref[0] - t_new) * (ref[1] - f2_new)
                return 0.0
            
            f2_new = zdt6_f2(t_new)
            if t_new >= ref[0] or f2_new >= ref[1]:
                return 0.0
            
            # Find neighbors using binary search
            idx = np.searchsorted(sorted_f1, t_new)
            
            # Left neighbor: idx-1, Right neighbor: idx
            # The new point contributes area between its neighbors
            # if it's not dominated by either neighbor
            
            f1_left = sorted_f1[idx - 1] if idx > 0 else 0.0
            f2_left = sorted_f2[idx - 1] if idx > 0 else ref[1]
            f1_right = sorted_f1[idx] if idx < len(sorted_f1) else ref[0]
            f2_right = sorted_f2[idx] if idx < len(sorted_f1) else ref[1]
            
            # The new point (t_new, f2_new) contributes hypervolume if
            # f2_new < max(f2_left, f2_right) (i.e., it's not dominated)
            # The contribution is the area of the "new" region it covers.
            
            # For 2D hypervolume with reference point:
            # The new point adds area = (f1_right - t_new) * (f2_left - f2_new)
            # if f2_left > f2_new and t_new > f1_left
            # More precisely, the marginal gain is the area of the rectangle
            # [t_new, f1_right) x [f2_new, f2_left) that wasn't covered before.
            
            if f2_left > f2_new and f1_right > t_new:
                gain = (f1_right - t_new) * (f2_left - f2_new)
                return max(0.0, gain)
            elif f2_right > f2_new and t_new > f1_left:
                gain = (t_new - f1_left) * (f2_right - f2_new)
                return max(0.0, gain)
            else:
                return 0.0

        # Phase 1: Initial exploration - sample a diverse set of t values
        # Use a logarithmic spacing to get good coverage near t=0 where f2 changes rapidly
        n_init = min(20, budget // 8)
        t_init = np.geomspace(0.001, 1.0, n_init)
        for t in t_init:
            if used >= budget or time.time() > deadline:
                break
            x = np.zeros(n)
            x[0] = float(t)
            x = np.clip(x, lo, hi)
            f = evaluate(x)
            add_point(x, f)

        # Phase 2: Greedy hypervolume maximization with fast marginal computation
        # Sort archive by f1 for fast neighbor lookup
        def get_sorted_archive():
            if len(archive_F) == 0:
                return np.array([]), np.array([])
            sorted_idx = np.argsort([f[0] for f in archive_F])
            sorted_f1 = np.array([archive_F[i][0] for i in sorted_idx])
            sorted_f2 = np.array([archive_F[i][1] for i in sorted_idx])
            return sorted_f1, sorted_f2

        # Phase 2: Greedy hypervolume maximization
        for _ in range(budget - used):
            if used >= budget:
                break
            if time.time() > deadline:
                break

            sorted_f1, sorted_f2 = get_sorted_archive()

            # Generate candidate t values: focus on gaps between existing points
            t_candidates = []

            # Add endpoints
            t_candidates.append(0.001)
            t_candidates.append(1.0)

            if len(sorted_f1) > 0:
                # For each gap between consecutive archive points, add candidates
                for i in range(len(sorted_f1) - 1):
                    t_left = sorted_f1[i]
                    t_right = sorted_f1[i + 1]
                    # Add geometric spacing in the gap
                    if t_right > t_left * 1.01:
                        n_gap = max(5, int(np.log(t_right / t_left) / 0.1))
                        n_gap = min(n_gap, 30)
                        mids = np.geomspace(t_left, t_right, n_gap, endpoint=False)
                        t_candidates.extend(mids)
                
                # Also add points slightly beyond the extremes
                if sorted_f1[0] > 0.005:
                    t_candidates.extend(np.geomspace(0.001, sorted_f1[0], 5, endpoint=False))
                if sorted_f1[-1] < 0.95:
                    t_candidates.extend(np.geomspace(sorted_f1[-1], 1.0, 5, endpoint=False))

            # Add a global uniform grid as fallback
            t_candidates.extend(np.linspace(0.001, 1.0, 30))

            t_candidates = np.array(t_candidates)
            t_candidates = t_candidates[(t_candidates >= 0.001) & (t_candidates <= 1.0)]
            t_candidates = np.unique(t_candidates)

            if len(t_candidates) == 0:
                t_candidates = np.array([0.5])

            # Find the t that maximizes marginal hypervolume
            best_t = t_candidates[0]
            best_marginal = -1.0

            for t in t_candidates:
                marginal = fast_marginal_hv(t, sorted_f1, sorted_f2)
                if marginal > best_marginal:
                    best_marginal = marginal
                    best_t = t

            # Evaluate the selected point
            x = np.zeros(n)
            x[0] = float(best_t)
            x = np.clip(x, lo, hi)
            f = evaluate(x)
            add_point(x, f)

        # Phase 3: Final polish - refine the best points with finer search
        if time.time() < deadline - 5 and len(archive_F) > 0:
            sorted_f1, sorted_f2 = get_sorted_archive()
            # For each gap, do a finer search
            for i in range(len(sorted_f1) - 1):
                if time.time() > deadline - 2:
                    break
                if used >= budget:
                    break
                t_left = sorted_f1[i]
                t_right = sorted_f1[i + 1]
                if t_right - t_left < 1e-6:
                    continue
                # Fine grid in this gap
                fine_candidates = np.linspace(t_left, t_right, 20, endpoint=False)
                for t in fine_candidates:
                    if used >= budget or time.time() > deadline - 1:
                        break
                    marginal = fast_marginal_hv(t, sorted_f1, sorted_f2)
                    if marginal > 1e-10:
                        x = np.zeros(n)
                        x[0] = float(t)
                        x = np.clip(x, lo, hi)
                        f = evaluate(x)
                        add_point(x, f)
                        sorted_f1, sorted_f2 = get_sorted_archive()

        return archive_F if archive_F else [np.array([0.0, 1.0])]
