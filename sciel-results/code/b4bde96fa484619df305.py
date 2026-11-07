# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-objective optimizer for WFG7 using hypervolume-driven search."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def _calc_hv_3d(self, points, ref):
        """
        Calculate the exact hypervolume of a set of points in 3D.
        points: Nx3 array
        ref: 3-array
        """
        if len(points) == 0:
            return 0.0
        
        # Filter points that are better than ref in all objectives
        mask = np.all(points < ref, axis=1)
        pts = points[mask]
        
        if len(pts) == 0:
            return 0.0
            
        # Sort by first objective (f0)
        idx = np.argsort(pts[:, 0])
        pts = pts[idx]
        
        def area_2d(points_2d, ref_2d):
            # points_2d: Mx2 array (f1, f2)
            # ref_2d: 2-array (ref1, ref2)
            # Calculate area of union of rectangles [p1, ref1] x [p2, ref2]
            if len(points_2d) == 0:
                return 0.0
            
            # Sort by f1
            idx2 = np.argsort(points_2d[:, 0])
            p2 = points_2d[idx2]
            
            total_area = 0.0
            active_intervals = []
            
            for i in range(len(p2)):
                active_intervals.append((p2[i][1], ref_2d[1]))
                
                if i == len(p2) - 1:
                    width = ref_2d[0] - p2[i][0]
                else:
                    width = p2[i+1][0] - p2[i][0]
                
                if not active_intervals:
                    h = 0.0
                else:
                    active_intervals.sort(key=lambda x: x[0])
                    merged = []
                    for interval in active_intervals:
                        if not merged:
                            merged.append(list(interval))
                        else:
                            last = merged[-1]
                            if interval[0] <= last[1]:
                                last[1] = max(last[1], interval[1])
                            else:
                                merged.append(list(interval))
                    h = sum(end - start for start, end in merged)
                
                total_area += width * h
                
            return total_area

        total_hv = 0.0
        ref_2d = ref[1:]
        
        for i in range(len(pts)):
            proj = pts[0:i+1, 1:]
            a = area_2d(proj, ref_2d)
            
            if i == len(pts) - 1:
                width = ref[0] - pts[i][0]
            else:
                width = pts[i+1][0] - pts[i][0]
            
            total_hv += width * a
            
        return total_hv

    def _is_dominated(self, f, F, exclude_idx):
        """Check if point f is dominated by any point in F (excluding exclude_idx)."""
        for i in range(len(F)):
            if i == exclude_idx:
                continue
            if np.all(F[i] <= f) and np.any(F[i] < f):
                return True
        return False

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 290.0

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        width = hi - lo

        dim = self.dim
        n_obj = self.n_obj
        budget = self.budget

        # Reference point for hypervolume (scoring reference)
        ref = np.array([3.0, 5.0, 7.0])

        # --- Initial population ---
        pop_size = 20
        X = lo + width * np.random.random((pop_size, dim))
        F = np.array([problem(x) for x in X])
        used = pop_size

        # --- Main search loop ---
        mut_scale = 0.2

        while used < budget:
            if time.time() > deadline - 30.0:
                break

            # Select parent: Random from population
            parent_idx = np.random.randint(0, len(X))
            parent = X[parent_idx].copy()

            # Mutation
            if np.random.random() < 0.1:
                # Large jump
                child = parent + width * np.random.standard_normal(dim) * 0.5
            else:
                child = parent + width * np.random.standard_normal(dim) * mut_scale
            
            child = np.clip(child, lo, hi)
            fc = problem(child)
            used += 1

            # Add to population
            X = np.vstack([X, child])
            F = np.vstack([F, fc])

            # If population exceeds size, remove dominated points first, then worst HV contributor
            if len(X) > pop_size:
                # First pass: remove any point that is dominated by another
                to_remove = []
                for i in range(len(X)):
                    if self._is_dominated(F[i], F, i):
                        to_remove.append(i)
                
                if to_remove:
                    mask = np.ones(len(X), dtype=bool)
                    mask[to_remove] = False
                    X = X[mask]
                    F = F[mask]
                
                # If still too large, use HV-based removal (expensive but rare)
                if len(X) > pop_size:
                    hv_total = self._calc_hv_3d(F, ref)
                    
                    min_contrib = float('inf')
                    min_idx = -1
                    
                    for i in range(len(X)):
                        mask = np.ones(len(X), dtype=bool)
                        mask[i] = False
                        F_without_i = F[mask]
                        hv_without_i = self._calc_hv_3d(F_without_i, ref)
                        contrib = hv_total - hv_without_i
                        
                        if contrib < min_contrib:
                            min_contrib = contrib
                            min_idx = i
                    
                    mask = np.ones(len(X), dtype=bool)
                    mask[min_idx] = False
                    X = X[mask]
                    F = F[mask]

            # Adapt mutation scale
            if used % 20 == 0:
                mut_scale *= 0.95
                mut_scale = max(mut_scale, 0.02)

        # --- Final polish phase: refine non-dominated points with local search ---
        remaining_budget = budget - used
        if remaining_budget > 0 and time.time() < deadline - 5.0:
            # Identify non-dominated points
            non_dom_idx = []
            for i in range(len(F)):
                if not self._is_dominated(F[i], F, i):
                    non_dom_idx.append(i)
            
            # Polish each non-dominated point with a few tight local steps
            steps_per_point = min(remaining_budget // max(len(non_dom_idx), 1), 10)
            if steps_per_point > 0:
                for idx in non_dom_idx:
                    if used >= budget or time.time() > deadline - 2.0:
                        break
                    
                    best_x = X[idx].copy()
                    best_f = F[idx].copy()
                    
                    for step in range(steps_per_point):
                        if used >= budget or time.time() > deadline - 2.0:
                            break
                        
                        # Try several small perturbations
                        best_trial = None
                        for _ in range(3):
                            trial = best_x + width * np.random.standard_normal(dim) * 0.05
                            trial = np.clip(trial, lo, hi)
                            tf = problem(trial)
                            used += 1
                            
                            # Check if this trial is better (non-dominated and improves HV)
                            # Simple heuristic: if it's not dominated by best_f and has lower sum
                            if np.any(tf < best_f) and not np.any(tf > best_f):
                                best_x = trial
                                best_f = tf
                                break
                            elif np.sum(tf) < np.sum(best_f) and not self._is_dominated(tf, F, -1):
                                if best_trial is None or np.sum(tf) < np.sum(best_trial[1]):
                                    best_trial = (trial, tf)
                        
                        if best_trial is not None:
                            best_x = best_trial[0]
                            best_f = best_trial[1]
                        elif np.all(best_f == F[idx]):
                            # No improvement, stop early
                            break
                    
                    # Update the point in population
                    X[idx] = best_x
                    F[idx] = best_f
        
        return X
# EVOLVE-BLOCK-END
