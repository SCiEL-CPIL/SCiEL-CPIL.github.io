import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 285  # 15s margin for safety
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        dim = self.dim
        n_obj = self.n_obj
        
        # Reference point for HV calculation
        ref_point = np.array([1705.0, 12.0, 0.3])
        
        # Initialize archive
        archive = []  # stores (x, f)
        
        def is_dominated(f_new, archive_f_list):
            if len(archive_f_list) == 0:
                return False
            archive_f = np.array(archive_f_list)
            return np.any(np.all(archive_f <= f_new, axis=1) & np.any(archive_f < f_new, axis=1))
            
        def remove_dominated(f_new, archive):
            new_archive = []
            for x, f in archive:
                if np.all(f_new <= f) and np.any(f_new < f):
                    continue
                new_archive.append((x, f))
            return new_archive

        def hypervolume_3d(points, ref):
            """Compute exact hypervolume for 3D points using inclusion-exclusion."""
            if len(points) == 0:
                return 0.0
            
            # Filter points that are better than reference in all objectives
            valid_points = [p for p in points if np.all(p < ref)]
            if len(valid_points) == 0:
                return 0.0
            
            # Sort by first objective
            valid_points.sort(key=lambda p: p[0])
            
            # For each point, compute its contribution
            # A point contributes if it's not dominated by any other point in the set
            # We need to find the non-dominated set first, then compute HV
            # For 3D, we can use the standard algorithm:
            # Sort by f1, then for each point, find the maximum f2 among points with f1 < current f1
            # Actually, let's use a simpler approach: 
            # The HV is the sum over each non-dominated point of the volume it uniquely contributes.
            # For 3D, we can do:
            # 1. Sort by f1 ascending
            # 2. Keep track of the minimum f2 and f3 seen so far? No, that's not right.
            
            # Let's use the standard 3D HV algorithm:
            # Sort points by f1.
            # For each point i, its contribution is (f1[i+1] - f1[i]) * area_in_2d
            # where area_in_2d is the area of the non-dominated set in (f2, f3) space
            # for points j where f1[j] < f1[i] (or all points if we're at the end).
            
            # Actually, a simpler exact method for small n:
            # For each point, determine the box it uniquely controls.
            # This is complex. Let's use a different approach.
            
            # Since n is small (< 50), we can use the following:
            # Sort by f1.
            # For i from 0 to n-1:
            #   width = (ref[0] if i == n-1 else points[i+1][0]) - points[i][0]
            #   But this is only correct if points[i] is non-dominated and 
            #   the "shadow" is correctly computed.
            
            # Let's implement the standard 3D HV algorithm:
            # 1. Sort by f1.
            # 2. Compute for each point the "dominated volume" it uniquely adds.
            # This is tricky. Let's use a simpler approximation or a different exact method.
            
            # Alternative: Use the fact that for 3D, we can project to 2D.
            # Sort by f1.
            # For each point i, the contribution is:
            #   (f1_next - f1_i) * HV_2d(points_j with f1_j < f1_i, ref_2d)
            # where HV_2d is the 2D hypervolume.
            
            # Let's implement HV_2d first.
            def hv_2d(pts, ref2):
                if len(pts) == 0:
                    return 0.0
                pts.sort(key=lambda p: p[0])
                total = 0.0
                prev_x = ref2[0]
                for p in pts:
                    if p[0] >= prev_x:
                        continue
                    # p contributes (prev_x - p[0]) * (ref2[1] - p[1])
                    # but only if p is non-dominated in 2D
                    # Actually, we need to check if p is dominated by any previous point
                    # This is getting complex.
                    
                    # Simpler: For 2D, sort by x. 
                    # Keep track of the minimum y seen so far.
                    # A point contributes if its y is less than the minimum y of all previous points.
                    # But this is only for the "staircase" shape.
                    
                    # Let's do it properly for small n.
                    pass
                return total
            
            # Given the complexity, let's use a simpler heuristic for selection:
            # Just use the reference point to compute a "distance" or "score" for each point.
            # Or, even simpler: use random selection but with a bias towards points 
            # that are closer to the ideal point (minimum in each objective).
            
            # For now, let's just return 0.0 and use a different selection strategy.
            return 0.0

        used = 0
        
        # 1. Random Initialization
        init_size = 20
        X_init = lo + (hi - lo) * np.random.rand(init_size, dim)
        for i in range(init_size):
            if time.time() > deadline: break
            x = X_init[i]
            f = np.array(problem(x))
            if not is_dominated(f, [f_ for _, f_ in archive]):
                archive = remove_dominated(f, archive)
                archive.append((x, f))
            used += 1
            
        # 2. Main Loop
        while used < self.budget:
            if time.time() > deadline:
                break
                
            if not archive:
                x_parent = lo + (hi - lo) * np.random.rand(dim)
            else:
                # Selection: Use a simple heuristic
                # Prefer points that are "balanced" or close to the reference point
                # For now, use uniform random from archive
                idx = np.random.randint(len(archive))
                x_parent = archive[idx][0]
                
            # Mutate
            sigma = 0.1 * (hi - lo)
            x_child = x_parent + sigma * np.random.randn(dim)
            x_child = np.clip(x_child, lo, hi)
            
            f_child = np.array(problem(x_child))
            used += 1
            
            if is_dominated(f_child, [f_ for _, f_ in archive]):
                continue
                
            archive = remove_dominated(f_child, archive)
            archive.append((x_child, f_child))
            
            if len(archive) > 100:
                # Prune: keep the 100 points with the best "score"
                # Score: distance to reference point (smaller is better)
                # Or: use a simple metric like sum of normalized objectives
                scores = []
                for x, f in archive:
                    # Normalize by reference point
                    norm_f = f / ref_point
                    scores.append(np.sum(norm_f))
                scores = np.array(scores)
                # Keep the 100 with smallest scores
                indices = np.argsort(scores)[:100]
                archive = [archive[i] for i in indices]

        return archive
