import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        t0 = time.time()
        deadline = t0 + 280
        
        dim = self.dim
        n_obj = self.n_obj
        budget = self.budget
        
        lo = np.array([b[0] for b in bounds], dtype=float)
        hi = np.array([b[1] for b in bounds], dtype=float)
        span = hi - lo
        
        # Reference point
        ref = np.array([3.0, 5.0, 7.0])
        
        # Helper: Calculate Hypervolume (3D)
        # Simple method: sort by f1, then f2, then f3?
        # For 3D, we can use a sweep line or just sum volumes of boxes.
        # Since N is small, we can use a simple O(N^2) or O(N^3) method.
        # Actually, for 3D, we can project to 2D.
        
        def hypervolume_3d(points, ref):
            # points: list of (f1, f2, f3)
            # Filter points better than ref
            pts = [p for p in points if np.all(p < ref)]
            if not pts:
                return 0.0
            
            # Sort by f1
            pts.sort(key=lambda p: p[0])
            
            total_vol = 0.0
            n = len(pts)
            
            # For each point, calculate the volume it contributes
            # This is tricky. A standard algorithm is needed.
            # Let's use a simple inclusion-exclusion or grid?
            # Given small N, we can use a brute force volume calculation?
            # No, let's use the standard 3D HV algorithm.
            # 1. Sort by f1.
            # 2. For each point i, consider the rectangle in f2-f3 plane defined by 
            #    (f2_i, f3_i) and the "skyline" of points with f1 > f1_i.
            
            # This is complex to implement correctly from scratch in a short time.
            # Alternative: Use a library? No, only standard lib, numpy, scipy.
            # SciPy doesn't have a direct HV function.
            
            # Let's implement a simple HV for 3D.
            # Algorithm:
            # 1. Sort points by f1 ascending.
            # 2. Iterate through points. Maintain a list of "active" points in 2D (f2, f3).
            # 3. The contribution of point i is the volume of the box from (f1_i, f2_i, f3_i) to (f1_{i+1}, ref2, ref3) 
            #    minus the volume already covered by points with f1 > f1_i in the f2-f3 plane.
            
            # This is still complex.
            
            # Let's try a simpler approximation or a different scoring metric for selection?
            # No, we need to maximize actual HV.
            
            # Let's use a simple Monte Carlo estimate? No, must be exact.
            
            # Let's implement the 3D HV using the "slice" method.
            # Sort by f1.
            # For each interval [f1_i, f1_{i+1}], the cross-section is a 2D HV problem in (f2, f3)
            # with points from the set {p | p.f1 <= f1_{i+1}}.
            # This is O(N^3) which is fine for N=50.
            
            pts = np.array(pts)
            order = np.argsort(pts[:, 0])
            pts = pts[order]
            
            # Precompute 2D HV for subsets?
            # Let's just iterate.
            
            total = 0.0
            n = len(pts)
            
            # Add a sentinel point at ref
            # Actually, the volume is the union of boxes [p, ref].
            
            # We can use the fact that HV is the volume of the union of boxes.
            # In 3D, we can use the inclusion-exclusion principle? No, too slow.
            
            # Let's use the standard algorithm:
            # Sort by f1.
            # For i from 0 to n-1:
            #   Let f1_range = (f1_{i+1} - f1_i) if i < n-1 else (ref[0] - f1_i)
            #   The 2D region in (f2, f3) is the HV of the points {p_j | j >= i} in the (f2, f3) plane,
            #   but bounded by ref[1], ref[2].
            #   Wait, the points with f1 < f1_i are already "processed" and their volume is added.
            #   The volume added by the slice [f1_i, f1_{i+1}] is f1_range * HV_2D({p_j | j >= i}, ref[1:], ref[2:])
            #   Is this correct?
            #   The box for p_i is [f1_i, ref0] x [f2_i, ref1] x [f3_i, ref2].
            #   The union of boxes is not simply sum of slices because of overlaps in f2, f3.
            #   However, if we sort by f1, the slice [f1_i, f1_{i+1}] is covered by the "skyline" of points with f1 <= f1_{i+1}.
            #   Actually, the volume of the union of boxes up to f1_{i+1} is:
            #   V_{i+1} = V_i + (f1_{i+1} - f1_i) * HV_2D(S_{i+1}, ref2, ref3)
            #   Where S_{i+1} is the set of points with f1 <= f1_{i+1}.
            #   This is correct if we define HV_2D correctly.
            
            def hv_2d(points_2d, ref_2d):
                # points_2d: list of (f2, f3)
                # ref_2d: (ref2, ref3)
                pts2 = [p for p in points_2d if p[0] < ref_2d[0] and p[1] < ref_2d[1]]
                if not pts2:
                    return 0.0
                pts2.sort(key=lambda p: p[0])
                vol = 0.0
                max_f3 = 0.0
                prev_f2 = 0.0
                # This is not quite right.
                # Standard 2D HV:
                # Sort by f2.
                # Keep track of max f3 seen so far (from right to left? No, from left to right, we want the "skyline").
                # Actually, sort by f2 ascending.
                # The contribution of point i is (f2_{i+1} - f2_i) * (ref3 - max(f3 of points j >= i))
                # Wait, if we sort by f2 ascending, the "skyline" in f3 is non-increasing?
                # No, the Pareto front in 2D is a curve.
                # Let's use the standard algorithm:
                # Sort by f2 ascending.
                # Initialize max_f3 = 0.
                # Iterate from right to left (largest f2 to smallest).
                # If f3 > max_f3, then it contributes (f2_current - f2_next) * (ref3 - f3)? No.
                # If f3 > max_f3, it is on the front.
                # The volume is sum of (f2_i - f2_{i+1}) * (ref3 - f3_i) for i on the front.
                
                pts2.sort(key=lambda p: p[0], reverse=True)
                vol = 0.0
                max_f3 = 0.0
                for i in range(len(pts2)):
                    f2, f3 = pts2[i]
                    if f3 > max_f3:
                        # This point is on the skyline
                        # The width is f2 - (f2 of next point in sorted order, or 0 if last)
                        # Since we sorted descending, the "next" point is the one with smaller f2.
                        # The interval is [f2_next, f2].
                        if i == len(pts2) - 1:
                            width = f2
                        else:
                            width = f2 - pts2[i+1][0]
                        vol += width * (ref_2d[1] - f3)
                        max_f3 = f3
                return vol

            # Now back to 3D
            # We need HV_2D of the set of points with f1 <= f1_{i+1}.
            # This is O(N^2) if we recompute HV_2D each time. N=50, 2500 ops. Fine.
            
            for i in range(n):
                # f1 range
                if i < n - 1:
                    f1_range = pts[i+1, 0] - pts[i, 0]
                else:
                    f1_range = ref[0] - pts[i, 0]
                
                # Points with f1 <= pts[i+1, 0] (or ref[0] for last)
                # Actually, the slice is [f1_i, f1_{i+1}].
                # The 2D HV should be computed on the set of points with f1 <= f1_{i+1}.
                # But we are iterating i from 0 to n-1.
                # The set for slice i is {p_j | j >= i} in the sorted list?
                # No, the sorted list is by f1 ascending.
                # So points with f1 <= f1_{i+1} are indices 0 to i.
                # Wait, the volume of the union up to f1_{i+1} is:
                # V_{i+1} = V_i + (f1_{i+1} - f1_i) * HV_2D({p_0, ..., p_i}, ref[1:])
                # Is this correct?
                # The new volume added by the slice [f1_i, f1_{i+1}] is the area in (f2, f3) that is covered by boxes with f1 <= f1_{i+1} but NOT covered by boxes with f1 <= f1_i.
                # This is exactly HV_2D({p_0, ..., p_i}) - HV_2D({p_0, ..., p_{i-1}})?
                # No, the 2D HV is of the "skyline" of the points.
                # The formula V_{i+1} = (f1_{i+1} - f1_i) * HV_2D({p_0, ..., p_i}) is incorrect because it multiplies the full 2D HV by the slice width, but the 2D HV changes.
                
                # Correct formula:
                # The volume of the union of boxes is:
                # Sum_{i} (f1_{i+1} - f1_i) * HV_2D(S_i)
                # Where S_i is the set of points with f1 <= f1_{i+1}.
                # This works because the 2D cross-section at any f1 in [f1_i, f1_{i+1}] is determined by the points with f1 <= f1_{i+1}.
                # And the HV_2D is constant in that interval.
                
                S_i = pts[:i+1]
                hv2 = hv_2d(S_i, ref[1:])
                total += f1_range * hv2
                
            return total

        # Greedy Search
        # 1. Initialize with a few points
        init_size = 10
        X_pop = []
        F_pop = []
        
        # LHS for init
        X_init = np.zeros((init_size, dim))
        for d in range(dim):
            bins = np.linspace(lo[d], hi[d], init_size + 1)
            perm = np.random.permutation(init_size)
            for i in range(init_size):
                low_bin = bins[perm[i]]
                high_bin = bins[perm[i] + 1]
                X_init[i, d] = np.random.uniform(low_bin, high_bin)
        
        used = 0
        for x in X_init:
            if used >= budget: break
            if time.time() > deadline: break
            f = problem(x)
            X_pop.append(x)
            F_pop.append(f)
            used += 1
            
        # Calculate initial HV
        current_hv = hypervolume_3d(F_pop, ref)
        
        # 2. Greedy loop
        while used < budget and time.time() < deadline:
            # Generate candidates
            candidates = []
            
            # 1. Mutation of existing points
            for i in range(len(X_pop)):
                x_mut = X_pop[i] + np.random.randn(dim) * 0.1 * span
                x_mut = np.clip(x_mut, lo, hi)
                candidates.append(x_mut)
                
            # 2. Interpolation
            if len(X_pop) >= 2:
                i, j = np.random.choice(len(X_pop), 2, replace=False)
                alpha = np.random.rand()
                x_interp = alpha * X_pop[i] + (1 - alpha) * X_pop[j]
                candidates.append(x_interp)
                
            # 3. Random
            x_rand = np.random.uniform(lo, hi)
            candidates.append(x_rand)
            
            # Evaluate candidates and pick the one that maximizes HV
            best_candidate = None
            best_hv = current_hv
            best_f = None
            
            for x_c in candidates:
                if used >= budget: break
                if time.time() > deadline: break
                f_c = problem(x_c)
                used += 1
                
                # Check if it improves HV
                # We can check if it's non-dominated first
                # If it's dominated, it doesn't improve HV.
                dominated = False
                for f in F_pop:
                    if np.all(f <= f_c) and np.any(f < f_c):
                        dominated = True
                        break
                if dominated:
                    continue
                    
                # Calculate new HV
                # To save time, we can approximate or just calculate
                new_F = F_pop + [f_c]
                new_hv = hypervolume_3d(new_F, ref)
                
                if new_hv > best_hv:
                    best_hv = new_hv
                    best_candidate = x_c
                    best_f = f_c
                    
            if best_candidate is not None:
                X_pop.append(best_candidate)
                F_pop.append(best_f)
                current_hv = best_hv
            else:
                # If no candidate improves, we might need to try more or stop
                # To avoid infinite loop, we can just break or try random again
                # Let's try a few more random points
                improved = False
                for _ in range(5):
                    if used >= budget: break
                    x_rand = np.random.uniform(lo, hi)
                    f_rand = problem(x_rand)
                    used += 1
                    dominated = False
                    for f in F_pop:
                        if np.all(f <= f_rand) and np.any(f < f_rand):
                            dominated = True
                            break
                    if not dominated:
                        new_F = F_pop + [f_rand]
                        new_hv = hypervolume_3d(new_F, ref)
                        if new_hv > current_hv:
                            X_pop.append(x_rand)
                            F_pop.append(f_rand)
                            current_hv = new_hv
                            improved = True
                            break
                if not improved:
                    # Stagnation
                    # Try a larger mutation
                    # Or just break
                    pass

        # Return non-dominated points
        nd_indices = []
        for i, f in enumerate(F_pop):
            dominated = False
            for j, g in enumerate(F_pop):
                if i == j: continue
                if np.all(g <= f) and np.any(g < f):
                    dominated = True
                    break
            if not dominated:
                nd_indices.append(i)
                
        X_final = np.array([X_pop[i] for i in nd_indices])
        return X_final
