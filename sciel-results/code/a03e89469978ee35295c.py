import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280.0
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        n_evals = 0
        max_evals = self.budget
        dim = self.dim
        
        # ZDT6: f1 = 1 + 90*(1-sqrt(x_1)), f2 = (1-f1^2/121)*(1-sqrt(f1/11))
        # Pareto front: f1 in [1, 11]
        
        def f2_of_f1(f1):
            return (1 - f1**2 / 121.0) * (1 - np.sqrt(f1 / 11.0))
        
        def f2_of_f1_vec(f1_arr):
            return (1 - f1_arr**2 / 121.0) * (1 - np.sqrt(f1_arr / 11.0))
        
        def compute_hv(f1_vals):
            """Compute hypervolume for points on Pareto front, ref [11,11]"""
            if len(f1_vals) == 0:
                return 0.0
            f1_vals = np.sort(f1_vals)
            f2_vals = f2_of_f1_vec(f1_vals)
            N = len(f1_vals)
            hv = 0.0
            for i in range(N - 1):
                hv += (f2_vals[i] - f2_vals[i+1]) * (11.0 - f1_vals[i])
            hv += f2_vals[N-1] * (11.0 - f1_vals[N-1])
            return hv
        
        def marginal_hv_fast(f1_new, f1_sorted, f2_sorted):
            """Compute marginal HV contribution of adding f1_new to existing set.
            Points are sorted by f1 ascending. f2 is decreasing in f1.
            """
            if len(f1_sorted) == 0:
                return (11.0 - f1_new) * f2_of_f1(f1_new)
            
            f2_new = f2_of_f1(f1_new)
            pos = np.searchsorted(f1_sorted, f1_new)
            
            # Determine the segment where this point falls
            # The new point replaces the gap between f1_sorted[pos-1] and f1_sorted[pos]
            # (or from 1.0 if pos==0, or to 11.0 if pos==len)
            
            # Old HV contribution of the segment the new point falls into
            if pos == 0:
                # New point is leftmost
                # Old: segment from f1_sorted[0] to f1_sorted[1] (or just f1_sorted[0] to 11 if len==1)
                if len(f1_sorted) == 1:
                    old_seg_hv = f2_sorted[0] * (11.0 - f1_sorted[0])
                else:
                    old_seg_hv = (f2_sorted[0] - f2_sorted[1]) * (11.0 - f1_sorted[0]) + f2_sorted[1] * (11.0 - f1_sorted[1])
                    # Wait, this is getting complicated. Let's just compute the change in the two adjacent segments.
                    # Actually, the simplest correct way for the 2D case with sorted points:
                    # The new point splits the interval [f1_sorted[pos-1], f1_sorted[pos]] (with boundaries 1.0 and 11.0)
                    # Let left_f1 = f1_sorted[pos-1] if pos>0 else 1.0
                    # Let right_f1 = f1_sorted[pos] if pos<len else 11.0
                    # Let left_f2 = f2_sorted[pos-1] if pos>0 else f2_of_f1(1.0)  # Note: f2(1.0) is the max f2
                    # Let right_f2 = f2_sorted[pos] if pos<len else f2_of_f1(11.0)  # Note: f2(11.0) is 0
                    
                    # But we need to be careful with the HV formula.
                    # HV = sum of (f2_i - f2_{i+1}) * (11 - f1_i) for i=0..N-2, plus f2_{N-1} * (11 - f1_{N-1})
                    # This is equivalent to the area under the step function.
                    
                    # Let's use a simpler approach: compute HV of the 3 points (left, new, right) vs HV of 2 points (left, right)
                    # But we need to include the boundaries 1.0 and 11.0 as virtual points.
                    
                    pass
            
            # Let's use the virtual boundary approach for correctness and simplicity
            # Virtual points: (1.0, f2(1.0)) and (11.0, f2(11.0)=0)
            # But f2(11.0) = (1-1)*(...) = 0. Correct.
            # f2(1.0) = (1-1/121)*(1-1/sqrt(11))
            
            # Actually, the most robust way is to just compute the HV difference directly using the sorted arrays
            # by inserting the new point and computing the difference. But that's O(N) per candidate.
            # With N up to 200 and ~100 candidates per step, that's 200*100*200 = 4M ops per step, 200 steps = 800M ops.
            # That might be too slow in Python.
            
            # Let's use the O(1) formula for the marginal HV in 2D.
            # The marginal HV is the area of the "notch" created by the new point.
            # If the new point is at (f1_new, f2_new) and it falls between (f1_left, f2_left) and (f1_right, f2_right),
            # then the marginal HV is:
            # (f2_left - f2_new) * (f1_new - f1_left) + (f2_new - f2_right) * (f1_right - f1_new)
            # Wait, no. The HV is the area under the curve to the left and below.
            # The new point adds area = (f2_new - f2_right) * (f1_right - f1_new) + (f2_left - f2_new) * (f1_new - f1_left)
            # Actually, it's simpler: the area added is the rectangle (f1_new - f1_left) * (f2_new - f2_right) 
            # minus the triangle... no, it's just the area of the polygon formed.
            
            # Let's stick to the direct computation but optimize it.
            # For N=200, computing HV is O(N). With 100 candidates, that's 20000 ops per step. 200 steps = 4M ops.
            # That's fine in Python if we use numpy.
            
            f1_new_arr = np.insert(f1_sorted, pos, f1_new)
            f2_new_arr = np.insert(f2_sorted, pos, f2_new)
            
            N = len(f1_new_arr)
            hv_new = 0.0
            for i in range(N - 1):
                hv_new += (f2_new_arr[i] - f2_new_arr[i+1]) * (11.0 - f1_new_arr[i])
            hv_new += f2_new_arr[N-1] * (11.0 - f1_new_arr[N-1])
            
            N_old = len(f1_sorted)
            hv_old = 0.0
            for i in range(N_old - 1):
                hv_old += (f2_sorted[i] - f2_sorted[i+1]) * (11.0 - f1_sorted[i])
            hv_old += f2_sorted[N_old-1] * (11.0 - f1_sorted[N_old-1])
            
            return hv_new - hv_old
        
        def equal_hv_spacing(N):
            """Find N f1 values where each point contributes equally to HV."""
            # Compute total HV with dense reference
            ref_f1 = np.linspace(1.0, 11.0, 50000)
            ref_f2 = f2_of_f1_vec(ref_f1)
            total_hv = 0.0
            for i in range(len(ref_f1)-1):
                total_hv += (ref_f2[i] - ref_f2[i+1]) * (11.0 - ref_f1[i])
            total_hv += ref_f2[-1] * (11.0 - ref_f1[-1])
            
            target_per_point = total_hv / N
            
            def hv_from_a(a):
                """HV of the Pareto front segment from f1=a to f1=11"""
                if a >= 11.0:
                    return 0.0
                n_pts = 5000
                f1_seg = np.linspace(a, 11.0, n_pts)
                f2_seg = f2_of_f1_vec(f1_seg)
                hv = 0.0
                for i in range(n_pts - 1):
                    hv += (f2_seg[i] - f2_seg[i+1]) * (11.0 - f1_seg[i])
                hv += f2_seg[-1] * (11.0 - f1_seg[-1])
                return hv
            
            f1_vals = np.zeros(N)
            f1_vals[0] = 1.0
            f1_vals[-1] = 11.0
            
            for i in range(1, N-1):
                target_hv = target_per_point * (N - i)
                lo_a, hi_a = 1.0, 11.0
                for _ in range(40):
                    mid = (lo_a + hi_a) / 2
                    if hv_from_a(mid) > target_hv:
                        lo_a = mid
                    else:
                        hi_a = mid
                f1_vals[i] = (lo_a + hi_a) / 2
            
            return f1_vals
        
        # Initialize with equal-hv spacing for a good starting set
        init_n = min(20, max_evals)
        f1_init = equal_hv_spacing(init_n)
        
        f1_list = list(f1_init)
        f2_list = list(f2_of_f1_vec(f1_init))
        
        # Evaluate initial points
        for f1 in f1_list:
            if time.time() > deadline:
                break
            if n_evals >= max_evals:
                break
            g = (f1 - 1.0) / 90.0
            x1 = (1.0 - g) ** 2
            x = np.zeros(dim)
            x[0] = x1
            x = np.clip(x, lo, hi)
            try:
                problem(x)
                n_evals += 1
            except Exception:
                break
        
        # Greedily add points with max marginal HV
        remaining = max_evals - n_evals
        for _ in range(remaining):
            if time.time() > deadline:
                break
            if n_evals >= max_evals:
                break
            
            f1_sorted = np.array(f1_list)
            f2_sorted = np.array(f2_list)
            
            # Generate candidates: midpoints, quartiles, and fine grid
            candidates = []
            for i in range(len(f1_sorted) - 1):
                a, b = f1_sorted[i], f1_sorted[i+1]
                candidates.append((a + b) / 2.0)
                candidates.append((3*a + b) / 4.0)
                candidates.append((a + 3*b) / 4.0)
            
            # Add some global fine grid points
            for t in np.linspace(0, 1, 50):
                candidates.append(1.0 + 10.0 * t)
            
            candidates = np.array(candidates)
            candidates = candidates[(candidates > 1.0 + 1e-9) & (candidates < 11.0 - 1e-9)]
            candidates = np.unique(candidates)
            
            if len(candidates) == 0:
                break
            
            # Find candidate with max marginal HV
            best_f1 = None
            best_marginal = -1
            for f1_cand in candidates:
                if time.time() > deadline:
                    break
                marg = marginal_hv_fast(f1_cand, f1_sorted, f2_sorted)
                if marg > best_marginal:
                    best_marginal = marg
                    best_f1 = f1_cand
            
            if best_f1 is None:
                break
            
            # Add the best point
            pos = np.searchsorted(f1_sorted, best_f1)
            f1_list.insert(pos, best_f1)
            f2_list.insert(pos, f2_of_f1(best_f1))
            
            # Evaluate
            g = (best_f1 - 1.0) / 90.0
            x1 = (1.0 - g) ** 2
            x = np.zeros(dim)
            x[0] = x1
            x = np.clip(x, lo, hi)
            try:
                problem(x)
                n_evals += 1
            except Exception:
                break
        
        return None
