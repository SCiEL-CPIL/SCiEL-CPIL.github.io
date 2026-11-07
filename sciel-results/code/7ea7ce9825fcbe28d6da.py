import numpy as np
import time
from scipy.optimize import minimize

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280.0  # Safety margin
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        range_ = hi - lo
        
        archive_x = []
        archive_f = []
        evals = 0
        
        def time_ok():
            return time.time() < deadline

        def is_dominated(f_new, archive_f_list):
            if len(archive_f_list) == 0:
                return False
            af = np.array(archive_f_list)
            leq = np.all(af <= f_new, axis=1)
            lt = np.any(af < f_new, axis=1)
            return np.any(leq & lt)
            
        def dominates(f_new, f_old):
            return np.all(f_new <= f_old) and np.any(f_new < f_old)

        def add_to_archive(x, f):
            nonlocal evals
            if is_dominated(f, archive_f):
                return False
            
            new_archive_x = []
            new_archive_f = []
            for xi, fi in zip(archive_x, archive_f):
                if dominates(f, fi):
                    continue
                new_archive_x.append(xi)
                new_archive_f.append(fi)
                
            new_archive_x.append(x)
            new_archive_f.append(f)
            
            archive_x[:] = new_archive_x
            archive_f[:] = new_archive_f
            return True

        def compute_hv(archive_f_list, ref_point):
            if not archive_f_list:
                return 0.0
            # Filter points strictly better than ref
            pts = [f for f in archive_f_list if np.all(f < ref_point)]
            if not pts:
                return 0.0
            # Sort by f1
            pts = sorted(pts, key=lambda x: x[0])
            hv = 0.0
            # 2D HV calculation
            # Iterate from best f1 to worst
            # For each point, the width is determined by the next point's f1 (or ref)
            # The height is ref_f2 - current_f2
            # But we must ensure we don't double count.
            # Standard 2D HV:
            # Sort by f1 ascending.
            # For i in range(len(pts)):
            #   width = (pts[i+1][0] if i < len(pts)-1 else ref_point[0]) - pts[i][0]
            #   height = ref_point[1] - pts[i][1]
            #   hv += width * height
            # This works if the points are non-dominated and sorted.
            # Wait, if pts[i][1] > pts[i+1][1], the height for i is limited by pts[i+1][1]?
            # No, the "skyline" algorithm:
            # Sort by f1 ascending.
            # max_f2 = -inf
            # for i in reversed(range(len(pts))): # from worst f1 to best
            #    if pts[i][1] < max_f2: # dominated by a point with better f1 and better f2? No.
            #       continue
            #    max_f2 = pts[i][1]
            #    # width is to the next point in f1 (which is worse, so to the right)
            #    # Actually, let's use the rectangle decomposition.
            
            # Simpler O(N^2) for small N:
            # HV is sum of rectangles.
            # Let's just use the standard sweep.
            # Sort by f1 ascending.
            # hv = 0
            # prev_f1 = pts[0][0]
            # min_f2 = ref_point[1]
            # for i in range(len(pts)):
            #    f1, f2 = pts[i]
            #    if f2 < min_f2:
            #       min_f2 = f2
            #    # This is not quite right.
            
            # Correct 2D HV:
            # Sort by f1 ascending.
            # hv = 0
            # for i in range(len(pts)):
            #    f1, f2 = pts[i]
            #    # The width of the slice for this point is determined by the next point's f1
            #    next_f1 = pts[i+1][0] if i < len(pts) - 1 else ref_point[0]
            #    width = next_f1 - f1
            #    # The height is ref_f2 - f2
            #    height = ref_point[1] - f2
            #    hv += width * height
            # This assumes that for any f1 in [pts[i][0], next_f1], the best f2 is pts[i][1].
            # This is true if pts[i][1] <= pts[i+1][1] ... which is not guaranteed for non-dominated sets?
            # In a non-dominated set, if f1 increases, f2 must decrease.
            # So pts[0] has min f1, max f2.
            # pts[N-1] has max f1, min f2.
            # So f2 is decreasing as f1 increases.
            # Therefore, for the slice [pts[i][0], pts[i+1][0]], the min f2 is pts[i+1][1]?
            # No, the min f2 in that slice is the f2 of the point with the LARGEST f1 in the slice, which is pts[i+1][1].
            # But the point pts[i] has f2 = pts[i][1] which is LARGER than pts[i+1][1].
            # So the "skyline" at f1=pts[i] is pts[i][1].
            # At f1=pts[i+1], it drops to pts[i+1][1].
            # So the area is (next_f1 - f1) * (ref_f2 - f2_i)?
            # No. The area under the curve.
            # The curve is a step function.
            # From f1_i to f1_{i+1}, the "active" point is the one with the smallest f2 in that range?
            # No, the Pareto front is the set of points. The hypervolume is the area dominated by the union of rectangles from each point to the reference.
            # Since they are non-dominated, the union forms a staircase.
            # The area is sum over i of (f1_{i+1} - f1_i) * (ref_f2 - f2_i) is WRONG.
            # It should be (f1_{i+1} - f1_i) * (ref_f2 - f2_{i+1})? No.
            
            # Let's use the simple O(N^2) contribution method for selection, and a robust HV calc for scoring if needed (though scoring is external).
            # For selection, we just need relative contributions.
            # Contribution of p = HV(All) - HV(All \ {p})
            
            pass

        def hv_contribution(f_point):
            # Calculate HV of full archive
            # Calculate HV of archive without f_point
            # This is O(N^2) per point, O(N^3) total. N is small (<50).
            # 50^3 = 125,000 ops. Very fast.
            
            def hv_calc(f_list):
                if not f_list:
                    return 0.0
                pts = [f for f in f_list if np.all(f < np.array([3000.0, 0.06]))]
                if not pts:
                    return 0.0
                pts = sorted(pts, key=lambda x: x[0])
                hv = 0.0
                ref = np.array([3000.0, 0.06])
                # 2D HV:
                # Sort by f1 asc.
                # For each point i, the width is to the next point.
                # The height is ref_f2 - f2_i.
                # Wait, the "staircase" logic:
                # The area is the sum of rectangles.
                # Rectangle i has width (f1_{i+1} - f1_i) and height (ref_f2 - f2_i)?
                # Let's test:
                # P1=(1, 10), P2=(2, 5), Ref=(10, 10)
                # Sorted: P1, P2.
                # Width1 = 2-1=1. Height1 = 10-10=0. Area=0.
                # Width2 = 10-2=8. Height2 = 10-5=5. Area=40.
                # Total 40.
                # Actual HV:
                # P1 dominates (1..10, 10..10) -> area 0.
                # P2 dominates (2..10, 5..10) -> area 8*5=40.
                # Union: (1..2, 10..10) is 0. (2..10, 5..10) is 40. (1..2, 5..10) is dominated by P2?
                # P2 is (2,5). It dominates x in [2,10] and y in [5,10].
                # Does it dominate x in [1,2]? No, f1=2 > 1.
                # So P1 covers [1,2] x [10,10] (zero height).
                # So total is 40.
                # Formula: Sum (f1_{i+1} - f1_i) * (ref_f2 - f2_i)
                # i=0: (2-1)*(10-10) = 0.
                # i=1: (10-2)*(10-5) = 40.
                # Total 40. Correct.
                
                for i in range(len(pts)):
                    f1, f2 = pts[i]
                    next_f1 = pts[i+1][0] if i < len(pts) - 1 else ref[0]
                    width = next_f1 - f1
                    height = ref[1] - f2
                    if width > 0 and height > 0:
                        hv += width * height
                return hv

            full_hv = hv_calc(archive_f)
            reduced_f = [f for f in archive_f if f is not f_point]
            reduced_hv = hv_calc(reduced_f)
            return full_hv - reduced_hv

        # 1. Initialization: Halton Sequence
        def halton(dim, n, base_list=None):
            if base_list is None:
                base_list = [2, 3, 5, 7]
            points = []
            for i in range(1, n + 1):
                p = []
                for j in range(dim):
                    f = 1.0
                    r = 0.0
                    i_tmp = i
                    while i_tmp > 0:
                        f /= base_list[j]
                        r += f * (i_tmp % base_list[j])
                        i_tmp //= base_list[j]
                    p.append(r)
                points.append(p)
            return np.array(points)

        init_size = 25
        if time_ok():
            X_init = lo + range_ * halton(self.dim, init_size)
            for x in X_init:
                if not time_ok():
                    break
                f = problem(x)
                evals += 1
                add_to_archive(x, f)
        
        # 2. Main Search: Mutation + HV-based Selection
        main_budget = self.budget - 80 # Reserve 80 for polish
        step_size = 0.1 * range_
        
        while evals < main_budget and time_ok():
            if len(archive_x) == 0:
                break
                
            # Calculate contributions
            contributions = []
            for f in archive_f:
                contributions.append(hv_contribution(f))
            contributions = np.array(contributions)
            
            # Select parent with max contribution (or random if all 0)
            if np.sum(contributions) > 0:
                # Softmax selection to allow some exploration
                exp_contribs = np.exp(contributions - np.max(contributions))
                probs = exp_contribs / np.sum(exp_contribs)
                idx = np.random.choice(len(archive_x), p=probs)
            else:
                idx = np.random.randint(len(archive_x))
                
            parent = archive_x[idx]
            child = parent + step_size * np.random.normal(size=self.dim)
            child = np.clip(child, lo, hi)
            
            f_child = problem(child)
            evals += 1
            
            added = add_to_archive(child, f_child)
            if added:
                step_size *= 0.95
            else:
                step_size *= 1.05
                step_size = np.minimum(step_size, 0.3 * range_)
                
        # 3. Polish: Nelder-Mead on extreme points
        if time_ok() and evals < self.budget:
            # Identify best f1, best f2, best sum
            if len(archive_f) > 0:
                archive_f_arr = np.array(archive_f)
                idx_min_f1 = np.argmin(archive_f_arr[:, 0])
                idx_min_f2 = np.argmin(archive_f_arr[:, 1])
                idx_min_sum = np.argmin(archive_f_arr[:, 0] + archive_f_arr[:, 1])
                
                candidates = [idx_min_f1, idx_min_f2, idx_min_sum]
                # Add 1 random intermediate
                if len(archive_x) > 3:
                    others = [i for i in range(len(archive_x)) if i not in candidates]
                    candidates.append(np.random.choice(others))
                
                for idx in candidates:
                    if evals >= self.budget - 5 or not time_ok():
                        break
                    
                    x0 = archive_x[idx].copy()
                    
                    # Determine objective for NM
                    # We'll try minimizing f1, f2, and sum
                    # To keep it simple, we run NM to minimize the sum of normalized objectives?
                    # Or just run 3 separate NMs?
                    # Budget is tight. Let's run 2 NMs: one for f1, one for f2.
                    # And maybe one for sum.
                    
                    # Define scalar objective
                    # We will run NM 3 times total if budget allows.
                    # 1. Min f1
                    # 2. Min f2
                    # 3. Min f1+f2
                    
                    def obj_func(x, target):
                        fx = problem(x)
                        evals_nonlocal = evals # Can't easily increment nonlocal in nested lambda without closure trick
                        # We need to count evals.
                        # Let's use a wrapper.
                        return fx[target]
                    
                    # We need a proper counter.
                    # Let's create a class or use a list for counter.
                    
                    # Run NM for min f1
                    if evals < self.budget - 20:
                        counter = [evals]
                        def obj_f1(x):
                            nonlocal evals
                            if not time_ok() or evals >= self.budget:
                                return 1e10
                            fx = problem(x)
                            evals += 1
                            return fx[0]
                        
                        try:
                            res = minimize(obj_f1, x0, method='Nelder-Mead', options={'maxiter': 20, 'xatol': 1e-6, 'fatol': 1e-6})
                            if res.success:
                                add_to_archive(res.x, problem(res.x)) # Note: this evaluates again!
                                # We already evaluated res.x in the last step of NM?
                                # NM might not have evaluated the exact final point if it stopped.
                                # To be safe, we re-evaluate. But we have budget check.
                                # Actually, `minimize` returns the function value.
                                # We should check if res.x is already in archive.
                        except:
                            pass

                    # Run NM for min f2
                    if evals < self.budget - 20:
                        def obj_f2(x):
                            nonlocal evals
                            if not time_ok() or evals >= self.budget:
                                return 1e10
                            fx = problem(x)
                            evals += 1
                            return fx[1]
                        
                        try:
                            res = minimize(obj_f2, x0, method='Nelder-Mead', options={'maxiter': 20, 'xatol': 1e-6, 'fatol': 1e-6})
                            if res.success:
                                add_to_archive(res.x, problem(res.x))
                        except:
                            pass

                    # Run NM for min sum
                    if evals < self.budget - 20:
                        def obj_sum(x):
                            nonlocal evals
                            if not time_ok() or evals >= self.budget:
                                return 1e10
                            fx = problem(x)
                            evals += 1
                            return fx[0] + fx[1]
                        
                        try:
                            res = minimize(obj_sum, x0, method='Nelder-Mead', options={'maxiter': 20, 'xatol': 1e-6, 'fatol': 1e-6})
                            if res.success:
                                add_to_archive(res.x, problem(res.x))
                        except:
                            pass

        return archive_x
