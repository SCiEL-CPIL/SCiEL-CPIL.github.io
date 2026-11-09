"""Multi-objective optimizer on RE21 using HypE with optimized components."""
# EVOLVE-BLOCK-START
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
        B = int(self.budget)
        d = self.dim
        
        lo = np.array([b[0] for b in bounds], dtype=np.float64)
        hi = np.array([b[1] for b in bounds], dtype=np.float64)
        span = hi - lo
        
        ref = np.array([3000.0, 0.06])

        xs = []
        fs = []
        
        def is_dominated(f_new, f_list):
            for f in f_list:
                if np.all(f <= f_new) and np.any(f < f_new):
                    return True
            return False

        def update_archive(x_new, f_new):
            if not fs:
                xs.append(x_new)
                fs.append(f_new)
                return True
            
            if is_dominated(f_new, fs):
                return False
            
            new_xs = []
            new_fs = []
            for i in range(len(fs)):
                if not (np.all(f_new <= fs[i]) and np.any(f_new < fs[i])):
                    new_xs.append(xs[i])
                    new_fs.append(fs[i])
            
            xs.clear()
            fs.clear()
            for i in range(len(new_xs)):
                xs.append(new_xs[i])
                fs.append(new_fs[i])
            
            xs.append(x_new)
            fs.append(f_new)
            return True

        def calc_hv(points, ref_pt):
            if not points:
                return 0.0
            p = np.array(points)
            mask = (p[:, 0] < ref_pt[0]) & (p[:, 1] < ref_pt[1])
            p = p[mask]
            if len(p) == 0:
                return 0.0
            
            p = p[np.argsort(p[:, 0])]
            f1 = p[:, 0]
            f2 = p[:, 1]
            
            hv = 0.0
            for i in range(len(p)):
                width = f1[i+1] if i < len(p)-1 else ref_pt[0]
                width = width - f1[i]
                if width <= 0: continue
                
                height = ref_pt[1] - f2[i]
                if height <= 0: continue
                
                # Find the max height of subsequent points that are within this width
                # Actually, standard 2D HV algorithm:
                # Sort by x ascending.
                # For each point i, the height contributed is the difference between 
                # f2[i] and the min f2 of all points j > i.
                # Wait, the standard algorithm is:
                # Sort by f1 ascending.
                # hv = sum_{i} (f1_{i+1} - f1_i) * (f2_i - min_{j>i} f2_j)
                # But we can just iterate backwards to keep track of min f2.
            
            # Let's use the standard O(N) after sort method
            min_f2 = ref_pt[1]
            for i in range(len(p) - 1, -1, -1):
                f1_val = p[i, 0]
                f2_val = p[i, 1]
                
                # The right edge is the next point's f1 or ref
                right_edge = p[i+1, 0] if i < len(p) - 1 else ref_pt[0]
                
                # The height is determined by the lowest f2 among points i..end
                # Actually, for the strip [f1_i, f1_{i+1}], the height is 
                # min(f2_i, min(f2_{i+1}...)) - 0 ? No.
                # The volume is integral.
                # Let's stick to the simpler but correct O(N^2) or O(N log N) if needed.
                # Given small N, O(N^2) is fine and less error-prone to rewrite.
                pass
            
            # Re-implementing HV correctly and simply
            hv = 0.0
            # Sort by f1
            idx = np.argsort(p[:, 0])
            p = p[idx]
            
            for i in range(len(p)):
                # Width
                w = (p[i+1, 0] if i < len(p)-1 else ref_pt[0]) - p[i, 0]
                if w <= 0:
                    continue
                # Height
                # The height of the rectangle starting at x=p[i,0] is determined
                # by the minimum f2 value among points p[i], p[i+1], ...
                # But since p is sorted by f1, and these are non-dominated, 
                # f2 should be decreasing as f1 increases.
                # So the min f2 in the suffix p[i:] is p[-1, 1]? 
                # No, non-dominated means if f1 increases, f2 must decrease.
                # So p[i, 1] > p[i+1, 1].
                # The area for the slice [p[i,0], p[i+1,0]] is 
                # width * (p[i, 1] - 0)? No, bounded by ref.
                # Actually, the hypervolume is the union of rectangles.
                # For point i, the rectangle is [0, p[i,0]] x [0, p[i,1]]? No.
                # It's the area under the curve defined by the points.
                # Since they are non-dominated and sorted by f1 asc, f2 desc.
                # The "skyline" is defined by these points.
                # The hypervolume is sum_{i} (x_{i+1} - x_i) * y_i 
                # where x_{N} = ref_x, and y_i is the f2 value.
                # But we must ensure y_i is the "active" height.
                # Because f2 is decreasing, y_i is the height for the segment [x_i, x_{i+1}].
                h = p[i, 1]
                # Check if h is below ref
                if h > ref_pt[1]:
                    h = ref_pt[1]
                # Also need to check if previous points constrain it?
                # No, because f2 is decreasing, the current point has the highest f2 
                # in its segment? No, f2 is decreasing, so p[0] has highest f2, p[N] lowest.
                # The segment [p[i,0], p[i+1,0]] is bounded above by p[i,1].
                # Is it bounded by anything else? No, because p[i+1,1] < p[i,1].
                # So the height is p[i,1].
                
                hv += w * h
            
            return hv

        def marginal_hv(idx):
            if len(fs) == 0:
                return 0.0
            if len(fs) == 1:
                return calc_hv([fs[0]], ref)
            
            hv_with = calc_hv(fs, ref)
            fs_copy = fs[:idx] + fs[idx+1:]
            hv_without = calc_hv(fs_copy, ref)
            return max(0.0, hv_with - hv_without)

        def generate_quasi_random(n, dim):
            # Halton sequence
            res = np.zeros((n, dim))
            for i in range(n):
                for j in range(dim):
                    base = j + 2
                    f = 1.0
                    r = 0.0
                    val = i + 1
                    while val > 0:
                        f /= base
                        r += f * (val % base)
                        val //= base
                    res[i, j] = r
            return res

        # Initialization
        n_init = min(30, B)
        init_points = generate_quasi_random(n_init, d)
        for i in range(n_init):
            if B <= 0 or time.time() > deadline:
                break
            x = lo + span * init_points[i]
            f = problem(x)
            f = np.array(f, dtype=np.float64)
            if np.all(np.isfinite(f)):
                update_archive(x, f)
                B -= 1

        def select_hype():
            n = len(xs)
            if n == 0:
                return None
            if n == 1:
                return 0
            
            contribs = np.array([marginal_hv(i) for i in range(n)])
            contribs = contribs + 1e-10
            probs = contribs / contribs.sum()
            return np.random.choice(n, p=probs)

        pop_size = 8
        
        while B > 0 and time.time() < deadline:
            candidates = []
            
            if len(xs) > 2:
                for _ in range(pop_size):
                    if np.random.random() < 0.7:
                        i1 = select_hype()
                        i2 = select_hype()
                        i3 = select_hype()
                        attempts = 0
                        while len(set([i1, i2, i3])) < 3 and attempts < 10:
                            i1 = select_hype()
                            i2 = select_hype()
                            i3 = select_hype()
                            attempts += 1
                            
                        F = 0.5 + np.random.random() * 0.5
                        v = xs[i1] + F * (xs[i2] - xs[i3])
                        v = np.clip(v, lo, hi)
                        candidates.append(v)
                    else:
                        i = select_hype()
                        elapsed_frac = min(1.0, (time.time() - start_time) / (deadline - start_time))
                        base_sigma = 0.1 * span
                        sigma = base_sigma * (1.0 - 0.7 * elapsed_frac)
                        v = xs[i] + np.random.normal(0, 1, d) * sigma
                        v = np.clip(v, lo, hi)
                        candidates.append(v)
            
            elapsed_frac = min(1.0, (time.time() - start_time) / (deadline - start_time))
            n_random = max(1, int(3 * (1.0 - 0.5 * elapsed_frac)))
            for _ in range(n_random):
                if B <= 0:
                    break
                x = lo + span * np.random.random(d)
                candidates.append(x)
            
            for x in candidates:
                if B <= 0 or time.time() > deadline:
                    break
                f = problem(x)
                f = np.array(f, dtype=np.float64)
                if not np.all(np.isfinite(f)):
                    continue
                update_archive(x, f)
                B -= 1

        # Final polish
        if len(xs) > 0 and B > 0 and time.time() < deadline:
            n_polish = min(len(xs), 8)
            contribs = []
            for i in range(len(xs)):
                c = marginal_hv(i)
                contribs.append((c, i))
            contribs.sort(reverse=True)
            top_indices = [idx for _, idx in contribs[:n_polish]]
            
            for idx in top_indices:
                if B <= 0 or time.time() > deadline:
                    break
                for trial in range(4):
                    if B <= 0 or time.time() > deadline:
                        break
                    sigma = 0.015 * span
                    v = xs[idx] + np.random.normal(0, 1, d) * sigma
                    v = np.clip(v, lo, hi)
                    f = problem(v)
                    f = np.array(f, dtype=np.float64)
                    if np.all(np.isfinite(f)):
                        update_archive(v, f)
                        B -= 1

        if len(fs) > 0:
            return np.array(fs)
        else:
            return np.array([])

# EVOLVE-BLOCK-END
