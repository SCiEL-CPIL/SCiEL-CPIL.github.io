import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.calls = 0
        self.start_time = None

    def __call__(self, func):
        self.start_time = time.time()
        d = self.dim
        N = self.budget
        deadline = self.start_time + 290.0

        def time_check():
            return time.time() < deadline

        def safe_eval(x):
            if self.calls >= N or not time_check():
                return np.inf
            self.calls += 1
            try:
                val = float(func(x))
                return val
            except Exception:
                return np.inf

        # Track best globally
        best_x = np.zeros(d)
        best_f = np.inf

        def update_best(x, f):
            nonlocal best_x, best_f
            if f < best_f:
                best_f = f
                best_x = x.copy()

        # Initial evaluation at origin
        x0 = np.zeros(d)
        f0 = safe_eval(x0)
        update_best(x0, f0)

        # Phase 1: Coarse search with multiple Nelder-Mead runs
        # Start points: Origin and a few random points
        rng = np.random.default_rng(42)
        start_points = [x0.copy()]
        for _ in range(4):
            start_points.append(rng.uniform(-4, 4, d))

        # Nelder-Mead implementation
        def nelder_mead(x_start, f_start):
            nonlocal best_f, best_x
            if not time_check():
                return

            # Simplex initialization
            delta = 0.5
            simplex = [x_start.copy()]
            f_vals = [f_start]

            for i in range(d):
                x_i = x_start.copy()
                x_i[i] += delta
                f_i = safe_eval(x_i)
                update_best(x_i, f_i)
                simplex.append(x_i)
                f_vals.append(f_i)

            alpha = 1.0
            gamma = 2.0
            rho = 0.5
            sigma = 0.5
            tol = 1e-8

            # Run for a limited number of iterations or until convergence
            # We want to leave budget for Phase 2
            max_iters = 100
            for _ in range(max_iters):
                if not time_check() or self.calls >= N:
                    break

                # Sort simplex
                combined = list(zip(f_vals, simplex))
                combined.sort(key=lambda x: x[0])
                f_vals = [c[0] for c in combined]
                simplex = [c[1] for c in combined]

                x_worst = simplex[-1]
                f_worst = f_vals[-1]
                x_best = simplex[0]
                f_best = f_vals[0]

                if abs(f_worst - f_best) < tol:
                    break

                centroid = np.sum(simplex[:-1], axis=0) / d

                # Reflection
                x_r = centroid + alpha * (centroid - x_worst)
                f_r = safe_eval(x_r)
                update_best(x_r, f_r)

                if f_r < f_best:
                    # Expansion
                    x_e = centroid + gamma * (centroid - x_worst)
                    f_e = safe_eval(x_e)
                    update_best(x_e, f_e)
                    if f_e < f_r:
                        simplex[-1] = x_e
                        f_vals[-1] = f_e
                    else:
                        simplex[-1] = x_r
                        f_vals[-1] = f_r
                elif f_r < f_vals[-2]:
                    simplex[-1] = x_r
                    f_vals[-1] = f_r
                else:
                    # Contraction
                    x_c = centroid + rho * (x_worst - centroid)
                    f_c = safe_eval(x_c)
                    update_best(x_c, f_c)
                    if f_c < f_worst:
                        simplex[-1] = x_c
                        f_vals[-1] = f_c
                    else:
                        # Shrink
                        for i in range(1, d + 1):
                            simplex[i] = x_best + sigma * (simplex[i] - x_best)
                            f_vals[i] = safe_eval(simplex[i])
                            update_best(simplex[i], f_vals[i])

        for sp in start_points:
            if not time_check() or self.calls >= N:
                break
            f_sp = safe_eval(sp)
            update_best(sp, f_sp)
            nelder_mead(sp, f_sp)

        # Phase 2: Coordinate Descent with Golden Section Search
        # Bent Cigar has a narrow valley. Coordinate descent is effective if we can find the valley direction.
        # Since we have a good starting point from Phase 1, we can try to refine it.
        # We will perform a limited number of coordinate descent sweeps.
        
        # Check if we have budget left
        if self.calls < N - 10 and time_check():
            x_current = best_x.copy()
            f_current = best_f
            golden_ratio = (1 + np.sqrt(5)) / 2
            inv_phi = 1 / golden_ratio
            
            # We will try to optimize each coordinate.
            # To avoid getting stuck in local minima due to noise or bad estimates, 
            # we might need to be careful with the search range.
            # However, Bent Cigar is unimodal.
            
            max_cd_iters = 50
            for _ in range(max_cd_iters):
                if not time_check() or self.calls >= N:
                    break
                
                improved = False
                for i in range(d):
                    if not time_check() or self.calls >= N:
                        break
                    
                    # Determine search interval
                    # Start with a small interval around current best
                    # If we find improvement, expand? Or just use a fixed reasonable range?
                    # Given the valley, a small step might be needed.
                    # Let's use a dynamic range based on the previous step or a fixed small range.
                    # Since we don't have gradient info, we assume the minimum is close.
                    # Let's try a range of +/- 0.1 around the current value.
                    # If the function decreases, we can keep shrinking.
                    
                    x_lo = max(x_current[i] - 0.5, -5.0)
                    x_hi = min(x_current[i] + 0.5, 5.0)
                    
                    if x_lo == x_hi:
                        continue

                    # Golden section search
                    c = x_hi - inv_phi * (x_hi - x_lo)
                    dd = x_lo + inv_phi * (x_hi - x_lo)
                    
                    x_c = x_current.copy()
                    x_c[i] = c
                    f_c = safe_eval(x_c)
                    update_best(x_c, f_c)
                    
                    x_dd = x_current.copy()
                    x_dd[i] = dd
                    f_dd = safe_eval(x_dd)
                    update_best(x_dd, f_dd)
                    
                    # We need to compare f_c and f_dd against the current best in this coordinate
                    # But we also need to maintain the golden section interval properly.
                    # Standard golden section search assumes we are minimizing a unimodal function.
                    # We will update the interval [x_lo, x_hi] based on f_c and f_dd.
                    
                    if f_c < f_dd:
                        x_hi = dd
                        x_current[i] = c
                        f_current = f_c # This is not strictly the function value at x_current, but close enough for tracking
                        # Actually, x_current[i] is now c. We should update x_current
                        # But we need to be careful: x_current is the point we are optimizing.
                        # If f_c < f_dd, the minimum is in [x_lo, dd].
                        # We replace x_hi with dd.
                        # The new points to evaluate will be based on the new interval.
                        # We already have f_c. We need a new point.
                        # Let's re-evaluate the golden points for the new interval.
                        # This is inefficient. A better way is to keep track of the points and values.
                        
                        # Let's restart the golden section logic for this coordinate with the new interval.
                        # Or simply:
                        # If f_c < f_dd:
                        #   x_hi = dd
                        #   x_current[i] = c (tentatively)
                        #   # We have f_c. We need to evaluate a new point in [x_lo, dd].
                        #   # The point c is already evaluated.
                        #   # The new interval is [x_lo, dd].
                        #   # The golden points are c1 = x_hi - inv_phi*(x_hi-x_lo) and c2 = x_lo + inv_phi*(x_hi-x_lo).
                        #   # One of these is c, the other is new.
                        #   # Let's just do a simple loop.
                        
                    else:
                        x_lo = c
                        x_current[i] = dd
                        f_current = f_dd

                    # To simplify and ensure progress, let's do a few iterations of golden section properly.
                    # This block above was getting messy. Let's implement a clean helper for 1D search.

        # Let's re-implement the coordinate descent phase more cleanly.
        
        def golden_section_search(x_base, coord_idx, x_lo, x_hi, max_evals=10):
            nonlocal best_f, best_x
            if not time_check() or self.calls >= N - max_evals:
                return x_base[coord_idx], best_f
            
            inv_phi = (1 + np.sqrt(5)) / 2
            inv_phi = 1 / inv_phi
            
            c = x_hi - inv_phi * (x_hi - x_lo)
            dd = x_lo + inv_phi * (x_hi - x_lo)
            
            x_c = x_base.copy()
            x_c[coord_idx] = c
            f_c = safe_eval(x_c)
            update_best(x_c, f_c)
            
            x_dd = x_base.copy()
            x_dd[coord_idx] = dd
            f_dd = safe_eval(x_dd)
            update_best(x_dd, f_dd)
            
            for _ in range(max_evals - 2):
                if not time_check() or self.calls >= N:
                    break
                
                if f_c < f_dd:
                    x_hi = dd
                    dd = c
                    f_dd = f_c
                    c = x_hi - inv_phi * (x_hi - x_lo)
                    x_c = x_base.copy()
                    x_c[coord_idx] = c
                    f_c = safe_eval(x_c)
                    update_best(x_c, f_c)
                else:
                    x_lo = c
                    c = dd
                    f_c = f_dd
                    dd = x_lo + inv_phi * (x_hi - x_lo)
                    x_dd = x_base.copy()
                    x_dd[coord_idx] = dd
                    f_dd = safe_eval(x_dd)
                    update_best(x_dd, f_dd)
            
            # Return the best point found in this search
            # We have f_c and f_dd. The best is min(f_c, f_dd).
            # But we also updated global best.
            # We want to return the x value for this coordinate that gave the best f.
            if f_c < f_dd:
                return c, f_c
            else:
                return dd, f_dd

        # Now, use this helper in the main loop
        if self.calls < N - 10 and time_check():
            x_current = best_x.copy()
            max_cd_sweeps = 20
            for _ in range(max_cd_sweeps):
                if not time_check() or self.calls >= N:
                    break
                
                improved = False
                for i in range(d):
                    if not time_check() or self.calls >= N:
                        break
                    
                    # Define search range
                    # Use a small range around the current best
                    # If the previous step was large, maybe use a larger range?
                    # Let's stick to a fixed small range for now, e.g., +/- 0.1
                    # But if the minimum is far, this might be slow.
                    # However, Phase 1 should have brought us close.
                    x_lo = max(x_current[i] - 0.2, -5.0)
                    x_hi = min(x_current[i] + 0.2, 5.0)
                    
                    if x_hi - x_lo < 1e-6:
                        continue

                    new_val, new_f = golden_section_search(x_current, i, x_lo, x_hi, max_evals=8)
                    
                    if new_f < best_f:
                        x_current[i] = new_val
                        improved = True
                    else:
                        # If no improvement, maybe the range was too small or we are at a local min?
                        # For Bent Cigar, it's unimodal, so if we are in the valley, we should improve.
                        # If not, we might be off the valley.
                        # Let's try a slightly larger range if no improvement?
                        # Or just move to the next coordinate.
                        pass
                
                if not improved:
                    break

        return best_x, best_f
