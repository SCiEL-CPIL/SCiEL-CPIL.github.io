import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0  # Safety margin for 300s limit
        d = self.dim
        N = self.budget
        calls_used = 0
        best_x = np.zeros(d)
        best_f = np.inf

        def safe_eval(x):
            nonlocal calls_used, best_f, best_x
            if calls_used >= N or time.time() > deadline:
                return np.inf
            calls_used += 1
            try:
                val = float(func(x))
            except Exception:
                return np.inf
            if val < best_f:
                best_f = val
                best_x = x.copy()
            return val

        # 1. Initial coarse search to find a good basin
        # Start at origin, then a few random points
        x0 = np.zeros(d)
        f0 = safe_eval(x0)
        
        # Try a few random starts if budget allows
        rng = np.random.default_rng(42)
        for _ in range(5):
            if calls_used >= N or time.time() > deadline:
                break
            xr = rng.uniform(-4, 4, d)
            fr = safe_eval(xr)
            if fr < best_f:
                best_x = xr.copy()

        # 2. Coordinate Descent with Golden Section Search
        # This exploits the anisotropy of Bent Cigar (narrow valley)
        # by decoupling dimensions.
        if calls_used < N and time.time() < deadline:
            # Determine initial step size based on current best
            # If we are far from optimum, step should be larger.
            # Bent cigar has a narrow valley, so we need to be careful.
            # Let's use a fixed number of coordinate descent cycles.
            
            # Estimate scale: if best_f is large, we might be far away.
            # But we don't know f_opt. 
            # Let's just use a standard step size and adapt if stuck.
            
            # Golden section search parameters
            phi = (1 + np.sqrt(5)) / 2
            inv_phi = 1 / phi
            inv_phi2 = inv_phi * inv_phi
            
            # We will iterate over dimensions.
            # To save calls, we only do full 1D search if the function value
            # is not already extremely small or if we haven't converged.
            
            # Let's do a few passes of coordinate descent.
            # Each pass iterates through all dimensions.
            # Inside each dimension, we do a 1D search.
            
            # Initial step size for 1D search
            # If we are at the optimum of a dimension, the function is flat or convex.
            # Bent cigar: sum (x_i)^2 + 100 * (x_i - x_{i+1})^2 (simplified)
            # Actually f12 is sum (x_i)^2 + 100 * (x_i - x_{i+1})^2 ? No, bent cigar is:
            # f(x) = sum (x_i)^2 + 100 * (x_i - x_{i+1})^2 ? 
            # Standard Bent Cigar: f(x) = sum (x_i)^2 + 100 * (x_i - x_{i+1})^2 is not quite right.
            # BBOB f12: f(x) = sum (x_i)^2 + 100 * (x_i - x_{i+1})^2 is often cited as a variant.
            # The real Bent Cigar is: f(x) = sum (x_i)^2 + 100 * (x_i - x_{i+1})^2 ?
            # Actually, the classic Bent Cigar is: f(x) = sum (x_i)^2 + 100 * (x_i - x_{i+1})^2 ?
            # Wait, the standard Bent Cigar function is:
            # f(x) = sum (x_i)^2 + 100 * (x_i - x_{i+1})^2 ?
            # Let's check the definition. 
            # BBOB f12: Bent Cigar.
            # f(x) = sum (x_i)^2 + 100 * (x_i - x_{i+1})^2 ?
            # No, the standard definition is:
            # f(x) = sum (x_i)^2 + 100 * (x_i - x_{i+1})^2 ?
            # Actually, it is often defined as:
            # f(x) = sum (x_i)^2 + 100 * (x_i - x_{i+1})^2 ?
            # Let's assume the function is smooth and convex (or close to it).
            
            # We will perform coordinate descent.
            # For each dimension i, we fix other coordinates and minimize f(x) along x_i.
            # We use golden section search.
            
            # How many iterations of coordinate descent?
            # We have 500 calls.
            # Initial: 6 calls.
            # Remaining: ~494.
            # Each dimension search takes ~10-20 calls.
            # 5 dimensions * 20 calls = 100 calls per pass.
            # So we can do ~4 passes.
            
            # Let's implement a single pass first and see how many calls it uses.
            # Then loop until budget is exhausted or no improvement.
            
            prev_best_f = best_f
            pass_count = 0
            max_passes = 10
            
            while calls_used < N and time.time() < deadline and pass_count < max_passes:
                pass_count += 1
                improved = False
                for i in range(d):
                    if calls_used >= N or time.time() > deadline:
                        break
                    
                    # 1D Golden Section Search along dimension i
                    # Current value: best_x[i]
                    # We need to determine a search interval [a, b].
                    # Let's start with a small interval around the current best.
                    # If the function is very flat, we might need to expand.
                    # But Bent Cigar has a narrow valley, so the optimum is sharp in the "long" direction?
                    # No, Bent Cigar has a narrow valley in the direction perpendicular to the "cigar".
                    # The function is flat along the cigar and steep across it.
                    # Coordinate descent works well if the axes align with the principal components.
                    # But here the axes are rotated (hidden rotation).
                    # So coordinate descent might be slow if the rotation is bad.
                    
                    # However, the prompt says "hidden seeded shift, rotation and offset".
                    # If the rotation is arbitrary, coordinate descent might be very slow.
                    # But the inspiration says it works.
                    # Let's try it.
                    
                    # Determine step size
                    # If we are far from the minimum, we need larger steps.
                    # Let's use a step size that is a fraction of the box size, but shrink if we are close.
                    # Or just use a fixed step size and expand if we hit the boundary of the search interval.
                    
                    # Let's use a simple line search:
                    # Try x[i] +/- delta.
                    # If both are worse, try smaller delta.
                    # If one is better, search in that direction.
                    
                    # Simpler: Just do a few evaluations to bracket the minimum.
                    # x_center = best_x[i]
                    # delta = 1.0
                    # Check x_center - delta, x_center + delta.
                    # If both are worse, we are likely at the minimum (or very close).
                    # If one is better, expand or search.
                    
                    # Let's use a robust 1D search.
                    x_center = best_x.copy()
                    val_center = best_f
                    
                    # Initial step
                    step = 0.5
                    # Try to find a better point in the neighborhood
                    # We will do a small number of evaluations to find a descent direction.
                    
                    # Check left and right
                    x_left = x_center.copy()
                    x_left[i] -= step
                    f_left = safe_eval(x_left)
                    
                    x_right = x_center.copy()
                    x_right[i] += step
                    f_right = safe_eval(x_right)
                    
                    if f_left >= val_center and f_right >= val_center:
                        # We are at a local minimum (or very close)
                        # Try smaller step to be sure
                        step *= 0.5
                        x_left2 = x_center.copy()
                        x_left2[i] -= step
                        f_left2 = safe_eval(x_left2)
                        
                        x_right2 = x_center.copy()
                        x_right2[i] += step
                        f_right2 = safe_eval(x_right2)
                        
                        if f_left2 >= val_center and f_right2 >= val_center:
                            # Really at minimum
                            continue
                        elif f_left2 < val_center:
                            # Search left
                            a = x_left2[i]
                            b = x_center[i]
                            # Golden section in [a, b]
                            # But we need to evaluate f at a and b? We have f_left2 and val_center.
                            # Let's just do a few golden section steps.
                            # Actually, if f_left2 < val_center, the minimum is in [a, b] or further left.
                            # Let's assume it's in [a, b] for now.
                            # We need to define the function to evaluate.
                            def f_1d(val):
                                x_tmp = x_center.copy()
                                x_tmp[i] = val
                                return safe_eval(x_tmp)
                            
                            # Golden section search in [a, b]
                            # a < b
                            # c = b - inv_phi * (b - a)
                            # d = a + inv_phi * (b - a)
                            # Evaluate f(c), f(d)
                            # Keep the better half
                            # Do this for ~5-10 iterations
                            
                            # But we already have f(a) = f_left2 and f(b) = val_center.
                            # We need f(c) and f(d).
                            c = b - inv_phi * (b - a)
                            d = a + inv_phi * (b - a)
                            fc = f_1d(c)
                            fd = f_1d(d)
                            
                            for _ in range(5):
                                if fc < fd:
                                    b = d
                                    d = c
                                    fd = fc
                                    c = b - inv_phi * (b - a)
                                    fc = f_1d(c)
                                else:
                                    a = c
                                    c = d
                                    fc = fd
                                    d = a + inv_phi * (b - a)
                                    fd = f_1d(d)
                            
                            # Best point is min(c, d)
                            if fc < fd:
                                best_val_1d = c
                            else:
                                best_val_1d = d
                            
                            # Update best_x
                            new_x = x_center.copy()
                            new_x[i] = best_val_1d
                            new_f = f_1d(best_val_1d)
                            if new_f < best_f:
                                best_x = new_x
                                best_f = new_f
                                improved = True
                        elif f_right2 < val_center:
                            # Search right
                            a = x_center[i]
                            b = x_right2[i]
                            def f_1d(val):
                                x_tmp = x_center.copy()
                                x_tmp[i] = val
                                return safe_eval(x_tmp)
                            
                            c = b - inv_phi * (b - a)
                            d = a + inv_phi * (b - a)
                            fc = f_1d(c)
                            fd = f_1d(d)
                            
                            for _ in range(5):
                                if fc < fd:
                                    b = d
                                    d = c
                                    fd = fc
                                    c = b - inv_phi * (b - a)
                                    fc = f_1d(c)
                                else:
                                    a = c
                                    c = d
                                    fc = fd
                                    d = a + inv_phi * (b - a)
                                    fd = f_1d(d)
                            
                            if fc < fd:
                                best_val_1d = c
                            else:
                                best_val_1d = d
                            
                            new_x = x_center.copy()
                            new_x[i] = best_val_1d
                            new_f = f_1d(best_val_1d)
                            if new_f < best_f:
                                best_x = new_x
                                best_f = new_f
                                improved = True
                        else:
                            # Still at minimum
                            continue
                    else:
                        # One side is better
                        # We can do a golden section search in the interval [min(left, right), center] or [center, max(left, right)]
                        # But we might have overshot.
                        # Let's just use the better side to bracket.
                        # If f_left < val_center, search in [x_left[i], x_center[i]]
                        # If f_right < val_center, search in [x_center[i], x_right[i]]
                        
                        def f_1d(val):
                            x_tmp = x_center.copy()
                            x_tmp[i] = val
                            return safe_eval(x_tmp)
                        
                        if f_left < val_center:
                            a = x_left[i]
                            b = x_center[i]
                            # We have f(a) = f_left, f(b) = val_center
                            c = b - inv_phi * (b - a)
                            d = a + inv_phi * (b - a)
                            fc = f_1d(c)
                            fd = f_1d(d)
                            
                            for _ in range(5):
                                if fc < fd:
                                    b = d
                                    d = c
                                    fd = fc
                                    c = b - inv_phi * (b - a)
                                    fc = f_1d(c)
                                else:
                                    a = c
                                    c = d
                                    fc = fd
                                    d = a + inv_phi * (b - a)
                                    fd = f_1d(d)
                            
                            if fc < fd:
                                best_val_1d = c
                            else:
                                best_val_1d = d
                            
                            new_x = x_center.copy()
                            new_x[i] = best_val_1d
                            new_f = f_1d(best_val_1d)
                            if new_f < best_f:
                                best_x = new_x
                                best_f = new_f
                                improved = True
                        else:
                            a = x_center[i]
                            b = x_right[i]
                            c = b - inv_phi * (b - a)
                            d = a + inv_phi * (b - a)
                            fc = f_1d(c)
                            fd = f_1d(d)
                            
                            for _ in range(5):
                                if fc < fd:
                                    b = d
                                    d = c
                                    fd = fc
                                    c = b - inv_phi * (b - a)
                                    fc = f_1d(c)
                                else:
                                    a = c
                                    c = d
                                    fc = fd
                                    d = a + inv_phi * (b - a)
                                    fd = f_1d(d)
                            
                            if fc < fd:
                                best_val_1d = c
                            else:
                                best_val_1d = d
                            
                            new_x = x_center.copy()
                            new_x[i] = best_val_1d
                            new_f = f_1d(best_val_1d)
                            if new_f < best_f:
                                best_x = new_x
                                best_f = new_f
                                improved = True
                            
                if not improved:
                    break
            # End while loop

        return best_x, best_f
