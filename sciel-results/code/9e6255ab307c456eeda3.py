import math
import time
import numpy as np
from scipy.optimize import brentq

def find_root(f, x0):
    start_time = time.time()
    MAX_WALL = 115.0
    MAX_EVALS = 500
    TARGET_TOL = 1e-8
    
    eval_count = [0]
    cache = {}
    
    def time_ok():
        return time.time() - start_time < MAX_WALL

    def eval_f(x):
        if not time_ok():
            raise TimeoutError()
        if eval_count[0] >= MAX_EVALS:
            raise ValueError("Eval limit exceeded")
        
        # Use a tuple key to handle floating point precision in cache
        # Rounding to 15 digits is usually safe for cache hits without causing collisions
        key = round(float(x), 15)
        
        if key in cache:
            return cache[key]
            
        val = f(float(x))
        cache[key] = val
        eval_count[0] += 1
        
        if not math.isfinite(val):
            raise ValueError("Non-finite result")
            
        return val

    x0 = float(x0)
    best_x = x0
    best_fx = float('inf')
    
    try:
        fx0 = eval_f(x0)
        best_fx = fx0
    except Exception:
        return x0
        
    if abs(fx0) <= TARGET_TOL:
        return x0

    # 1. Generate a set of diverse starting points
    # We want to cover local variations and potentially escape local minima of |f|
    # or find brackets for different branches if f is non-monotonic.
    starts = [x0]
    scales = [1e-4, 1e-3, 1e-2, 0.1, 1.0]
    for s in scales:
        h = s * max(1.0, abs(x0))
        starts.append(x0 + h)
        starts.append(x0 - h)
    
    # 2. Parallel-like execution (sequential but independent tracks)
    # We run a limited number of steps for each track.
    # Track types: Newton, Secant, Bisection (if bracket found)
    
    # We'll maintain a list of "active" tracks.
    # Each track has: type, x1, f1, x2, f2, h (for newton), iters
    tracks = []
    
    for sx in starts:
        try:
            fsx = eval_f(sx)
            if abs(fsx) < abs(best_fx):
                best_fx = fsx
                best_x = sx
            if abs(fsx) <= TARGET_TOL:
                return sx
                
            # Add a Newton track
            tracks.append({'type': 'newton', 'x': sx, 'f': fsx, 'h': 1e-5 * max(1.0, abs(sx)), 'iters': 0})
            
            # Add a Secant track (needs 2 points, so we'll pair it with x0 if distinct)
            if sx != x0:
                tracks.append({'type': 'secant', 'x1': sx, 'f1': fsx, 'x2': x0, 'f2': fx0, 'iters': 0})
                
        except Exception:
            continue

    # Run tracks for a fixed number of iterations or until convergence
    MAX_ITERS_PER_TRACK = 10
    
    while tracks and time_ok() and eval_count[0] < MAX_EVALS - 5:
        new_tracks = []
        for t in tracks:
            if not time_ok() or eval_count[0] >= MAX_EVALS - 5:
                break
                
            try:
                if t['type'] == 'newton':
                    x, fx = t['x'], t['f']
                    h = t['h']
                    
                    # Finite difference derivative
                    if eval_count[0] >= MAX_EVALS - 1:
                        break
                    fx_plus = eval_f(x + h)
                    df = (fx_plus - fx) / h
                    
                    if not math.isfinite(df) or abs(df) < 1e-15:
                        # Try larger h
                        h_new = h * 10
                        if eval_count[0] >= MAX_EVALS - 1:
                            break
                        fx_plus = eval_f(x + h_new)
                        df = (fx_plus - fx) / h_new
                        if not math.isfinite(df) or abs(df) < 1e-15:
                            continue # Skip this step, keep track alive? Or kill?
                        h = h_new
                    else:
                        h = h * 0.5 # Shrink h if it worked? Or keep? 
                        # Actually, standard Newton uses fixed h or adaptive. 
                        # Let's stick to a reasonable h.
                        
                    step = fx / df
                    if not math.isfinite(step):
                        continue
                        
                    # Damping
                    max_step = 10 * max(1.0, abs(x))
                    if abs(step) > max_step:
                        step = math.copysign(max_step, step)
                        
                    x_new = x - step
                    if not math.isfinite(x_new):
                        continue
                        
                    if eval_count[0] >= MAX_EVALS:
                        break
                    fx_new = eval_f(x_new)
                    
                    if abs(fx_new) < abs(best_fx):
                        best_x = x_new
                        best_fx = fx_new
                        
                    if abs(fx_new) <= TARGET_TOL:
                        return x_new
                        
                    # Check for divergence
                    if abs(fx_new) > 10 * abs(fx):
                        # Diverging, kill track
                        continue
                        
                    # Update track
                    new_tracks.append({
                        'type': 'newton', 
                        'x': x_new, 
                        'f': fx_new, 
                        'h': 1e-5 * max(1.0, abs(x_new)), 
                        'iters': t['iters'] + 1
                    })
                    
                elif t['type'] == 'secant':
                    x1, f1, x2, f2 = t['x1'], t['f1'], t['x2'], t['f2']
                    
                    if abs(x1 - x2) < 1e-15 * max(1.0, abs(x1)):
                        continue # Points too close, secant fails
                        
                    denom = f1 - f2
                    if abs(denom) < 1e-15:
                        continue
                        
                    x_new = x1 - f1 * (x1 - x2) / denom
                    if not math.isfinite(x_new):
                        continue
                        
                    if eval_count[0] >= MAX_EVALS:
                        break
                    f_new = eval_f(x_new)
                    
                    if abs(f_new) < abs(best_fx):
                        best_x = x_new
                        best_fx = f_new
                        
                    if abs(f_new) <= TARGET_TOL:
                        return x_new
                        
                    # Check for divergence
                    if abs(f_new) > 10 * abs(f1):
                        continue
                        
                    # Update track: drop the older point (x2)
                    new_tracks.append({
                        'type': 'secant', 
                        'x1': x_new, 
                        'f1': f_new, 
                        'x2': x1, 
                        'f2': f1, 
                        'iters': t['iters'] + 1
                    })
                        
            except Exception:
                continue
                
        tracks = new_tracks
        if not tracks:
            break

    # 3. Bracketing and Brent's Method
    # Try to find a bracket using the points we've evaluated
    # We have x0, fx0 and many other points in cache.
    # Let's check pairs in cache for sign changes.
    
    points = list(cache.keys())
    # Sort points to make bracketing more systematic? Or just check neighbors?
    # Checking all pairs is O(N^2) but N is small (<= 500).
    # However, we can just check points near x0 or best_x.
    
    bracket_found = False
    a = None
    b = None
    fa = None
    fb = None
    
    # Sort points to find adjacent sign changes
    points.sort()
    
    for i in range(len(points) - 1):
        if not time_ok() or eval_count[0] >= MAX_EVALS - 10:
            break
        p1 = points[i]
        p2 = points[i+1]
        
        # Skip if points are too far apart? 
        # For ill-conditioned problems, the root might be in a narrow interval.
        # But if we have a sign change, Brent is robust.
        # Let's restrict to points within a reasonable range of x0 or best_x
        # to avoid picking up distant roots if f has multiple roots.
        # "Local" test equations imply we should stay local.
        # Let's define a local window around x0.
        window = 10 * max(1.0, abs(x0))
        if abs(p1 - x0) > window or abs(p2 - x0) > window:
            continue
            
        f1 = cache[round(p1, 15)]
        f2 = cache[round(p2, 15)]
        
        if f1 * f2 < 0:
            a, b = p1, p2
            fa, fb = f1, f2
            bracket_found = True
            break

    if bracket_found:
        def f_brent(x):
            if not time_ok():
                raise ValueError("Time limit exceeded")
            if eval_count[0] >= MAX_EVALS:
                raise ValueError("Eval limit exceeded")
            return eval_f(x)

        try:
            root = brentq(f_brent, a, b, xtol=1e-12, rtol=1e-12, maxiter=100)
            # Verify
            if abs(f_brent(root)) <= TARGET_TOL:
                return root
        except Exception:
            pass

    # 4. Final Polish Pass
    # If we didn't converge, take the best_x found and try to refine it
    # using a very tight Newton step or secant step with high precision derivatives
    # if we have budget.
    
    if time_ok() and eval_count[0] < MAX_EVALS - 2:
        x = best_x
        try:
            fx = eval_f(x)
            # Try a few steps of Newton with very small h
            for h in [1e-8, 1e-9, 1e-10]:
                if not time_ok() or eval_count[0] >= MAX_EVALS - 1:
                    break
                try:
                    fx_plus = eval_f(x + h)
                    df = (fx_plus - fx) / h
                    if abs(df) > 1e-15:
                        step = fx / df
                        if abs(step) < 1e-5: # Only take small steps
                            x_new = x - step
                            fx_new = eval_f(x_new)
                            if abs(fx_new) < abs(fx):
                                x = x_new
                                fx = fx_new
                except Exception:
                    break
        except Exception:
            pass

    return best_x
