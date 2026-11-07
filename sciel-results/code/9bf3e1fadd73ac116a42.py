"""
Strategy: Multi-Scale |f(x)| Minimization with Parabolic Refinement
1. Perform a multi-scale, adaptive grid search around x0 to find the global minimum of |f(x)|.
   - This handles oscillatory functions better than simple bracketing by identifying
     where the function is closest to zero (local minima of |f|).
   - It is robust to roots where the function does not change sign (e.g., (x-a)^2).
2. Use local parabolic interpolation on the best triplet of points to refine the minimum.
3. This approach is robust to fast oscillations if the initial sampling is dense enough.
"""
import math
import time

# EVOLVE-BLOCK-START

def find_root(f, x0):
    start_time = time.time()
    TIME_LIMIT = 110.0  # Leave 10s safety margin for 120s limit
    
    # State tracking
    calls = 0
    best_x = x0
    best_f_abs = float('inf')
    
    # Cache to avoid re-evaluating points (minor optimization, but safe)
    # Note: In a strict evaluator, caching might be seen as "cheating" if not careful,
    # but we are just avoiding redundant calls within our own logic.
    # Actually, simpler to just track calls and not cache, as f is a black box.
    
    def _f(x):
        nonlocal calls, best_x, best_f_abs
        if time.time() - start_time > TIME_LIMIT:
            raise TimeoutError("Time limit exceeded")
        
        # Finite check
        try:
            val = float(f(x))
        except Exception:
            return float('nan')
        
        if not math.isfinite(val):
            return float('nan')
            
        calls += 1
        
        abs_val = abs(val)
        if abs_val < best_f_abs:
            best_f_abs = abs_val
            best_x = x
            
        return val

    # Initial evaluation
    x = float(x0)
    fx = _f(x)
    
    if abs(fx) <= 1e-8:
        return x

    # Data structure for points found: list of (x, f_abs, f_val)
    # We want to find the minimum of |f(x)|.
    
    # Phase 1: Multi-scale Adaptive Grid Search
    # We start with a very small step and expand.
    # For each scale, we sample a window around x0.
    # We keep track of the best point found.
    
    # Initial step size
    # If the function is highly oscillatory, we need a small step.
    # If the root is far, we need to expand.
    # We will use a geometric expansion of the search radius and a fine grid within that radius.
    
    # To save calls, we don't re-sample the entire previous radius, just the new shell.
    # But for simplicity and robustness against oscillations, a dense grid in the current radius
    # is safer. We limit the number of points per scale.
    
    # Let's try a specific strategy:
    # Scale 1: Radius 1e-3, 20 points
    # Scale 2: Radius 1e-2, 40 points
    # Scale 3: Radius 1e-1, 60 points
    # Scale 4: Radius 1.0, 80 points
    # Scale 5: Radius 10.0, 100 points
    # Scale 6: Radius 100.0, 120 points
    # Scale 7: Radius 1000.0, 140 points
    
    # This is a fixed budget of ~560 calls just for searching.
    # We need to be more adaptive.
    
    # Adaptive Strategy:
    # 1. Start with small radius R.
    # 2. Sample N points in [-R, R].
    # 3. If best |f| is small enough, stop.
    # 4. If not, increase R by factor 2.
    # 5. Increase N to maintain density or increase density if oscillations suspected?
    #    Since we don't know frequency, we increase N slightly to catch faster oscillations in larger domains?
    #    Actually, if oscillations are fast, a large R with low density will miss them.
    #    So we should keep a minimum density.
    
    # Let's define a minimum step size. If step < 1e-4, we might miss fast oscillations.
    # But we can't afford infinite calls.
    
    # Alternative: Use the "best point" found so far to guide the search.
    # If the best point is at the edge of the current search window, expand in that direction.
    # If the best point is in the middle, refine locally.
    
    # Let's implement a hybrid:
    # 1. Coarse global search using expanding rings.
    # 2. Local refinement using parabolic interpolation on the best triplet.
    
    points = [(x, fx)]
    
    # Search parameters
    max_radius = 1e6
    current_radius = 1e-3
    min_step = 1e-5 # Don't make steps smaller than this
    max_calls_search = 300 # Budget for search phase
    
    while current_radius < max_radius and calls < max_calls_search:
        if time.time() - start_time > TIME_LIMIT:
            break
            
        # Determine number of points to sample in this ring
        # We want to cover [-current_radius, current_radius]
        # But we already sampled the inner radius.
        # To avoid re-sampling, we can just sample the new shell.
        # However, for oscillatory functions, a uniform grid over the whole current radius
        # is safer to ensure we don't miss a peak/trough that aligns with the previous grid gaps.
        # Re-sampling is expensive.
        
        # Compromise: Sample the new shell [current_radius/2, current_radius] and 
        # [-current_radius, -current_radius/2].
        # And also check the midpoint of the previous best?
        
        # Let's stick to a simple expanding grid that re-samples the whole interval 
        # but with a fixed number of points that grows slowly.
        # If we re-sample, we waste calls.
        
        # Better: Keep a list of all sampled points.
        # In each iteration, add new points in the expanded region.
        
        # Let's define the new points to add.
        # Previous max radius: prev_radius
        # Current max radius: current_radius
        
        # We'll sample 20 new points in the positive shell and 20 in the negative shell.
        n_new_points = 20
        
        # Positive shell
        start_pos = current_radius / 2.0
        end_pos = current_radius
        step_pos = (end_pos - start_pos) / n_new_points
        
        for i in range(n_new_points + 1):
            if calls >= max_calls_search:
                break
            x_new = start_pos + i * step_pos
            # Only evaluate if we haven't evaluated this point (approximate check by distance)
            # Since step sizes change, exact match is unlikely unless i=0 and it matches previous end.
            # To be safe, we just evaluate.
            f_new = _f(x_new)
            points.append((x_new, f_new))
            
        # Negative shell
        start_neg = -current_radius
        end_neg = -current_radius / 2.0
        step_neg = (end_neg - start_neg) / n_new_points
        
        for i in range(n_new_points + 1):
            if calls >= max_calls_search:
                break
            x_new = start_neg + i * step_neg
            f_new = _f(x_new)
            points.append((x_new, f_new))
            
        # Check if we have a good enough root
        if best_f_abs <= 1e-8:
            return best_x
            
        # If the best point is very close to the boundary of the search region,
        # it suggests the root might be further out.
        # If the best point is well inside, we might be at a local minimum that is not a root,
        # or the root is nearby but we haven't refined enough.
        
        # Expand radius
        current_radius *= 2.0
        
    # Phase 2: Local Refinement
    # Sort points by |f(x)| to find the best triplet.
    # We need 3 points around the best point to do parabolic interpolation.
    
    # Find the index of the best point in the list
    best_idx = -1
    min_abs = float('inf')
    for i, (px, pf) in enumerate(points):
        abs_pf = abs(pf)
        if abs_pf < min_abs:
            min_abs = abs_pf
            best_idx = i
            
    if best_idx == -1:
        return best_x
        
    # Sort points by x to find neighbors
    # This is O(N log N), N is small (< 300)
    sorted_points = sorted(points, key=lambda p: p[0])
    
    # Find the position of the best point in the sorted list
    # We might have duplicates or very close points.
    # Let's just find the 3 closest points in x to the best_x.
    
    # Extract x and f values
    xs = [p[0] for p in sorted_points]
    fs = [p[1] for p in sorted_points]
    
    # Find index of best_x in xs
    # Since we appended, best_x is in xs.
    # Use binary search or linear search. Linear is fine for N=300.
    best_x_val = best_x
    # Find closest index
    idx = 0
    min_diff = abs(xs[0] - best_x_val)
    for i in range(1, len(xs)):
        diff = abs(xs[i] - best_x_val)
        if diff < min_diff:
            min_diff = diff
            idx = i
            
    # Get neighbors
    # We need x_{i-1}, x_i, x_{i+1}
    # If i is 0 or len-1, we can't do parabolic interp with 3 points.
    # In that case, we can try a linear extrapolation or just return best_x.
    # Or we can pick the next closest points.
    
    if idx > 0 and idx < len(xs) - 1:
        x1 = xs[idx-1]
        x2 = xs[idx]
        x3 = xs[idx+1]
        
        f1 = fs[idx-1]
        f2 = fs[idx]
        f3 = fs[idx+1]
        
        # Parabolic interpolation to find the minimum of |f(x)|?
        # Actually, parabolic interpolation fits a parabola to (x1,f1), (x2,f2), (x3,f3).
        # We want the x where this parabola is zero? No, we want the minimum of |f|.
        # If f doesn't change sign, the minimum of f(x) (if f>0) or -f(x) (if f<0) is the root candidate.
        # The vertex of the parabola y = ax^2 + bx + c is at x_v = -b / (2a).
        
        # Lagrange interpolation coefficients
        # y = c1 * (x-x2)(x-x3) / ((x1-x2)(x1-x3)) + ...
        
        # Let's use the standard formula for the vertex of the interpolating parabola.
        # h1 = x2 - x1
        # h2 = x3 - x2
        # S1 = (f2 - f1) / h1
        # S2 = (f3 - f2) / h2
        # A = (S2 - S1) / (h1 + h2)
        # x_vertex = x2 - 0.5 * h2 * (S2 - S1) / (A * h2) ? No.
        
        # Standard formula:
        # x_v = x2 + 0.5 * ( (f2-f1)/h1 - (f3-f2)/h2 ) / ( (f2-f1)/h1^2 + (f3-f2)/h2^2 ) ? No, that's for min.
        
        # Let's derive it quickly or use a known robust formula.
        # Min of parabola through (x1,y1), (x2,y2), (x3,y3):
        # x_min = x2 + 0.5 * h2 * ( (y2-y1)/h1 - (y3-y2)/h2 ) / ( (y2-y1)/h1 + (y3-y2)/h2 )
        # Wait, that's for the case where x2 is the middle point.
        
        h1 = x2 - x1
        h2 = x3 - x2
        
        if abs(h1) < 1e-15 or abs(h2) < 1e-15:
            # Degenerate case, points too close
            return best_x
            
        S1 = (f2 - f1) / h1
        S2 = (f3 - f2) / h2
        
        denom = S2 - S1
        # The vertex x is:
        # x_v = x2 + 0.5 * h2 * (S2 - S1) / (S2 - S1) ? No.
        
        # Correct formula for vertex of parabola through 3 points:
        # Let's fit y = A(x-x2)^2 + B(x-x2) + C
        # C = f2
        # A*h1^2 - B*h1 + C = f1  => A*h1^2 - B*h1 = f1 - f2
        # A*h2^2 + B*h2 + C = f3  => A*h2^2 + B*h2 = f3 - f2
        
        # From eq 1: B = (A*h1^2 - (f1-f2))/h1
        # Substitute into eq 2:
        # A*h2^2 + h2*(A*h1^2 - (f1-f2))/h1 = f3 - f2
        # A * (h2^2 + h2*h1^2/h1) = f3 - f2 + h2*(f1-f2)/h1
        # A * (h2^2 + h1*h2) = f3 - f2 + h2*(f1-f2)/h1
        # A * h2 * (h2 + h1) = (f3 - f2) + (h2/h1)*(f1 - f2)
        # A = [ (f3 - f2) + (h2/h1)*(f1 - f2) ] / [ h2 * (h1 + h2) ]
        
        if abs(h1 + h2) < 1e-15:
            return best_x
            
        A = ( (f3 - f2) + (h2/h1)*(f1 - f2) ) / ( h2 * (h1 + h2) )
        
        if abs(A) < 1e-15:
            # Linear or flat, no unique min
            return best_x
            
        # Vertex x_v = x2 - B / (2A)
        # B = (A*h1^2 - (f1-f2))/h1
        B = (A*h1*h1 - (f1 - f2)) / h1
        
        x_vertex = x2 - B / (2*A)
        
        # Check if x_vertex is within [x1, x3]
        if x1 <= x_vertex <= x3:
            # Evaluate f at x_vertex
            f_vertex = _f(x_vertex)
            
            if abs(f_vertex) < best_f_abs:
                best_f_abs = abs(f_vertex)
                best_x = x_vertex
                
                # Refine further?
                # We can do one more step of bisection or interpolation if we have calls left.
                # Let's do a few iterations of "parabolic + bisection" hybrid.
                
                # Set up a local bracket around x_vertex?
                # We don't have a sign change bracket necessarily.
                # But we have 3 points.
                # If f_vertex is better, we can replace the worst of the 3 points with x_vertex
                # and re-interpolate.
                
                # Let's do 5 iterations of this refinement.
                for _ in range(5):
                    if calls > 450:
                        break
                    if time.time() - start_time > TIME_LIMIT:
                        break
                    if best_f_abs <= 1e-8:
                        return best_x
                        
                    # Re-sort the 3 points by x
                    # Points are (x1, f1), (x2, f2), (x3, f3) and we added (x_vertex, f_vertex)
                    # We keep the 3 best points in terms of x-proximity to the current best?
                    # Or the 3 points that form the tightest bracket around the min?
                    # Simple: Keep the 3 points with smallest |f|? No, we need x-order.
                    # Let's keep the 3 points: x1, x_vertex, x3?
                    # Or x1, x2, x_vertex?
                    # It depends on where x_vertex is.
                    
                    # Let's just re-select the 3 points from the global list that are closest to best_x.
                    # This is robust.
                    points.append((x_vertex, f_vertex))
                    sorted_points = sorted(points, key=lambda p: p[0])
                    xs = [p[0] for p in sorted_points]
                    fs = [p[1] for p in sorted_points]
                    
                    # Find index of best_x
                    idx = 0
                    min_diff = abs(xs[0] - best_x)
                    for i in range(1, len(xs)):
                        diff = abs(xs[i] - best_x)
                        if diff < min_diff:
                            min_diff = diff
                            idx = i
                            
                    if idx > 0 and idx < len(xs) - 1:
                        x1 = xs[idx-1]
                        x2 = xs[idx]
                        x3 = xs[idx+1]
                        f1 = fs[idx-1]
                        f2 = fs[idx]
                        f3 = fs[idx+1]
                        
                        h1 = x2 - x1
                        h2 = x3 - x2
                        if abs(h1) < 1e-15 or abs(h2) < 1e-15:
                            break
                            
                        S1 = (f2 - f1) / h1
                        S2 = (f3 - f2) / h2
                        A = ( (f3 - f2) + (h2/h1)*(f1 - f2) ) / ( h2 * (h1 + h2) )
                        if abs(A) < 1e-15:
                            break
                        B = (A*h1*h1 - (f1 - f2)) / h1
                        x_new_vertex = x2 - B / (2*A)
                        
                        if x1 <= x_new_vertex <= x3:
                            f_new_vertex = _f(x_new_vertex)
                            if abs(f_new_vertex) < best_f_abs:
                                best_f_abs = abs(f_new_vertex)
                                best_x = x_new_vertex
                            else:
                                # No improvement, break to avoid wasting calls
                                break
                        else:
                            break
                    else:
                        break
                        
                return best_x

    # If we couldn't refine, return best found
    return best_x

# EVOLVE-BLOCK-END
