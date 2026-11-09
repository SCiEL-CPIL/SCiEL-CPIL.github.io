"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import math

# EVOLVE-BLOCK-START
def find_root(f, x0):
    # Deadline guard: kill after ~115s to stay under 120s wall-clock
    t_start = time.time()
    deadline = 115.0
    
    eval_count = 0
    max_evals = 500
    
    def f_counted(x):
        nonlocal eval_count
        eval_count += 1
        if eval_count > max_evals:
            raise ValueError("Max evaluations exceeded")
        return f(x)
    
    x = float(x0)
    
    # First evaluation
    fx = f_counted(x)
    best_x = x
    best_fx = fx
    best_abs_fx = abs(fx)
    
    if best_abs_fx <= 1e-8:
        return x
    
    # IQI (Inverse Quadratic Interpolation) method
    # We maintain three points and use quadratic interpolation to find the next root candidate.
    
    # Start with small perturbations to get three points
    h = 1e-4 * max(1.0, abs(x))
    x1 = x + h
    x2 = x - h
    f1 = f_counted(x1)
    f2 = f_counted(x2)
    
    if abs(f1) <= 1e-8:
        return x1
    if abs(f2) <= 1e-8:
        return x2
    
    # Track best
    if abs(f1) < best_abs_fx:
        best_x = x1
        best_abs_fx = abs(f1)
    if abs(f2) < best_abs_fx:
        best_x = x2
        best_abs_fx = abs(f2)
    
    # Maintain three points for IQI
    # Sort by |f| to keep the best three
    points = [(x, fx), (x1, f1), (x2, f2)]
    points.sort(key=lambda p: abs(p[1]))
    
    for i in range(max_evals - 3):
        # Check time
        if time.time() - t_start > deadline:
            break
        
        # Check if we've converged
        for px, pf in points:
            if abs(pf) <= 1e-8:
                return px
        
        # Extract the three points
        x0_s, f0_s = points[0]
        x1_s, f1_s = points[1]
        x2_s, f2_s = points[2]
        
        # Try IQI first, fall back to bisection if it fails
        x_new = None
        
        # IQI: fit parabola through three points and find where it crosses zero
        # Using Lagrange interpolation formula
        # f(x) ≈ L(x) where L is quadratic
        # Find x such that L(x) = 0
        
        # Check for degenerate cases
        if abs(x1_s - x0_s) > 1e-15 and abs(x2_s - x0_s) > 1e-15 and abs(x2_s - x1_s) > 1e-15:
            # Compute coefficients for quadratic interpolation
            # L(x) = a*(x-x1)*(x-x2) + b*(x-x0)*(x-x2) + c*(x-x0)*(x-x1)
            # where a = f0/((x0-x1)*(x0-x2)), etc.
            
            a = f0_s / ((x0_s - x1_s) * (x0_s - x2_s))
            b = f1_s / ((x1_s - x0_s) * (x1_s - x2_s))
            c = f2_s / ((x2_s - x0_s) * (x2_s - x1_s))
            
            # The quadratic is: A*x^2 + B*x + C = 0
            # Expand: a*(x^2 - (x1+x2)*x + x1*x2) + b*(x^2 - (x0+x2)*x + x0*x2) + c*(x^2 - (x0+x1)*x + x0*x1)
            # = (a+b+c)*x^2 - [a*(x1+x2) + b*(x0+x2) + c*(x0+x1)]*x + [a*x1*x2 + b*x0*x2 + c*x0*x1]
            
            A = a + b + c
            B = -(a * (x1_s + x2_s) + b * (x0_s + x2_s) + c * (x0_s + x1_s))
            C = a * x1_s * x2_s + b * x0_s * x2_s + c * x0_s * x1_s
            
            if abs(A) > 1e-15:
                # Quadratic formula
                disc = B * B - 4 * A * C
                if disc >= 0:
                    sqrt_disc = math.sqrt(disc)
                    x_candidate1 = (-B + sqrt_disc) / (2 * A)
                    x_candidate2 = (-B - sqrt_disc) / (2 * A)
                    
                    # Choose the candidate that's closer to the current best
                    # and within reasonable bounds
                    candidates = []
                    if math.isfinite(x_candidate1):
                        candidates.append(x_candidate1)
                    if math.isfinite(x_candidate2):
                        candidates.append(x_candidate2)
                    
                    if candidates:
                        # Pick the one closest to the point with smallest |f|
                        x_new = min(candidates, key=lambda xc: abs(xc - x0_s))
                else:
                    x_new = None
            else:
                # Linear case
                if abs(B) > 1e-15:
                    x_new = -C / B
                else:
                    x_new = None
        
        # Fallback to bisection if IQI failed or gives unreasonable value
        if x_new is None or not math.isfinite(x_new) or abs(x_new - x0_s) > 10 * max(1.0, abs(x0_s)):
            # Bisection: find sign change among the three points
            # Sort by x value
            sorted_points = sorted(points, key=lambda p: p[0])
            (xa, fa), (xb, fb), (xc, fc) = sorted_points
            
            # Check for sign changes
            if fa * fb < 0:
                x_new = (xa + xb) / 2
            elif fb * fc < 0:
                x_new = (xb + xc) / 2
            elif fa * fc < 0:
                x_new = (xa + xc) / 2
            else:
                # No sign change; use inverse linear interpolation on the two closest points
                # Pick the two points with smallest |f|
                p1, p2 = points[0], points[1]
                if abs(p2[1] - p1[1]) > 1e-15:
                    x_new = p1[0] - p1[1] * (p2[0] - p1[0]) / (p2[1] - p1[1])
                else:
                    # Flat region; perturb
                    x_new = x0_s + 1e-6 * max(1.0, abs(x0_s))
        
        # Check for NaN/Inf
        if not math.isfinite(x_new):
            break
        
        # Evaluate
        f_new = f_counted(x_new)
        
        if not math.isfinite(f_new):
            break
        
        # Update best
        if abs(f_new) < best_abs_fx:
            best_x = x_new
            best_abs_fx = abs(f_new)
            if best_abs_fx <= 1e-8:
                return x_new
        
        # Update points: keep the three points with smallest |f|
        points.append((x_new, f_new))
        points.sort(key=lambda p: abs(p[1]))
        points = points[:3]
        
        # Check convergence
        for px, pf in points:
            if abs(pf) <= 1e-8:
                return px
    
    # If IQI didn't work well, try a local search / golden section on |f(x)|
    # to find a minimum of |f| near the current best point
    if time.time() - t_start < deadline - 5.0:
        # Golden section search on |f(x)| in a window around best_x
        a = best_x - 1e-3 * max(1.0, abs(best_x))
        b = best_x + 1e-3 * max(1.0, abs(best_x))
        
        # Ensure we have valid evaluations
        try:
            fa = f_counted(a)
            fb = f_counted(b)
        except ValueError:
            return best_x
        
        if abs(fa) <= 1e-8:
            return a
        if abs(fb) <= 1e-8:
            return b
        
        if abs(fa) < best_abs_fx:
            best_x = a
            best_abs_fx = abs(fa)
        if abs(fb) < best_abs_fx:
            best_x = b
            best_abs_fx = abs(fb)
        
        # Golden section
        gr = (math.sqrt(5) - 1) / 2  # ~0.618
        c = b - gr * (b - a)
        d = a + gr * (b - a)
        
        try:
            fc = f_counted(c)
            fd = f_counted(d)
        except ValueError:
            return best_x
        
        if abs(fc) < best_abs_fx:
            best_x = c
            best_abs_fx = abs(fc)
        if abs(fd) < best_abs_fx:
            best_x = d
            best_abs_fx = abs(fd)
        
        # Iterate a few times
        for _ in range(20):
            if time.time() - t_start > deadline - 2.0:
                break
            if eval_count >= max_evals - 5:
                break
            
            if abs(fc) < abs(fd):
                b = d
                d = c
                fd = fc
                c = b - gr * (b - a)
                try:
                    fc = f_counted(c)
                except ValueError:
                    break
            else:
                a = c
                c = d
                fc = fd
                d = a + gr * (b - a)
                try:
                    fd = f_counted(d)
                except ValueError:
                    break
            
            if abs(fc) < best_abs_fx:
                best_x = c
                best_abs_fx = abs(fc)
            if abs(fd) < best_abs_fx:
                best_x = d
                best_abs_fx = abs(fd)
            
            if best_abs_fx <= 1e-8:
                return best_x
    
    return best_x
# EVOLVE-BLOCK-END
