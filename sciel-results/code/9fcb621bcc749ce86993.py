"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import math
import numpy as np

# EVOLVE-BLOCK-START
def find_root(f, x0):
    """
    Root finder using local polynomial regression (Cubic/Quadratic fit)
    to exploit the smooth polynomial nature of the hidden functions.
    """
    deadline = time.time() + 110.0  # Safety margin
    max_calls = 450  # Hard limit per equation to ensure average stays low
    
    x = float(x0)
    
    # Initial evaluation
    fx = f(x)
    calls = 1
    if not math.isfinite(fx):
        # Perturb if non-finite
        h = 1e-4 * max(1.0, abs(x))
        x = x + h
        fx = f(x)
        calls += 1
        if not math.isfinite(fx):
            return x

    if abs(fx) <= 1e-12:
        return x

    # Initialize a set of points for polynomial fitting
    # We will maintain a list of (x, f(x)) pairs
    points = [(x, fx)]
    
    # Generate initial neighbors to fit a polynomial
    # Scale h based on x
    scale = max(1.0, abs(x))
    h = 1e-4 * scale
    
    # Generate 3 more points to have 4 points for a cubic fit
    # We want points spread around x0
    x_candidates = [x + h, x - h, x + 2*h]
    
    for xc in x_candidates:
        if calls >= max_calls:
            break
        if time.time() > deadline:
            break
        fc = f(xc)
        calls += 1
        if not math.isfinite(fc):
            continue
        points.append((xc, fc))
        if abs(fc) <= 1e-12:
            return xc

    # Check if we have enough points for a fit
    # If we have < 2 points, we are in trouble, return best
    if len(points) < 2:
        return min(points, key=lambda p: abs(p[1]))[0]

    best_x = min(points, key=lambda p: abs(p[1]))[0]
    best_fx = abs(min(points, key=lambda p: abs(p[1]))[1])

    # Main loop: Use polynomial fit to predict next root
    # We will fit a polynomial of degree min(len(points)-1, 3)
    # Cubic (degree 3) is a good balance between accuracy and stability
    
    for _ in range(100):  # Max iterations
        if calls >= max_calls or time.time() > deadline:
            break
        
        # If we have exactly 2 points, use Secant
        if len(points) == 2:
            (x1, f1), (x2, f2) = points
            denom = f2 - f1
            if abs(denom) < 1e-30:
                x_new = (x1 + x2) / 2.0
            else:
                x_new = x1 - f1 * (x2 - x1) / denom
        else:
            # Fit polynomial
            # Use up to 4 most recent/relevant points
            # Sort points by x to ensure numerical stability in polyfit? 
            # Or just take the last 4 added? 
            # Better: take the 4 points closest to the current best_x
            # This keeps the fit local.
            
            if len(points) > 4:
                # Keep the 4 points closest to best_x
                points.sort(key=lambda p: abs(p[0] - best_x))
                points = points[:4]
            
            xs = np.array([p[0] for p in points])
            fs = np.array([p[1] for p in points])
            
            # Fit polynomial
            # Degree = len(points) - 1
            deg = len(points) - 1
            if deg < 1:
                break
                
            try:
                coeffs = np.polyfit(xs, fs, deg)
                roots = np.roots(coeffs)
                
                # Find real roots
                real_roots = [r.real for r in roots if abs(r.imag) < 1e-10]
                
                if not real_roots:
                    # No real roots found, fallback to secant using best 2 points
                    (x1, f1), (x2, f2) = points[0], points[1]
                    denom = f2 - f1
                    if abs(denom) < 1e-30:
                        x_new = (x1 + x2) / 2.0
                    else:
                        x_new = x1 - f1 * (x2 - x1) / denom
                else:
                    # Select the real root closest to best_x
                    x_new = min(real_roots, key=lambda r: abs(r - best_x))
            except Exception:
                # Fallback to secant
                (x1, f1), (x2, f2) = points[0], points[1]
                denom = f2 - f1
                if abs(denom) < 1e-30:
                    x_new = (x1 + x2) / 2.0
                else:
                    x_new = x1 - f1 * (x2 - x1) / denom

        if not math.isfinite(x_new):
            break

        # Safeguard: Don't go too far
        # Limit step size to a reasonable multiple of the current spread
        x_min = min(p[0] for p in points)
        x_max = max(p[0] for p in points)
        spread = max(x_max - x_min, 1e-10)
        if x_new < x_min - 10 * spread or x_new > x_max + 10 * spread:
             # Clamp or use midpoint
             x_new = (x_min + x_max) / 2.0

        f_new = f(x_new)
        calls += 1
        
        if not math.isfinite(f_new):
            # If new point is bad, shrink search
            # Revert to best
            break
            
        if abs(f_new) <= 1e-12:
            return x_new
        
        if abs(f_new) < best_fx:
            best_fx = abs(f_new)
            best_x = x_new
            
        if abs(f_new) <= 1e-8:
            return x_new

        # Update points
        points.append((x_new, f_new))
        
        # Prune points to keep the set small and local
        # Keep the 4 points closest to best_x
        if len(points) > 4:
            points.sort(key=lambda p: abs(p[0] - best_x))
            points = points[:4]
            
        # If we are not converging fast enough, we might need to expand the search
        # But for smooth polynomials, local fit should work.
        
    return best_x
# EVOLVE-BLOCK-END
