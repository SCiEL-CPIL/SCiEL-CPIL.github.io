"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

# EVOLVE-BLOCK-START
def find_root(f, x0):
    # Deadline guard: 120s limit, use 110s as safety margin
    start_time = time.time()
    deadline = start_time + 110.0
    
    x0 = float(x0)
    
    # Strategy: Single-shot quintic (degree 5) polynomial reconstruction.
    # We evaluate f at 6 distinct points to determine the coefficients of a 
    # degree-5 polynomial that matches f at those points.
    # Then we find the roots of this polynomial and select the one nearest to x0.
    # This is exact if f is a polynomial of degree <= 5.
    # For higher degree polynomials, it provides a very good local approximation 
    # if the points are chosen close to the root.
    
    # Choose 6 points around x0.
    # To balance accuracy and conditioning, we use a symmetric grid.
    # Let h be the step size. Points: x0 - 2.5h, x0 - 1.5h, x0 - 0.5h, 
    # x0 + 0.5h, x0 + 1.5h, x0 + 2.5h
    # We need to choose h. If h is too large, the polynomial might not be 
    # accurate. If h is too small, numerical issues in polyfit.
    # A standard choice for local approximation is h=1.0 or adaptive.
    # Let's try h=1.0 first.
    
    h = 1.0
    offsets = [-2.5, -1.5, -0.5, 0.5, 1.5, 2.5]
    points = [x0 + off * h for off in offsets]
    
    try:
        vals = []
        for p in points:
            if time.time() > deadline - 10.0:
                break
            vals.append(f(p))
        
        if len(vals) < 6:
            raise ValueError("Not enough evaluations")
        
        # Fit quintic polynomial
        # np.polyfit returns coefficients in descending order [a5, a4, ..., a0]
        coeffs = np.polyfit(points, vals, 5)
        
        # Find roots of the quintic
        roots = np.roots(coeffs)
        
        # Filter for real roots
        real_roots = []
        for r in roots:
            if abs(r.imag) < 1e-8:
                real_roots.append(r.real)
        
        if not real_roots:
            raise ValueError("No real roots")
        
        # Find the root nearest to x0
        best_root = min(real_roots, key=lambda r: abs(r - x0))
        
        # Verify the root
        fx = f(best_root)
        if abs(fx) <= 1e-8:
            return best_root
        else:
            # If the quintic approximation doesn't give a good enough root,
            # we can try one step of Newton's method using the derivative 
            # of the fitted polynomial, or just return the best guess.
            # Given the strict call limit and the fact that these are "smooth 
            # polynomial equations", the quintic fit is likely very accurate.
            # However, to be safe, we can do a quick Newton step using 
            # finite difference on f, but that costs 2 more calls.
            # Total calls so far: 6 (fit) + 1 (verify) = 7.
            # We have plenty of budget (limit 500).
            
            # Let's do a few Newton steps from best_root to polish.
            x = best_root
            for _ in range(10):
                if time.time() > deadline - 5.0:
                    break
                fx = f(x)
                if abs(fx) <= 1e-8:
                    return x
                # Derivative via central difference
                h_step = 1e-5 * max(1.0, abs(x))
                d = (f(x + h_step) - f(x - h_step)) / (2 * h_step)
                if d == 0:
                    break
                x_new = x - fx / d
                # Damping
                if abs(x_new - x) > 10 * max(1.0, abs(x)):
                    x_new = x + 0.5 * (x_new - x)
                x = x_new
            return x
            
    except Exception:
        # Fallback: If quintic fit fails (e.g., complex roots only, or numerical issue),
        # try a simpler cubic fit or just return x0 (though x0 is likely not a root).
        # Let's try a cubic fit as a fallback.
        try:
            h = 1.0
            offsets = [-1.5, -0.5, 0.5, 1.5]
            points = [x0 + off * h for off in offsets]
            vals = []
            for p in points:
                if time.time() > deadline - 10.0:
                    break
                vals.append(f(p))
            
            if len(vals) < 4:
                raise ValueError("Not enough evaluations")
            
            coeffs = np.polyfit(points, vals, 3)
            roots = np.roots(coeffs)
            real_roots = [r.real for r in roots if abs(r.imag) < 1e-8]
            
            if real_roots:
                best_root = min(real_roots, key=lambda r: abs(r - x0))
                fx = f(best_root)
                if abs(fx) <= 1e-8:
                    return best_root
                # Polish with Newton
                x = best_root
                for _ in range(10):
                    if time.time() > deadline - 5.0:
                        break
                    fx = f(x)
                    if abs(fx) <= 1e-8:
                        return x
                    h_step = 1e-5 * max(1.0, abs(x))
                    d = (f(x + h_step) - f(x - h_step)) / (2 * h_step)
                    if d == 0:
                        break
                    x_new = x - fx / d
                    if abs(x_new - x) > 10 * max(1.0, abs(x)):
                        x_new = x + 0.5 * (x_new - x)
                    x = x_new
                return x
        except Exception:
            pass
        
        # Last resort: return x0 (will likely fail, but prevents crash)
        return x0
# EVOLVE-BLOCK-END
