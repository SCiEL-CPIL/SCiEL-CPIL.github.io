"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

# EVOLVE-BLOCK-START
def find_root(f, x0):
    # Deadline guard: 120s limit, use 110s as safety margin
    start_time = time.time()
    deadline = start_time + 110.0
    
    x0 = float(x0)
    
    # Try single-shot polynomial reconstruction (cubic interpolation)
    # Evaluate f at 4 points to determine cubic coefficients
    # Use points around x0
    
    # Choose 4 points: x0, x0+1, x0+2, x0+3 (or adaptively)
    # Actually, let's use a symmetric set around x0 for better conditioning
    # Points: x0 - 1.5, x0 - 0.5, x0 + 0.5, x0 + 1.5
    
    h = 1.0
    points = [x0 - 1.5*h, x0 - 0.5*h, x0 + 0.5*h, x0 + 1.5*h]
    
    try:
        vals = []
        for p in points:
            if time.time() > deadline - 10.0:
                break
            vals.append(f(p))
        
        if len(vals) < 4:
            raise ValueError("Not enough evaluations")
        
        # Fit cubic polynomial through these 4 points
        # Use numpy polyfit
        coeffs = np.polyfit(points, vals, 3)
        
        # Find roots of the cubic
        roots = np.roots(coeffs)
        
        # Filter for real roots
        real_roots = []
        for r in roots:
            if abs(r.imag) < 1e-10:
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
            # If the cubic approximation doesn't give a good root,
            # try Newton's method from x0 as fallback
            x = x0
            for _ in range(20):
                if time.time() > deadline - 5.0:
                    break
                fx = f(x)
                if abs(fx) <= 1e-8:
                    return x
                h_new = 1e-6 * max(1.0, abs(x))
                d = (f(x + h_new) - fx) / h_new
                if d == 0:
                    break
                x_new = x - fx / d
                # Damping to avoid divergence
                if abs(x_new - x) > 10 * max(1.0, abs(x)):
                    x_new = x + 0.5 * (x_new - x)
                x = x_new
            return x
            
    except Exception:
        # Fallback to simple Newton's method
        x = x0
        for _ in range(30):
            if time.time() > deadline - 5.0:
                break
            fx = f(x)
            if abs(fx) <= 1e-8:
                return x
            h = 1e-6 * max(1.0, abs(x))
            d = (f(x + h) - fx) / h
            if d == 0:
                break
            x = x - fx / d
        return x
# EVOLVE-BLOCK-END
