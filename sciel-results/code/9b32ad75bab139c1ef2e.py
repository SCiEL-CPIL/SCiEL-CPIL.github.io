import math
import time
import numpy as np
import scipy.optimize

# EVOLVE-BLOCK-START
def find_root(f, x0):
    """
    Find a root of f starting near x0, returning x with |f(x)| <= 1e-8.
    Uses a multi-strategy approach:
    1. Check if x0 is already a root.
    2. Try polynomial reconstruction (low-degree polynomials) using finite differences.
    3. Try to find a bracket [a, b] where f(a) and f(b) have opposite signs.
       If found, use scipy.optimize.brentq which is robust and efficient.
    4. If no bracket is found, use a safeguarded Newton's method with finite difference derivative.
    Includes a deadline guard to prevent timeout.
    """
    TOL = 1e-8
    DEADLINE = 140.0  # Safety margin before 150s kill
    START_TIME = time.time()
    
    def is_finite(x):
        return math.isfinite(x)

    def check_deadline():
        return (time.time() - START_TIME) < DEADLINE

    # Initial check
    try:
        fx0 = f(x0)
    except Exception:
        return x0 # Should not happen for valid f, but safe
        
    if not is_finite(fx0):
        return x0

    if abs(fx0) <= TOL:
        return x0

    # Strategy 1: Polynomial reconstruction
    # Try to detect if f is a low-degree polynomial by checking finite differences
    try:
        # Evaluate at several points to attempt polynomial reconstruction
        # Use points spaced around x0
        h_base = max(1.0, abs(x0)) * 0.1
        if h_base < 1e-3:
            h_base = 1.0
        
        # Try degrees 2, 3, 4 (need deg+1 points)
        for deg in [2, 3, 4]:
            if not check_deadline():
                break
                
            n_points = deg + 1
            # Use evenly spaced points
            points = [x0 + i * h_base for i in range(n_points)]
            values = []
            valid = True
            for pt in points:
                try:
                    val = f(pt)
                except Exception:
                    valid = False
                    break
                if not is_finite(val):
                    valid = False
                    break
                values.append(val)
            
            if not valid:
                break
            
            # Check if finite differences become zero (indicating polynomial of that degree)
            diffs = list(values)
            is_poly = True
            for d in range(deg):
                new_diffs = []
                for i in range(len(diffs) - 1):
                    new_diffs.append(diffs[i + 1] - diffs[i])
                diffs = new_diffs
                if len(diffs) == 0:
                    is_poly = False
                    break
                # Check if all differences are approximately zero
                max_diff = max(abs(d) for d in diffs)
                scale = max(1.0, max(abs(v) for v in values))
                if d == deg - 1:
                    # Last level should be zero for a degree-deg polynomial
                    if max_diff > 1e-6 * scale:
                        is_poly = False
                        break
                else:
                    # Earlier levels should not all be zero (unless it's a lower degree)
                    pass
            
            if is_poly:
                # Attempt to reconstruct polynomial coefficients using numpy
                try:
                    # Use np.polyfit to get coefficients
                    coeffs = np.polyfit(points, values, deg)
                    # Verify the reconstruction
                    max_error = 0.0
                    for pt, val in zip(points, values):
                        poly_val = np.polyval(coeffs, pt)
                        if not is_finite(poly_val):
                            max_error = float('inf')
                            break
                        max_error = max(max_error, abs(poly_val - val))
                    
                    if max_error < 1e-4:  # Reasonable tolerance for reconstruction
                        # Find all roots of the polynomial
                        roots = np.roots(coeffs)
                        # Filter for real roots
                        real_roots = []
                        for r in roots:
                            if abs(r.imag) < 1e-8:
                                real_roots.append(r.real)
                        
                        if real_roots:
                            # Select the best root: closest to x0 with smallest |f(x)|
                            best_root = None
                            best_score = float('inf')
                            for r in real_roots:
                                if not is_finite(r):
                                    continue
                                # Prefer roots close to x0
                                dist = abs(r - x0)
                                # Verify with actual f
                                try:
                                    fr = f(r)
                                except Exception:
                                    continue
                                if not is_finite(fr):
                                    continue
                                if abs(fr) <= TOL:
                                    # Score: prefer closer to x0
                                    score = dist
                                    if score < best_score:
                                        best_score = score
                                        best_root = r
                            
                            if best_root is not None:
                                return float(best_root)
                            # If no root passes the tolerance, try the one with smallest |f(r)|
                            best_root2 = None
                            best_fr = float('inf')
                            for r in real_roots:
                                if not is_finite(r):
                                    continue
                                try:
                                    fr = f(r)
                                except Exception:
                                    continue
                                if not is_finite(fr):
                                    continue
                                if abs(fr) < best_fr:
                                    best_fr = abs(fr)
                                    best_root2 = r
                            
                            if best_root2 is not None and best_fr <= TOL:
                                return float(best_root2)
                except Exception:
                    pass
    except Exception:
        pass

    # Strategy 2: Bracketing + Brent's Method
    # Try to find a bracket by expanding from x0
    a = x0 - 1.0
    b = x0 + 1.0
    fa = None
    fb = None
    
    bracket_found = False
    # Expand bracket
    for i in range(30): # Max expansion steps
        if not check_deadline():
            break
            
        try:
            fa = f(a)
            fb = f(b)
        except Exception:
            break
            
        if not (is_finite(fa) and is_finite(fb)):
            break
            
        if fa == 0:
            return a
        if fb == 0:
            return b
            
        if fa * fb < 0:
            bracket_found = True
            break
            
        # Expand
        a = x0 - (x0 - a) * 2
        b = x0 + (b - x0) * 2
        
        # Safety: don't expand to infinity
        if abs(a) > 1e10 or abs(b) > 1e10:
            break

    if bracket_found:
        try:
            # brentq is very robust
            root = scipy.optimize.brentq(f, a, b, xtol=1e-12, rtol=1e-12, maxiter=100)
            if is_finite(root):
                return root
        except Exception:
            pass # Fall through to Newton

    # Strategy 3: Safeguarded Newton's Method
    x = float(x0)
    fx = fx0
    
    for i in range(60):
        if not check_deadline():
            break
            
        if abs(fx) <= TOL:
            return x
            
        # Finite difference derivative
        h = 1e-6 * max(1.0, abs(x))
        try:
            fx_plus = f(x + h)
        except Exception:
            break
            
        if not is_finite(fx_plus):
            break
            
        dfx = (fx_plus - fx) / h
        
        if dfx == 0:
            # Derivative is zero, try a small step
            x_new = x + 1e-4
            fx = f(x_new)
            if is_finite(fx):
                x = x_new
            else:
                break
            continue
            
        x_new = x - fx / dfx
        
        # Safeguard: if step is too large or non-finite, dampen
        if not is_finite(x_new) or abs(x_new - x) > 1e6 * max(1.0, abs(x)):
            x_new = x - 0.5 * (fx / dfx)
            
        if not is_finite(x_new):
            break
            
        # Evaluate new point
        try:
            fx_new = f(x_new)
        except Exception:
            break
            
        if not is_finite(fx_new):
            # Dampen further
            x_new = x - 0.1 * (fx / dfx)
            if not is_finite(x_new):
                break
            try:
                fx_new = f(x_new)
            except Exception:
                break
            if not is_finite(fx_new):
                break
                
        x = x_new
        fx = fx_new
        
    return x
# EVOLVE-BLOCK-END
