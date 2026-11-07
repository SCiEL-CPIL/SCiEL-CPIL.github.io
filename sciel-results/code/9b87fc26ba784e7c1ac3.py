import math
import time
import numpy as np
import scipy.optimize

# EVOLVE-BLOCK-START
def find_root(f, x0):
    """
    Find a root of f starting near x0, returning x with |f(x)| <= 1e-8.
    Strategy:
    1. Check x0.
    2. Quadratic interpolation (3 pts) -> verify.
    3. Cubic interpolation (4 pts) -> verify.
    4. Newton polish if close.
    5. Brent's method fallback.
    """
    TOL = 1e-8
    TOL_POLISH = 1e-12
    DEADLINE = 115.0
    START_TIME = time.time()
    
    def is_finite(x):
        return math.isfinite(x)

    def check_deadline():
        return (time.time() - START_TIME) < DEADLINE

    def polish_root(x, fx_val, max_iter=10):
        """Final high-precision polish using Newton's method with tight tolerance."""
        if not is_finite(x):
            return x, fx_val
        x_cur = x
        f_cur = fx_val
        for _ in range(max_iter):
            if not check_deadline():
                break
            if abs(f_cur) <= TOL_POLISH:
                return x_cur, f_cur
            h = 1e-8 * max(1.0, abs(x_cur))
            try:
                f_plus = f(x_cur + h)
                f_minus = f(x_cur - h)
            except Exception:
                break
            if not (is_finite(f_plus) and is_finite(f_minus)):
                break
            df = (f_plus - f_minus) / (2 * h)
            if abs(df) < 1e-15:
                break
            step = f_cur / df
            x_new = x_cur - step
            if not is_finite(x_new):
                break
            if abs(step) > 100 * max(1.0, abs(x_cur)):
                step = 0.5 * step
                x_new = x_cur - step
            try:
                f_new = f(x_new)
            except Exception:
                break
            if not is_finite(f_new):
                break
            x_cur = x_new
            f_cur = f_new
        return x_cur, f_cur

    # 1. Initial check
    try:
        fx0 = f(x0)
    except Exception:
        return x0
        
    if not is_finite(fx0):
        return x0

    if abs(fx0) <= TOL:
        return x0

    best_x = x0
    best_f_abs = abs(fx0)

    # 2. Quadratic Fit
    # Points: x0 - 1, x0, x0 + 1
    h = 1.0
    try:
        x1 = x0 - h
        x2 = x0
        x3 = x0 + h
        
        f1 = f(x1)
        # f2 = fx0
        f3 = f(x3)
        
        if all(math.isfinite(v) for v in [f1, fx0, f3]):
            # Solve for quadratic coefficients a, b, c
            # p(x) = ax^2 + bx + c
            # Using Lagrange interpolation or simple linear system
            # x1^2*a + x1*b + c = f1
            # x2^2*a + x2*b + c = fx0
            # x3^2*a + x3*b + c = f3
            
            # To avoid numpy overhead for 3x3, we can use explicit formulas
            # Or just use numpy, it's fast enough
            V = np.array([
                [x1**2, x1, 1],
                [x2**2, x2, 1],
                [x3**2, x3, 1]
            ])
            Y = np.array([f1, fx0, f3])
            
            try:
                coeffs = np.linalg.solve(V, Y)
                a, b, c = coeffs
                # ax^2 + bx + c = 0
                if abs(a) > 1e-12:
                    disc = b*b - 4*a*c
                    if disc >= 0:
                        sqrt_disc = math.sqrt(disc)
                        r1 = (-b + sqrt_disc) / (2*a)
                        r2 = (-b - sqrt_disc) / (2*a)
                        candidates = [r1, r2]
                    else:
                        candidates = []
                else:
                    if abs(b) > 1e-12:
                        candidates = [-c/b]
                    else:
                        candidates = []
                
                if candidates:
                    # Select root closest to x0
                    best_r = min(candidates, key=lambda r: abs(r - x0))
                    f_best = f(best_r)
                    if is_finite(f_best):
                        if abs(f_best) < best_f_abs:
                            best_f_abs = abs(f_best)
                            best_x = best_r
                        if abs(f_best) <= TOL:
                            return best_r
            except np.linalg.LinAlgError:
                pass
    except Exception:
        pass

    # 3. Cubic Fit
    # Add points: x0 - 1.5, x0 + 1.5
    # We already have x0-1, x0, x0+1.
    try:
        x4 = x0 - 1.5 * h
        x5 = x0 + 1.5 * h
        
        f4 = f(x4)
        f5 = f(x5)
        
        # We have points: x4, x1, x2, x3, x5
        # Let's use x4, x1, x3, x5 for a symmetric-ish set or just 4 distinct points
        # Using x4, x1, x3, x5 (skip x0 to keep them spread? No, x0 is valuable)
        # Let's use x1, x2, x3, x5 (x0-1, x0, x0+1, x0+1.5)
        # Or x4, x1, x2, x3
        # Actually, any 4 distinct points work.
        # Let's use x1, x2, x3, x5
        
        pts = [x1, x2, x3, x5]
        vals = [f1, fx0, f3, f5]
        
        if all(math.isfinite(v) for v in vals):
            V = np.array([
                [p**3, p**2, p, 1] for p in pts
            ])
            Y = np.array(vals)
            
            try:
                coeffs = np.linalg.solve(V, Y)
                # coeffs are [d, c, b, a] for dx^3 + cx^2 + bx + a
                # np.roots expects [a, b, c, d]
                roots = np.roots(coeffs[::-1])
                
                real_roots = []
                for r in roots:
                    if abs(r.imag) < 1e-9:
                        real_roots.append(r.real)
                
                if real_roots:
                    best_r = min(real_roots, key=lambda r: abs(r - x0))
                    f_best = f(best_r)
                    if is_finite(f_best):
                        if abs(f_best) < best_f_abs:
                            best_f_abs = abs(f_best)
                            best_x = best_r
                        if abs(f_best) <= TOL:
                            return best_r
            except np.linalg.LinAlgError:
                pass
    except Exception:
        pass

    # 4. Polish best candidate if it's close
    if is_finite(best_x) and best_f_abs < 1e-4:
        best_x, f_best = polish_root(best_x, f(best_x) if is_finite(best_x) else 0)
        if is_finite(best_x) and abs(f_best) <= TOL:
            return best_x

    # 5. Fallback: Bracketing + Brent
    step = 1.0
    # Reuse points if possible?
    # We have values at x0, x0±1, x0±1.5
    # Check if any existing point brackets a root
    # f(x0) = fx0
    # f(x0-1) = f1
    # f(x0+1) = f3
    # f(x0-1.5) = f4
    # f(x0+1.5) = f5
    
    points = [x0, x0-1, x0+1, x0-1.5, x0+1.5]
    fvals = [fx0, f1, f3, f4, f5]
    
    # Check pairs for sign change
    for i in range(len(points)):
        for j in range(i+1, len(points)):
            if is_finite(points[i]) and is_finite(points[j]) and is_finite(fvals[i]) and is_finite(fvals[j]):
                if fvals[i] * fvals[j] < 0:
                    a, b = points[i], points[j]
                    if a > b: a, b = b, a
                    try:
                        root = scipy.optimize.brentq(f, a, b, xtol=1e-12, rtol=1e-12, maxiter=50)
                        if is_finite(root):
                            f_root = f(root)
                            if is_finite(f_root) and abs(f_root) <= TOL:
                                return root
                            # Polish
                            root, f_root = polish_root(root, f_root)
                            if is_finite(root) and abs(f_root) <= TOL:
                                return root
                            return root
                    except Exception:
                        pass

    # Expand bracket
    a = x0 - step
    b = x0 + step
    fa = f1 # x0-1
    fb = f3 # x0+1
    
    # Note: step started at 1.0, so a=x0-1, b=x0+1
    # We already checked these.
    # Next step: 2.0 -> x0-2, x0+2
    
    for i in range(10):
        if not check_deadline():
            break
        if i == 0:
            # Already have fa, fb for step 1
            pass
        else:
            a = x0 - step
            b = x0 + step
            try:
                fa = f(a)
                fb = f(b)
            except Exception:
                break
            if not (is_finite(fa) and is_finite(fb)):
                break
        
        if abs(fa) < best_f_abs:
            best_f_abs = abs(fa)
            best_x = a
        if abs(fb) < best_f_abs:
            best_f_abs = abs(fb)
            best_x = b
            
        if abs(fa) <= TOL:
            return a
        if abs(fb) <= TOL:
            return b
            
        if fa * fb < 0:
            try:
                root = scipy.optimize.brentq(f, a, b, xtol=1e-12, rtol=1e-12, maxiter=50)
                if is_finite(root):
                    f_root = f(root)
                    if is_finite(f_root) and abs(f_root) <= TOL:
                        return root
                    root, f_root = polish_root(root, f_root)
                    if is_finite(root) and abs(f_root) <= TOL:
                        return root
                    return root
            except Exception:
                break
                
        step *= 2.0
        if step > 1e6:
            break

    # Final polish
    if is_finite(best_x):
        try:
            f_best = f(best_x)
            if is_finite(f_best):
                best_x, f_best = polish_root(best_x, f_best)
                if is_finite(best_x) and abs(f_best) <= TOL:
                    return best_x
        except Exception:
            pass

    return best_x
# EVOLVE-BLOCK-END
