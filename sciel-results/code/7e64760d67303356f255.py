import math
import time
from scipy.optimize import brentq

# EVOLVE-BLOCK-START
def find_root(f, x0):
    start_time = time.time()
    deadline = start_time + 105.0  # Safety margin before 120s limit
    
    x = float(x0)
    fx = f(x)
    
    # Check if x0 is already a root
    if abs(fx) <= 1e-8:
        return x

    # Strategy: Find a bracket [a, b] where f(a) and f(b) have opposite signs.
    # Then use Brent's method to converge quickly.
    
    best_x = x
    best_f = fx
    best_abs = abs(fx)
    
    a = x
    fa = fx
    
    # Try expanding in both directions
    # We will test points x0 + delta and x0 - delta
    # delta starts small and grows exponentially
    
    # Denser scan: use factor of 3 instead of 10 to catch narrow oscillations
    # Start with very fine steps, then expand
    deltas = []
    d = 1e-6
    while d <= 1e6:
        deltas.append(d)
        d *= 3.0
    
    bracket_found = False
    lo = 0.0
    hi = 0.0
    flo = 0.0
    fhi = 0.0
    
    for d in deltas:
        if time.time() > deadline:
            break
            
        # Check right side
        x_right = x + d
        try:
            fr = f(x_right)
        except:
            fr = None
        
        if fr is not None and math.isfinite(fr):
            abs_fr = abs(fr)
            if abs_fr < best_abs:
                best_abs = abs_fr
                best_x = x_right
                best_f = fr
            if abs_fr <= 1e-8:
                return x_right
            if fa * fr < 0:
                lo, hi = x, x_right
                flo, fhi = fa, fr
                bracket_found = True
                break
        
        # Check left side
        x_left = x - d
        try:
            fl = f(x_left)
        except:
            fl = None
            
        if fl is not None and math.isfinite(fl):
            abs_fl = abs(fl)
            if abs_fl < best_abs:
                best_abs = abs_fl
                best_x = x_left
                best_f = fl
            if abs_fl <= 1e-8:
                return x_left
            if fa * fl < 0:
                lo, hi = x_left, x
                flo, fhi = fl, fa
                bracket_found = True
                break
        
        if time.time() > deadline:
            break

    if bracket_found:
        # Use Brent's method for fast, robust convergence
        try:
            root = brentq(f, lo, hi, xtol=1e-10, rtol=1e-10, maxiter=100)
            if time.time() < deadline:
                # Verify
                f_root = f(root)
                if abs(f_root) <= 1e-8:
                    return root
        except Exception:
            pass
        
        # If brentq failed or wasn't precise enough, try bisection as fallback
        if time.time() < deadline - 5:
            try:
                # Re-evaluate endpoints to ensure we have valid bracket
                f_lo = f(lo)
                f_hi = f(hi)
                if f_lo * f_hi < 0:
                    # Manual bisection with tight tolerance
                    a_bis, b_bis = lo, hi
                    fa_bis, fb_bis = f_lo, f_hi
                    for _ in range(60):
                        if time.time() > deadline:
                            break
                        m = 0.5 * (a_bis + b_bis)
                        fm = f(m)
                        if abs(fm) <= 1e-8:
                            return m
                        if fa_bis * fm < 0:
                            b_bis = m
                            fb_bis = fm
                        else:
                            a_bis = m
                            fa_bis = fm
                    # Return midpoint of final bracket
                    m_final = 0.5 * (a_bis + b_bis)
                    if abs(f(m_final)) <= 1e-8:
                        return m_final
            except Exception:
                pass

    # If no bracket found, refine best_x with local search
    if best_abs > 1e-8 and time.time() < deadline - 5:
        # Try a few Newton-like steps from best_x with numerical derivative
        h = 1e-6 * max(1.0, abs(best_x))
        for _ in range(3):
            try:
                f_best = f(best_x)
                f_pert = f(best_x + h)
                if time.time() > deadline:
                    break
                d = (f_pert - f_best) / h
                if d != 0 and math.isfinite(d):
                    x_new = best_x - f_best / d
                    if math.isfinite(x_new):
                        f_new = f(x_new)
                        if math.isfinite(f_new) and abs(f_new) < best_abs:
                            best_x = x_new
                            best_f = f_new
                            best_abs = abs(f_new)
                        else:
                            break
                    else:
                        break
                else:
                    break
            except:
                break

    return best_x
# EVOLVE-BLOCK-END
