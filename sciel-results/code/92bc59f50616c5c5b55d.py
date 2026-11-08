"""Derivative-free Illinois bisection with adaptive bracketing for oscillatory roots."""
import time

# EVOLVE-BLOCK-START
def find_root(f, x0):
    x0 = float(x0)
    T0 = time.time()
    DEADLINE = 145.0
    TOL = 1e-12

    def elapsed_ok():
        return (time.time() - T0) < DEADLINE

    # Evaluate start
    try:
        f0 = f(x0)
    except Exception:
        f0 = float('nan')

    best_x = x0
    best_val = abs(f0) if f0 == f0 else float('inf')

    def track(x, fx):
        nonlocal best_x, best_val
        if fx == fx:
            av = abs(fx)
            if av < best_val:
                best_val = av
                best_x = x

    track(x0, f0)

    if f0 == f0 and abs(f0) <= TOL:
        return x0

    # ------------------------------------------------------------------
    # 1) Find a bracket [a, b] with a sign change via outward scanning
    # ------------------------------------------------------------------
    a = b = None
    fa = fb = None

    base = max(1.0, abs(x0))
    
    # Phase 1: Coarse outward scan with geometric step growth
    step = base * 1e-3
    lo = x0
    flo = f0
    for _ in range(50):
        if not elapsed_ok():
            break
        hi = lo + step
        try:
            fhi = f(hi)
        except Exception:
            fhi = float('nan')
        track(hi, fhi)
        if fhi == fhi and flo == flo and flo * fhi < 0:
            a, b, fa, fb = lo, hi, flo, fhi
            break
        
        # Check negative side
        lo_n = x0 - step
        try:
            flon = f(lo_n)
        except Exception:
            flon = float('nan')
        track(lo_n, flon)
        if flon == flon and f0 == f0 and f0 * flon < 0:
            a, b, fa, fb = lo_n, x0, flon, f0
            break
        
        # Grow step
        step = step * 1.8 + base * 1e-8
        lo = hi
        if fhi == fhi:
            flo = fhi
        else:
            flo = float('nan')

    # Phase 2: If no bracket, do denser local scan for oscillatory roots
    if a is None:
        for r in [base * 1e-2, base * 1e-1, base, base * 10.0]:
            if not elapsed_ok():
                break
            n = 200
            left = x0 - r
            right = x0 + r
            prev_x = left
            try:
                prev_f = f(prev_x)
            except Exception:
                prev_f = float('nan')
            track(prev_x, prev_f)
            for i in range(1, n + 1):
                if not elapsed_ok():
                    break
                x_i = left + (right - left) * i / n
                try:
                    f_i = f(x_i)
                except Exception:
                    f_i = float('nan')
                track(x_i, f_i)
                if f_i == f_i and prev_f == prev_f and prev_f * f_i < 0:
                    a, b, fa, fb = prev_x, x_i, prev_f, f_i
                    break
                prev_x, prev_f = x_i, f_i
            if a is not None:
                break

    # ------------------------------------------------------------------
    # 2) Refine with Illinois method (derivative-free, 1 call/step)
    # ------------------------------------------------------------------
    if a is not None:
        if fa is None or fb is None:
            try:
                fa = f(a)
            except Exception:
                fa = float('nan')
            try:
                fb = f(b)
            except Exception:
                fb = float('nan')
        
        # Ensure a < b for consistent bracket
        if a > b:
            a, b = b, a
            fa, fb = fb, fa
        
        # Illinois method
        for _ in range(300):
            if not elapsed_ok():
                break
            if fa == fa and abs(fa) <= TOL:
                return a
            if fb == fb and abs(fb) <= TOL:
                return b
            
            # Illinois: replace the endpoint with larger |f| by bisection,
            # but halve the weight of that side
            if abs(fa) > abs(fb):
                # Replace a with midpoint, but weight toward b
                mid = 0.5 * (a + b)
                try:
                    fm = f(mid)
                except Exception:
                    fm = float('nan')
                track(mid, fm)
                if fm == fm:
                    if fm * fb < 0:
                        a, fa = mid, fm
                    else:
                        b, fb = mid, fm
                    # Halve the weight: effectively, we just use midpoint
                    # but the Illinois trick is to set the new point to
                    # (a + b) / 2 when replacing the larger side
                    # Standard Illinois: x_new = (a+b)/2 if replacing a,
                    # but then the "weight" is handled by the fact that
                    # we always bisect the larger side
                else:
                    # NaN: just bisect
                    mid = 0.5 * (a + b)
                    try:
                        fm = f(mid)
                    except Exception:
                        break
                    track(mid, fm)
                    if fm == fm and fm * fb < 0:
                        a, fa = mid, fm
                    elif fm == fm and fm * fa < 0:
                        b, fb = mid, fm
                    else:
                        break
            else:
                # Replace b with midpoint
                mid = 0.5 * (a + b)
                try:
                    fm = f(mid)
                except Exception:
                    fm = float('nan')
                track(mid, fm)
                if fm == fm:
                    if fm * fa < 0:
                        b, fb = mid, fm
                    else:
                        a, fa = mid, fm
                else:
                    mid = 0.5 * (a + b)
                    try:
                        fm = f(mid)
                    except Exception:
                        break
                    track(mid, fm)
                    if fm == fm and fm * fa < 0:
                        b, fb = mid, fm
                    elif fm == fm and fm * fb < 0:
                        a, fa = mid, fm
                    else:
                        break
        
        if best_val <= 1e-8:
            return best_x

    # ------------------------------------------------------------------
    # 3) No bracket: return best candidate
    # ------------------------------------------------------------------
    if best_x == best_x and abs(best_val) < float('inf'):
        return best_x
    return x0
# EVOLVE-BLOCK-END
