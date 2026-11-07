import math
import time

def find_root(f, x0):
    start_time = time.time()
    TIME_LIMIT = 118.0  # Safety margin for 120s limit
    max_calls = 490
    
    calls = 0
    best_x = float(x0)
    best_abs_f = float('inf')
    
    def _f(x):
        nonlocal calls, best_x, best_abs_f
        if calls >= max_calls:
            return float('nan')
        if time.time() - start_time > TIME_LIMIT:
            return float('nan')
        calls += 1
        try:
            val = f(x)
        except:
            return float('nan')
        if math.isfinite(val):
            abs_val = abs(val)
            if abs_val < best_abs_f:
                best_abs_f = abs_val
                best_x = x
        return val

    def time_check():
        return time.time() - start_time < TIME_LIMIT - 5.0

    x = float(x0)
    fx = _f(x)
    if math.isnan(fx):
        return best_x
    if abs(fx) <= 1e-8:
        return x

    # 1. Find Bracket - try multiple strategies
    a, b, fa, fb = None, None, None, None
    found_bracket = False
    
    # Strategy: Try expanding in both directions with adaptive steps
    base_step = 1.0
    if abs(x) > 1e6: base_step = 1e6
    elif abs(x) > 1e3: base_step = 1e3
    elif abs(x) > 1: base_step = 1.0
    else: base_step = 0.1
    
    # Try positive direction first
    step = base_step * 0.1
    for i in range(1, 18):
        if not time_check() or calls >= max_calls - 50:
            break
        xp = x + step
        fp = _f(xp)
        if math.isnan(fp): break
        if abs(fp) <= 1e-8: return xp
        if fx * fp < 0:
            a, b, fa, fb = x, xp, fx, fp
            found_bracket = True
            break
        step *= 1.5
        
    if not found_bracket:
        # Try negative direction
        step = base_step * 0.1
        for i in range(1, 18):
            if not time_check() or calls >= max_calls - 50:
                break
            xn = x - step
            fn = _f(xn)
            if math.isnan(fn): break
            if abs(fn) <= 1e-8: return xn
            if fx * fn < 0:
                a, b, fa, fb = xn, x, fn, fx
                found_bracket = True
                break
            step *= 1.5
    
    # If no bracket found yet, try more aggressive expansion
    if not found_bracket:
        # Try larger steps
        for scale in [10, 50, 100, 500, 1000]:
            if not time_check() or calls >= max_calls - 50:
                break
            for sign in [1, -1]:
                if not time_check() or calls >= max_calls - 50:
                    break
                xp = x + sign * base_step * scale
                fp = _f(xp)
                if math.isnan(fp): continue
                if abs(fp) <= 1e-8: return xp
                if fx * fp < 0:
                    if sign > 0:
                        a, b, fa, fb = x, xp, fx, fp
                    else:
                        a, b, fa, fb = xp, x, fp, fx
                    found_bracket = True
                    break
            if found_bracket:
                break

    if found_bracket:
        # 2. Use Brent's method for robust convergence, then polish with Newton
        try:
            from scipy.optimize import brentq
            # Brent's method with tight tolerance
            root = brentq(_f, a, b, xtol=1e-14, rtol=1e-14, maxiter=50)
            if math.isfinite(root):
                f_root = _f(root)
                if not math.isnan(f_root) and abs(f_root) <= 1e-8:
                    return root
                # If brentq result is close but not quite, use it as starting point for Newton
                if not math.isnan(f_root) and abs(f_root) < 1e-4:
                    # Newton polish from brentq result
                    nx = root
                    for _ in range(15):
                        if not time_check() or calls >= max_calls - 2:
                            break
                        fn = _f(nx)
                        if math.isnan(fn):
                            break
                        if abs(fn) <= 1e-8:
                            return nx
                        # Finite difference derivative with adaptive h
                        h = max(1e-10 * abs(nx), 1e-14)
                        fn_plus = _f(nx + h)
                        fn_minus = _f(nx - h)
                        if math.isnan(fn_plus) and math.isnan(fn_minus):
                            break
                        if math.isnan(fn_plus):
                            df = (fn - fn_minus) / h
                        elif math.isnan(fn_minus):
                            df = (fn_plus - fn) / h
                        else:
                            df = (fn_plus - fn_minus) / (2 * h)
                        if abs(df) > 1e-30:
                            new_x = nx - fn / df
                            if math.isfinite(new_x):
                                nx = new_x
                            else:
                                break
                        else:
                            break
                    if time_check() and calls < max_calls:
                        f_final = _f(nx)
                        if not math.isnan(f_final) and abs(f_final) <= 1e-8:
                            return nx
                        if not math.isnan(f_final) and abs(f_final) < best_abs_f:
                            best_x = nx
                            best_abs_f = abs(f_final)
        except:
            pass
        
        # Fallback: manual secant/bisection if brentq failed
        result_x = a if abs(fa) < abs(fb) else b
        if time_check() and calls < max_calls - 10:
            for i in range(60):
                if not time_check() or calls >= max_calls - 5:
                    break
                if abs(fa) <= 1e-8: return a
                if abs(fb) <= 1e-8: return b
                
                # Try Secant
                if abs(fa - fb) > 1e-30:
                    m = (a * fb - b * fa) / (fb - fa)
                else:
                    m = (a + b) / 2
                    
                if a < m < b:
                    fm = _f(m)
                    if math.isnan(fm): break
                    if abs(fm) <= 1e-8: return m
                    if fa * fm < 0:
                        b, fb = m, fm
                    else:
                        a, fa = m, fm
                else:
                    # Bisection
                    m = (a + b) / 2
                    fm = _f(m)
                    if math.isnan(fm): break
                    if abs(fm) <= 1e-8: return m
                    if fa * fm < 0:
                        b, fb = m, fm
                    else:
                        a, fa = m, fm
            
            # Final Newton polish
            result_x = a if abs(fa) < abs(fb) else b
            if time_check() and calls < max_calls - 5:
                nx = result_x
                for _ in range(10):
                    if not time_check() or calls >= max_calls - 2:
                        break
                    fn = _f(nx)
                    if math.isnan(fn):
                        break
                    if abs(fn) <= 1e-8:
                        return nx
                    h = max(1e-10 * abs(nx), 1e-14)
                    fn_plus = _f(nx + h)
                    fn_minus = _f(nx - h)
                    if math.isnan(fn_plus) and math.isnan(fn_minus):
                        break
                    if math.isnan(fn_plus):
                        df = (fn - fn_minus) / h
                    elif math.isnan(fn_minus):
                        df = (fn_plus - fn) / h
                    else:
                        df = (fn_plus - fn_minus) / (2 * h)
                    if abs(df) > 1e-30:
                        new_x = nx - fn / df
                        if math.isfinite(new_x):
                            nx = new_x
                        else:
                            break
                    else:
                        break
                if time_check() and calls < max_calls:
                    f_final = _f(nx)
                    if not math.isnan(f_final) and abs(f_final) <= 1e-8:
                        return nx
                    if not math.isnan(f_final) and abs(f_final) < best_abs_f:
                        best_x = nx
                        best_abs_f = abs(f_final)
        
        return best_x if best_abs_f < abs(_f(result_x)) else result_x

    # 3. If no bracket found, do local minimization of |f(x)|
    if time_check() and calls < max_calls - 30:
        # Try a few candidate points around x0
        candidates = [x]
        for scale in [0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0]:
            for sign in [1, -1]:
                if abs(x) > 1e6:
                    step = 1e6 * scale
                elif abs(x) > 1e3:
                    step = 1e3 * scale
                elif abs(x) > 1:
                    step = scale
                else:
                    step = scale * 0.1
                candidates.append(x + sign * step)
        
        # Evaluate candidates and keep the best
        for cx in candidates:
            if not time_check() or calls >= max_calls - 20:
                break
            cf = _f(cx)
            if math.isnan(cf):
                continue
            if abs(cf) <= 1e-8:
                return cx
        
        # If we have a good candidate, try to refine it with local search
        if best_abs_f < 1.0:
            # Local search using golden section on |f(x)|
            best_x_local = best_x
            h = 0.1
            if abs(x) > 1e3:
                h = 10.0
            elif abs(x) > 1:
                h = 1.0
            else:
                h = 0.01
            
            # Expand h to find a region where |f| increases on both sides
            for _ in range(10):
                if not time_check() or calls >= max_calls - 15:
                    break
                x1 = best_x_local - h
                x2 = best_x_local + h
                f1 = _f(x1)
                f2 = _f(x2)
                if math.isnan(f1) or math.isnan(f2):
                    break
                if abs(f1) > best_abs_f and abs(f2) > best_abs_f:
                    break
                h *= 2
            
            # Now do golden section search
            if time_check() and calls < max_calls - 10:
                gr = (5 ** 0.5 - 1) / 2  # Golden ratio conjugate
                a_gs = best_x_local - h
                b_gs = best_x_local + h
                c_gs = b_gs - gr * (b_gs - a_gs)
                d_gs = a_gs + gr * (b_gs - a_gs)
                fc = _f(c_gs)
                fd = _f(d_gs)
                
                for i in range(40):
                    if not time_check() or calls >= max_calls - 5:
                        break
                    if math.isnan(fc) and math.isnan(fd):
                        break
                    if math.isnan(fc):
                        a_gs = c_gs
                        c_gs = d_gs
                        fc = fd
                        d_gs = a_gs + gr * (b_gs - a_gs)
                        fd = _f(d_gs)
                        continue
                    if math.isnan(fd):
                        b_gs = d_gs
                        d_gs = c_gs
                        fd = fc
                        c_gs = b_gs - gr * (b_gs - a_gs)
                        fc = _f(c_gs)
                        continue
                    
                    if abs(fc) <= 1e-8:
                        return c_gs
                    if abs(fd) <= 1e-8:
                        return d_gs
                    
                    if abs(fc) < abs(fd):
                        b_gs = d_gs
                        d_gs = c_gs
                        fd = fc
                        c_gs = b_gs - gr * (b_gs - a_gs)
                        fc = _f(c_gs)
                    else:
                        a_gs = c_gs
                        c_gs = d_gs
                        fc = fd
                        d_gs = a_gs + gr * (b_gs - a_gs)
                        fd = _f(d_gs)
                    
                    if abs(b_gs - a_gs) < 1e-12:
                        break
                
                best_x_local = (a_gs + b_gs) / 2
                f_local = _f(best_x_local)
                if not math.isnan(f_local) and abs(f_local) <= 1e-8:
                    return best_x_local
                if not math.isnan(f_local) and abs(f_local) < best_abs_f:
                    best_x = best_x_local
                    best_abs_f = abs(f_local)
            
            # Final Newton polish on the local minimum
            if time_check() and calls < max_calls - 5:
                nx = best_x
                for _ in range(12):
                    if not time_check() or calls >= max_calls - 2:
                        break
                    fn = _f(nx)
                    if math.isnan(fn):
                        break
                    if abs(fn) <= 1e-8:
                        return nx
                    h = max(1e-10 * abs(nx), 1e-14)
                    fn_plus = _f(nx + h)
                    fn_minus = _f(nx - h)
                    if math.isnan(fn_plus) and math.isnan(fn_minus):
                        break
                    if math.isnan(fn_plus):
                        df = (fn - fn_minus) / h
                    elif math.isnan(fn_minus):
                        df = (fn_plus - fn) / h
                    else:
                        df = (fn_plus - fn_minus) / (2 * h)
                    if abs(df) > 1e-30:
                        new_x = nx - fn / df
                        if math.isfinite(new_x):
                            nx = new_x
                        else:
                            break
                    else:
                        break
                if time_check() and calls < max_calls:
                    f_final = _f(nx)
                    if not math.isnan(f_final) and abs(f_final) <= 1e-8:
                        return nx
                    if not math.isnan(f_final) and abs(f_final) < best_abs_f:
                        best_x = nx
                        best_abs_f = abs(f_final)
        
        # Final check
        final_f = _f(best_x)
        if not math.isnan(final_f) and abs(final_f) <= 1e-8:
            return best_x

    return best_x
