import math
import time
import numpy as np
import scipy.optimize

# EVOLVE-BLOCK-START

def find_root(f, x0):
    """
    Root finder for 3 fixed smooth polynomials.
    Strategy:
      1. Check x0.
      2. Try 4-point interpolation -> cubic reconstruction (4 calls + 1 verify).
      3. If that fails, try 5-point interpolation -> quartic reconstruction (1 more call + 1 verify).
      4. If those fail, use Brent's method with bracket expansion.
      5. Final fallback: secant / damped Newton.
    """
    TOL = 1e-8
    DEADLINE = 140.0
    t0 = time.time()

    def ok(v):
        return math.isfinite(v)

    def deadline_ok():
        return (time.time() - t0) < DEADLINE

    # ---- 1. initial check
    try:
        fx0 = f(x0)
    except Exception:
        return x0
    if not ok(fx0):
        return x0
    if abs(fx0) <= TOL:
        return x0

    best_x = x0
    best_af = abs(fx0)

    # ---- 2: 4-point interpolation (cubic)
    # Use 4 well-separated points around x0.
    # Points: x0-1.5, x0-0.5, x0+0.5, x0+1.5 (scaled)
    scale = max(1.0, abs(x0))
    # Choose 4 points symmetrically around x0
    pts4 = [x0 - 1.5*scale, x0 - 0.5*scale, x0 + 0.5*scale, x0 + 1.5*scale]
    
    try:
        vals4 = []
        for p in pts4:
            v = f(p)
            if not ok(v):
                vals4 = None
                break
            vals4.append(v)
            afv = abs(v)
            if afv < best_af:
                best_af = afv
                best_x = p
            if afv <= TOL:
                return p

        if vals4 is not None and len(pts4) == 4:
            if len(set(pts4)) == 4:
                V4 = np.vander(pts4, N=4, increasing=True)
                try:
                    coeffs4 = np.linalg.solve(V4, np.array(vals4))
                    roots4 = np.roots(coeffs4[::-1])
                    real_roots4 = [r.real for r in roots4 if abs(r.imag) < 1e-10]
                    if real_roots4:
                        cand4 = min(real_roots4, key=lambda r: abs(r - x0))
                        try:
                            fc4 = f(cand4)
                            if ok(fc4) and abs(fc4) <= TOL:
                                return cand4
                        except Exception:
                            pass
                except np.linalg.LinAlgError:
                    pass
    except Exception:
        pass

    if not deadline_ok():
        return best_x

    # ---- 3: 5-point interpolation (quartic) - add one more point
    # Add x0 - 2*scale or x0 + 2*scale
    # Let's add x0 - 2*scale to make 5 points
    pts5 = pts4 + [x0 - 2*scale]
    
    try:
        vals5 = list(vals4) if vals4 is not None else []
        if vals4 is not None:
            v5 = f(pts5[4])
            if ok(v5):
                vals5.append(v5)
                afv5 = abs(v5)
                if afv5 < best_af:
                    best_af = afv5
                    best_x = pts5[4]
                if afv5 <= TOL:
                    return pts5[4]
            else:
                vals5 = None
        else:
            # If vals4 was None, we need to re-evaluate all 5
            vals5 = []
            for p in pts5:
                v = f(p)
                if not ok(v):
                    vals5 = None
                    break
                vals5.append(v)
                afv = abs(v)
                if afv < best_af:
                    best_af = afv
                    best_x = p
                if afv <= TOL:
                    return p

        if vals5 is not None and len(pts5) == 5:
            if len(set(pts5)) == 5:
                V5 = np.vander(pts5, N=5, increasing=True)
                try:
                    coeffs5 = np.linalg.solve(V5, np.array(vals5))
                    roots5 = np.roots(coeffs5[::-1])
                    real_roots5 = [r.real for r in roots5 if abs(r.imag) < 1e-10]
                    if real_roots5:
                        cand5 = min(real_roots5, key=lambda r: abs(r - x0))
                        try:
                            fc5 = f(cand5)
                            if ok(fc5) and abs(fc5) <= TOL:
                                return cand5
                        except Exception:
                            pass
                except np.linalg.LinAlgError:
                    pass
    except Exception:
        pass

    if not deadline_ok():
        return best_x

    # ---- 4: Brent with bracket expansion
    step = scale
    a = None
    b = None
    fa = None
    fb = None

    # Use the points already evaluated to try to find a sign change
    # We have pts4 and maybe pts5
    all_pts = pts5 if (vals5 is not None and len(pts5) == 5) else pts4
    all_vals = vals5 if (vals5 is not None and len(pts5) == 5) else (vals4 if vals4 is not None else [])
    
    try:
        evaled = list(zip(all_pts, all_vals))
        for i in range(len(evaled) - 1):
            (pa, va), (pb, vb) = evaled[i], evaled[i+1]
            if va * vb < 0:
                a, b = min(pa, pb), max(pa, pb)
                fa, fb = va, vb
                break
    except Exception:
        a = b = None

    if a is None:
        for i in range(30):
            if not deadline_ok():
                break
            a = x0 - step
            b = x0 + step
            try:
                fa = f(a)
                fb = f(b)
            except Exception:
                break
            if not (ok(fa) and ok(fb)):
                break
            af_a, af_b = abs(fa), abs(fb)
            if af_a < best_af:
                best_af = af_a; best_x = a
            if af_b < best_af:
                best_af = af_b; best_x = b
            if af_a <= TOL:
                return a
            if af_b <= TOL:
                return b
            if fa * fb < 0:
                break
            step *= 2.0
            if step > 1e12:
                break
        else:
            pass

    if a is not None and fa is not None and fb is not None and fa * fb < 0:
        try:
            root = scipy.optimize.brentq(f, a, b, xtol=1e-12, rtol=1e-14, maxiter=200)
            if ok(root):
                try:
                    fr = f(root)
                    if ok(fr) and abs(fr) <= TOL:
                        return root
                except Exception:
                    pass
                if 'fr' in dir() and ok(fr) and abs(fr) < best_af:
                    best_x = root
                    best_af = abs(fr)
                return root
        except Exception:
            pass

    if not deadline_ok():
        return best_x

    # ---- 5: Secant / damped Newton fallback
    x = best_x
    try:
        fx = f(x)
    except Exception:
        return x
    if not ok(fx):
        return x

    # Use a secant-like step with a nearby point
    h = 1e-4 * max(1.0, abs(x))
    try:
        fxh = f(x + h)
    except Exception:
        fxh = fx

    if not ok(fxh):
        return x

    for i in range(60):
        if not deadline_ok():
            break
        if abs(fx) <= TOL:
            return x

        dfx = (fxh - fx) / h if h != 0 else 0.0
        if abs(dfx) < 1e-18:
            break

        step_n = fx / dfx
        x_new = x - step_n

        if not ok(x_new) or abs(step_n) > 20 * max(1.0, abs(x)):
            # damp
            found = False
            for d in [0.5, 0.25, 0.125, 0.0625, 0.03125, 0.015625]:
                x_d = x - d * step_n
                if not ok(x_d):
                    continue
                try:
                    fx_d = f(x_d)
                except Exception:
                    continue
                if not ok(fx_d):
                    continue
                if abs(fx_d) <= TOL:
                    return x_d
                if abs(fx_d) < best_af:
                    best_af = abs(fx_d); best_x = x_d
                if abs(fx_d) < abs(fx):
                    # update secant pair
                    fxh = fx
                    fx = fx_d
                    xh = x
                    x = x_d
                    h = x - xh if x != xh else 1e-4 * max(1.0, abs(x))
                    if abs(h) < 1e-15:
                        h = 1e-4 * max(1.0, abs(x))
                    found = True
                    break
            if not found:
                break
            continue

        try:
            fx_new = f(x_new)
        except Exception:
            break
        if not ok(fx_new):
            break

        if abs(fx_new) < best_af:
            best_af = abs(fx_new); best_x = x_new
        if abs(fx_new) <= TOL:
            return x_new

        # update secant
        fxh = fx
        xh = x
        fx = fx_new
        x = x_new
        h = x - xh if abs(x - xh) > 1e-15 else 1e-4 * max(1.0, abs(x))

    return best_x

# EVOLVE-BLOCK-END
