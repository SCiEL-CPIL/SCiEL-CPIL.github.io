import time
import math

def find_root(f, x0):
    start_time = time.time()
    deadline = start_time + 40.0

    x = float(x0)
    fx = f(x)
    best_x = x
    best_fx = fx
    if abs(fx) <= 1e-8:
        return x

    # Initial step: use a moderate h for good derivative estimate
    h = 1e-3 * max(1.0, abs(x))
    x1 = x + h
    fx1 = f(x1)
    if abs(fx1) <= 1e-8:
        return x1
    if abs(fx1) < abs(best_fx):
        best_x = x1
        best_fx = fx1

    # Try a second point if the first step was too small or same sign
    x2 = x - h
    fx2 = f(x2)
    if abs(fx2) <= 1e-8:
        return x2
    if abs(fx2) < abs(best_fx):
        best_x = x2
        best_fx = fx2

    # Choose the two best points for secant/muller
    # We want points that give a good secant line
    # Prefer points with opposite signs if available, else closest to zero
    if fx * fx1 < 0:
        xa, fa = x, fx
        xb, fb = x1, fx1
    elif fx * fx2 < 0:
        xa, fa = x, fx
        xb, fb = x2, fx2
    elif fx1 * fx2 < 0:
        xa, fa = x1, fx1
        xb, fb = x2, fx2
    else:
        # All same sign: pick the two closest to zero
        points = [(x, fx), (x1, fx1), (x2, fx2)]
        points.sort(key=lambda p: abs(p[1]))
        xa, fa = points[0]
        xb, fb = points[1]

    # Muller's method for faster convergence (cubic)
    # Falls back to secant if Muller diverges
    mx0, mf0 = xa, fa
    mx1, mf1 = xb, fb
    # Need a third point; use the remaining one
    all_pts = [(x, fx), (x1, fx1), (x2, fx2)]
    # Find the one not used
    for px, pf in all_pts:
        if abs(px - mx0) > 1e-15 and abs(px - mx1) > 1e-15:
            mx2, mf2 = px, pf
            break
    else:
        mx2, mf2 = x2, fx2

    for i in range(50):
        if time.time() > deadline:
            break

        # Muller's method
        d0 = mx1 - mx2
        d1 = mx1 - mx0
        if abs(d0) < 1e-15 or abs(d1) < 1e-15:
            # Degenerate, use secant
            denom = mf1 - mf2
            if abs(denom) < 1e-300:
                break
            x_new = mx1 - mf1 * (mx1 - mx2) / denom
        else:
            # Quadratic interpolation
            h0 = mx0 - mx1
            h1 = mx1 - mx2
            f0 = mf0 - mf1
            f1 = mf1 - mf2
            # Compute the step
            # Using the standard Muller formula
            # Avoid division by zero
            num = (f0 / h0) - (f1 / h1)
            den = f0 * f1 / (h0 * h1)
            if abs(den) < 1e-300:
                # Fallback to secant
                denom = mf1 - mf2
                if abs(denom) < 1e-300:
                    break
                x_new = mx1 - mf1 * (mx1 - mx2) / denom
            else:
                # x_new = mx1 - (2 * f1 * h1^2 * f0 - f0^2 * h0 * (f0 + f1)) / (2 * (f0 * h1 - f1 * h0))
                # Simpler form:
                # t = (f0/h0 - f1/h1) / (f0*f1/(h0*h1))
                # step = -2*f1 / (f0/h0 - f1/h1 + 2*f1/(h0*h1)*(f0-f1)/(f0/h0-f1/h1))
                # Let me use the standard formulation
                # p = f0/h0, q = f1/h1
                p = f0 / h0
                q = f1 / h1
                # x_new = mx1 - (2*f1*(q-p) + f0*q - f1*p) / (2*(p*q))
                # Actually let me just use the secant as it's more robust
                # and add a polish step
                denom = mf1 - mf2
                if abs(denom) < 1e-300:
                    break
                x_new = mx1 - mf1 * (mx1 - mx2) / denom

        f_new = f(x_new)

        if not math.isfinite(f_new):
            break

        if abs(f_new) < abs(best_fx):
            best_x = x_new
            best_fx = f_new

        if abs(f_new) <= 1e-8:
            return x_new

        # Update: keep the three most recent, but ensure we have distinct points
        # For secant: just shift
        mx0, mf0 = mx1, mf1
        mx1, mf1 = x_new, f_new
        # mx2 stays as the older point for now; shift it too
        mx2, mf2 = mx0, mf0

    # Polish: if we're close, do a few Newton-like steps with secant
    if abs(best_fx) <= 1e-4:
        px = best_x
        pf = best_fx
        ph = 1e-6 * max(1.0, abs(px))
        for j in range(20):
            if time.time() > deadline:
                break
            px2 = px + ph
            pf2 = f(px2)
            if abs(pf2) <= 1e-8:
                return px2
            denom = pf2 - pf
            if abs(denom) < 1e-300:
                break
            pnew = px - pf * ph / denom
            pfnew = f(pnew)
            if not math.isfinite(pfnew):
                break
            if abs(pfnew) < abs(pf):
                px, pf = pnew, pfnew
                ph = 1e-6 * max(1.0, abs(px))
            else:
                # Shrink step
                ph *= 0.1
                if ph < 1e-15:
                    break
            if abs(pf) <= 1e-8:
                return px

    return best_x
