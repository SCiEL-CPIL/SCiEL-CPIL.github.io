import math
import time

_START = time.time()
_DEADLINE = 120.0

def _feval(f, x):
    return f(x)

def _cbrt(v):
    if v >= 0:
        return v ** (1.0 / 3.0)
    return -((-v) ** (1.0 / 3.0))

def _cubic_root(a, b, c, d):
    # smallest-magnitude real root of a x^3 + b x^2 + c x + d = 0
    if abs(a) < 1e-300:
        if abs(b) < 1e-300:
            if abs(c) < 1e-300:
                return 0.0
            return -d / c
        return -c / b
    a, b, c, d = float(a), float(b), float(c), float(d)
    p = (3.0 * a * c - b * b) / (3.0 * a * a)
    q = (2.0 * b * b * b - 9.0 * a * b * c + 27.0 * a * a * d) / (27.0 * a * a * a)
    disc = (q * q) / 4.0 + (p * p * p) / 27.0
    if disc > 0.0:
        u = -q / 2.0 + math.sqrt(disc)
        v = -q / 2.0 - math.sqrt(disc)
        r1 = _cbrt(u) + _cbrt(v) - b / (3.0 * a)
        r2 = -(r1 + b / a) / 3.0
        r3 = -(r1 * r1 + r1 * b / a + c / a) / 3.0
        cand = (r1, r2, r3)
    else:
        r = math.sqrt(-(p * p * p) / 27.0)
        if r == 0.0:
            m = -q / (2.0 * r) if r else 0.0
            return -b / (3.0 * a)
        if m > 1.0:
            m = 1.0
        elif m < -1.0:
            m = -1.0
        th = math.acos(m)
        rr = 2.0 * math.sqrt(-p / 3.0)
        off = -b / (3.0 * a)
        cand = (rr * math.cos(th / 3.0) + off,
                rr * math.cos((th + 2.0 * math.pi) / 3.0) + off,
                rr * math.cos((th + 4.0 * math.pi) / 3.0) + off)
    best = None
    for r in cand:
        if math.isfinite(r):
            if best is None or abs(r) < abs(best):
                best = r
    if best is None:
        return 0.0
    return best

def _cubic_interp(x0, y0, x1, y1, x2, y2, x3, y3):
    # smallest-magnitude root of the cubic through the 4 points
    h01 = x1 - x0; h02 = x2 - x0; h03 = x3 - x0
    h12 = x2 - x1; h13 = x3 - x1; h23 = x3 - x2
    a = (y0 / (h01 * h02 * h03) + y1 / (h12 * h13 * (-h01))
         + y2 / ((-h02) * (-h12) * h23) + y3 / ((-h03) * (-h13) * (-h23)))
    b = (-y0 * (x1 + x2 + x3) / (h01 * h02 * h03)
         - y1 * (x0 + x2 + x3) / (h12 * h13 * (-h01))
         - y2 * (x0 + x1 + x3) / ((-h02) * (-h12) * h23)
         - y3 * (x0 + x1 + x2) / ((-h03) * (-h13) * (-h23)))
    c = (y0 * (x1 * x2 + x1 * x3 + x2 * x3) / (h01 * h02 * h03)
         + y1 * (x0 * x2 + x0 * x3 + x2 * x3) / (h12 * h13 * (-h01))
         + y2 * (x0 * x1 + x0 * x3 + x1 * x3) / ((-h02) * (-h12) * h23)
         + y3 * (x0 * x1 + x0 * x2 + x1 * x2) / ((-h03) * (-h13) * (-h23)))
    d = -y0
    return _cubic_root(a, b, c, d)

def _quad_interp(x0, y0, x1, y1, x2, y2):
    # smallest-magnitude root of the parabola through the 3 points
    h01 = x1 - x0; h02 = x2 - x0; h12 = x2 - x1
    a = y0 / (h01 * h02) + y1 / (h12 * (-h01)) + y2 / ((-h02) * (-h12))
    b = (-y0 * (x1 + x2) / (h01 * h02) - y1 * (x0 + x2) / (h12 * (-h01))
         - y2 * (x0 + x1) / ((-h02) * (-h12)))
    c = y0
    return _cubic_root(a, b, c, d)

def _brent_step(a, b, fa, fb, c, fc, d, fd):
    # choose a safe next point inside [a, b] (a, b bracket a root)
    x = d
    if (c - d) * (c - 2.0 * d - a) <= 0.0:
        # secant between d and c
        if fd != fc:
            x = d - fd * (d - c) / (fd - fc)
    elif (b - d) * (b + a - 2.0 * d) <= 0.0:
        # secant between d and b
        if fd != fb:
            x = d - fd * (d - b) / (fd - fb)
    elif (b - c) * (a - c) < 0.0:
        # inverse quadratic through b, c, d
        if fd == fb:
            x = c - 0.5 * (b - c)
        else:
            x = _cubic_interp(b, fb, c, fc, d, fd, d, fd)
    elif (b - c) * (a - d) < 0.0:
        # inverse cubic through a, b, c, d
        x = _cubic_interp(a, fa, b, fb, c, fc, d, fd)
    else:
        x = 0.5 * (a + b)
    # safeguard: keep inside [a, b] and away from the endpoints
    if x <= min(a, b) or x >= max(a, b):
        x = 0.5 * (a + b)
    else:
        lo = min(a, b); hi = max(a, b); m = 0.5 * (lo + hi)
        if abs(x - m) < 0.25 * abs(hi - lo):
            x = m
        elif abs(x - d) < 0.25 * abs(d - c):
            x = 0.5 * (d + c)
    return x

def find_root(f, x0):
    x0 = float(x0)
    best_x = x0
    best_abs = math.inf

    def upd(x, fx):
        nonlocal best_x, best_abs
        if math.isfinite(x) and math.isfinite(fx) and abs(fx) < best_abs:
            best_x = x
            best_abs = abs(fx)

    # initial bracketing
    a = b = x0
    fa = _feval(f, a)
    upd(a, fa)
    if abs(fa) <= 1e-8:
        return a
    if not math.isfinite(fa):
        return a
    s = 1.0
    d = 1.0
    for _ in range(200):
        b = a + s * d
        fb = _feval(f, b)
        upd(b, fb)
        if not math.isfinite(fb):
            break
        if abs(fb) <= 1e-8:
            return b
        if fa * fb < 0.0:
            break
        s = 2.0 * s
        if not math.isfinite(b) or abs(b) > 1e300:
            break
    else:
        return best_x
    if not math.isfinite(fb) or fa * fb >= 0.0:
        return best_x

    # main loop
    c, fc = a, fa
    d, fd = a, fa
    for _ in range(60):
        if time.time() > _DEADLINE - 2.0:
            break
        if abs(fc) < abs(fd):
            a, b, c, d = b, a, d, c
            fa, fb, fc, fd = fb, fa, fd, fc
        if abs(fc) <= 1e-8:
            return c
        x, xp = c, d
        if fa * fb < 0.0:
            a, b, fa, fb = b, a, fb, fa
        if abs(b - a) < 1e-14 * max(1.0, abs(b)) and abs(fb) <= 1e-8:
            return b
        x = _brent_step(a, b, fa, fb, c, fc, d, fd)
        fx = _feval(f, x)
        upd(x, fx)
        if not math.isfinite(fx):
            return best_x
        if abs(fx) <= 1e-8:
            return x
        if fa * fx < 0.0:
            b, fb = x, fx
        else:
            a, fa = x, fx
        if (abs(x - xp) <= 1e-14 * max(1.0, abs(x))) or (abs(fx) > 1e-8 and abs(fx) >= abs(fc)):
            d, fd = c, fc
            c, fc = x, fx
        else:
            d, fd = c, fc
            c, fc = x, fx
    return best_x
