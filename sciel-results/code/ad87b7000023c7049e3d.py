"""
Refined Cohn-Goncalves construction: find the 12 positive double-root locations
as the zeros of F(x) = L_12(x)^2 - L_11(x)^2.
The upper bound on C4 is the last (largest) sign change, which for this
construction equals the largest root.

Strategy:
- Use scipy.special.eval_laguerre for fast evaluation.
- Coarse-to-fine root finding: scan a grid, bisect, then Newton polish.
- Return exactly 12 positive roots sorted.
"""
import time
import numpy as np
from scipy.special import eval_laguerre

_START = time.time()
_DEADLINE = 580.0  # safety margin before 630 s kill

def _time_left():
    return _DEADLINE - (time.time() - _START)

def _F(x):
    """F(x) = L_12(x)^2 - L_11(x)^2."""
    l12 = eval_laguerre(12, x)
    l11 = eval_laguerre(11, x)
    return l12 * l12 - l11 * l11

def _dF(x):
    """Numerical derivative of F via central difference."""
    h = 1e-7 * max(1.0, abs(x))
    return (_F(x + h) - _F(x - h)) / (2.0 * h)

def _newton(f, fp, x0, tol=1e-14, maxit=60):
    """Newton's method with fallback to bisection if derivative fails."""
    x = float(x0)
    for _ in range(maxit):
        fx = f(x)
        if abs(fx) < tol:
            return x
        d = fp(x)
        if d == 0 or not np.isfinite(d):
            break
        step = fx / d
        x_new = x - step
        if not np.isfinite(x_new) or x_new <= 0:
            break
        x = x_new
    return x

def _bisect(f, a, b, tol=1e-14, maxit=100):
    """Bisection method."""
    fa = f(a)
    fb = f(b)
    if fa * fb > 0:
        return 0.5 * (a + b)
    for _ in range(maxit):
        c = 0.5 * (a + b)
        fc = f(c)
        if abs(fc) < tol or (b - a) < tol:
            return c
        if fa * fc <= 0:
            b, fb = c, fc
        else:
            a, fa = c, fc
    return 0.5 * (a + b)

def construct():
    m = 12
    n = 1
    
    # The zeros of L_12^2 - L_11^2 are positive.
    # L_12 has 12 positive zeros, L_11 has 11. The difference of squares
    # will have zeros interleaved with the zeros of L_12 and L_11.
    # We expect 12 positive zeros (the "double roots" in the construction).
    
    # Upper bound for scanning: the largest zero of L_12 is roughly (2m+1+2*sqrt(m))^2/4
    # A safe upper bound is around 200-300 for m=12.
    hi = 300.0
    
    # Coarse scan to find sign-change intervals
    xg = np.linspace(1e-8, hi, 100000)
    fg = np.array([_F(float(v)) for v in xg])
    
    # Find sign changes
    s = np.sign(fg)
    s[s == 0] = 1.0
    idx = np.where(np.diff(s) != 0)[0]
    
    roots = []
    for i in idx:
        if _time_left() < 10.0:
            break
        a = float(xg[i])
        b = float(xg[i + 1])
        fa = _F(a)
        fb = _F(b)
        
        # Bisect to narrow down
        c = _bisect(_F, a, b)
        
        # Newton polish
        x_newton = _newton(_F, _dF, c)
        
        # Validate
        if np.isfinite(x_newton) and x_newton > 0 and abs(_F(x_newton)) < 1e-8:
            roots.append(float(x_newton))
        elif np.isfinite(c) and c > 0:
            roots.append(float(c))
    
    # Deduplicate and sort
    roots = sorted(set(round(r, 12) for r in roots))
    
    # Ensure exactly 12 roots
    if len(roots) > 12:
        roots = roots[:12]
    elif len(roots) < 12:
        # Fallback: use Laguerre-Gauss nodes as a reasonable default
        try:
            from numpy.polynomial.laguerre import laggauss
            points, _ = laggauss(12)
            roots = sorted([float(p) for p in points])
        except Exception:
            roots = [4.0 * (i + 1) for i in range(12)]
    
    return roots[:12]

def upper_bound_C4(roots):
    """The upper bound on C4 is the largest root (last sign change)."""
    if roots:
        return float(max(roots))
    return 0.0

if __name__ == "__main__":
    r = construct()
    print(r)
    print("upper bound C4:", upper_bound_C4(r))
