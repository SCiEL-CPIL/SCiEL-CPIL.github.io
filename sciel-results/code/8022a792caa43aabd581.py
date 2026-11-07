"""
Scalar root finder on 3 fixed oscillatory equations.

Strategy:
  1. Evaluate f at x0. If |f| <= 1e-8, return immediately.
  2. Systematic bracket search: try small steps first (root may be near x0),
     then expand geometrically. For oscillatory functions, try multiple
     offset patterns to catch sign changes.
  3. Use scipy.optimize.brentq if bracket found.
  4. Fallback: secant/Newton iteration from best point, then bounded
     minimization of |f(x)|.
  5. Track best (smallest |f|) point throughout and return it if deadline
     is approached.

All f calls are counted by the evaluator. The final check is not counted.
"""
import time
import math
import numpy as np
from scipy.optimize import brentq, minimize_scalar

# EVOLVE-BLOCK-START
_START_TIME = None
_DEADLINE = 142.0  # 142s out of 150s wall-clock limit (8s safety margin)

def _elapsed():
    global _START_TIME
    if _START_TIME is None:
        _START_TIME = time.time()
    return time.time() - _START_TIME

def _is_finite(x):
    return math.isfinite(x)

def _update_best(best_x, best_f, x, fval):
    """Return updated (best_x, best_f) if x has smaller |f|."""
    if _is_finite(fval) and abs(fval) < abs(best_f):
        return x, fval
    return best_x, best_f

def find_root(f, x0):
    """
    Find a root x of f such that |f(x)| <= 1e-8.
    """
    x0 = float(x0)
    if not _is_finite(x0):
        x0 = 0.0

    best_x = x0
    best_f = float('inf')

    # Phase 0: Check x0
    try:
        f0 = f(x0)
        if _is_finite(f0):
            best_x, best_f = _update_best(best_x, best_f, x0, f0)
            if abs(f0) <= 1e-8:
                return x0
    except:
        pass

    # Phase 1: Find a bracket [a, b] such that f(a)*f(b) < 0
    # Strategy: try a sequence of offsets from x0.
    # For oscillatory functions, the root could be at various distances.
    # We try: small steps, medium steps, large steps.
    
    # First, try to find a finite f value near x0 if x0 is bad
    a = x0
    fa = best_f
    if not _is_finite(fa) or abs(fa) > 1e100:
        # Try nearby points to get a finite starting value
        offsets = [0.1, -0.1, 0.5, -0.5, 1.0, -1.0, 2.0, -2.0, 5.0, -5.0]
        found = False
        for dx in offsets:
            if _elapsed() > _DEADLINE - 3:
                break
            try:
                x_try = x0 + dx
                f_try = f(x_try)
                if _is_finite(f_try):
                    a = x_try
                    fa = f_try
                    best_x, best_f = _update_best(best_x, best_f, x_try, f_try)
                    if abs(f_try) <= 1e-8:
                        return x_try
                    found = True
                    break
            except:
                continue
        if not found:
            return best_x

    # Now search for a bracket
    # Try multiple step sizes: start small, then expand
    # For oscillatory functions with unknown period, try:
    # 0.01, 0.1, 0.5, 1, 2, 5, 10, 20, 50, 100, ...
    
    b = None
    fb = None
    
    # List of step sizes to try
    steps = [0.01, 0.05, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0, 200.0, 500.0, 1000.0]
    
    for step in steps:
        if _elapsed() > _DEADLINE - 3:
            break
        
        # Check right
        try:
            b_try = a + step
            fb_try = f(b_try)
            if _is_finite(fb_try):
                best_x, best_f = _update_best(best_x, best_f, b_try, fb_try)
                if abs(fb_try) <= 1e-8:
                    return b_try
                if fa * fb_try < 0:
                    b = b_try
                    fb = fb_try
                    break
        except:
            pass
        
        # Check left
        try:
            b_try = a - step
            fb_try = f(b_try)
            if _is_finite(fb_try):
                best_x, best_f = _update_best(best_x, best_f, b_try, fb_try)
                if abs(fb_try) <= 1e-8:
                    return b_try
                if fa * fb_try < 0:
                    b = b_try
                    fb = fb_try
                    break
        except:
            pass
    
    # Also try expanding from x0 directly (in case a is not the best base)
    if b is None:
        steps2 = [0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0, 200.0, 500.0, 1000.0, 2000.0, 5000.0]
        for step in steps2:
            if _elapsed() > _DEADLINE - 3:
                break
            
            # Check right from x0
            try:
                b_try = x0 + step
                fb_try = f(b_try)
                if _is_finite(fb_try):
                    best_x, best_f = _update_best(best_x, best_f, b_try, fb_try)
                    if abs(fb_try) <= 1e-8:
                        return b_try
                    f_x0 = best_f if a == x0 else None
                    if f_x0 is not None and f_x0 * fb_try < 0:
                        b = b_try
                        fb = fb_try
                        a = x0
                        fa = f_x0
                        break
            except:
                pass
            
            # Check left from x0
            try:
                b_try = x0 - step
                fb_try = f(b_try)
                if _is_finite(fb_try):
                    best_x, best_f = _update_best(best_x, best_f, b_try, fb_try)
                    if abs(fb_try) <= 1e-8:
                        return b_try
                    f_x0 = best_f if a == x0 else None
                    if f_x0 is not None and f_x0 * fb_try < 0:
                        b = b_try
                        fb = fb_try
                        a = x0
                        fa = f_x0
                        break
            except:
                pass

    # Phase 2: If bracket found, use brentq
    if b is not None and fb is not None and fa is not None and fa * fb < 0:
        try:
            root = brentq(f, a, b, xtol=1e-12, rtol=1e-12, maxiter=100)
            if _is_finite(root):
                return root
        except Exception:
            pass

    # Phase 3: Fallback - try secant iteration from best point
    # Secant method doesn't require a bracket
    if _is_finite(best_f) and abs(best_f) < 1e100:
        x_prev = best_x
        f_prev = best_f
        # Need two points for secant
        dx = 0.1
        x_curr = x_prev + dx
        try:
            f_curr = f(x_curr)
            if _is_finite(f_curr):
                best_x, best_f = _update_best(best_x, best_f, x_curr, f_curr)
                for _ in range(50):
                    if _elapsed() > _DEADLINE - 2:
                        break
                    if abs(f_curr) <= 1e-8:
                        return x_curr
                    denom = f_curr - f_prev
                    if abs(denom) < 1e-15:
                        break
                    x_new = x_curr - f_curr * (x_curr - x_prev) / denom
                    if not _is_finite(x_new):
                        break
                    try:
                        f_new = f(x_new)
                    except:
                        break
                    if not _is_finite(f_new):
                        break
                    best_x, best_f = _update_best(best_x, best_f, x_new, f_new)
                    if abs(f_new) <= 1e-8:
                        return x_new
                    x_prev, f_prev = x_curr, f_curr
                    x_curr, f_curr = x_new, f_new
        except:
            pass

    # Phase 4: Bounded minimization of |f(x)|
    def abs_f(x):
        try:
            val = f(x)
            if not _is_finite(val):
                return 1e100
            return abs(val)
        except:
            return 1e100

    best_candidate_x = best_x
    best_candidate_val = abs(best_f) if _is_finite(best_f) else float('inf')

    # Try minimization from best_x and a few nearby points
    candidates = [best_x, x0, x0 + 1.0, x0 - 1.0, x0 + 0.5, x0 - 0.5]
    
    for start in candidates:
        if _elapsed() > _DEADLINE - 2:
            break
        
        for width in [1.0, 10.0, 100.0, 1000.0]:
            if _elapsed() > _DEADLINE - 2:
                break
            
            lower = start - width
            upper = start + width
            
            try:
                res = minimize_scalar(abs_f, bounds=(lower, upper), method='bounded', options={'xatol': 1e-12})
                if res.success and _is_finite(res.x):
                    if res.fun < best_candidate_val:
                        best_candidate_val = res.fun
                        best_candidate_x = res.x
                        if best_candidate_val <= 1e-8:
                            return best_candidate_x
            except Exception:
                pass

    # Return the best point found
    if best_candidate_val <= 1e-8:
        return best_candidate_x
    if best_candidate_val < abs(best_f) if _is_finite(best_f) else True:
        return best_candidate_x
    return best_x

# EVOLVE-BLOCK-END
