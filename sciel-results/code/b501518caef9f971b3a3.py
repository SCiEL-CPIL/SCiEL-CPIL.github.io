"""
Scalar root finder on 3 fixed oscillatory equations.

Strategy:
  1. Evaluate f at x0. If |f| <= 1e-8, return immediately.
  2. Coarse scan to identify region with smallest |f(x)|.
  3. Use that region as tight bracket for minimize_scalar on f(x)^2.
  4. If bracket found via sign change, use brentq for fast convergence.
  5. Final high-precision polish pass if close but not converged.

All f calls are counted by the evaluator. The final check is not counted.
"""
import time
import math
import numpy as np
from scipy.optimize import brentq, minimize_scalar

# EVOLVE-BLOCK-START
_START_TIME = None
_DEADLINE = 145.0  # 145s out of 150s wall-clock limit
_CALL_BUDGET = 480  # Keep under 500 calls

def _elapsed():
    global _START_TIME
    if _START_TIME is None:
        _START_TIME = time.time()
    return time.time() - _START_TIME

def _is_finite(x):
    return math.isfinite(x)

class _CallCounter:
    def __init__(self, f):
        self._f = f
        self.count = 0
    
    def __call__(self, x):
        self.count += 1
        return self._f(x)

def find_root(f, x0):
    """
    Find a root x of f such that |f(x)| <= 1e-8.
    Strategy:
    1. Check x0.
    2. Coarse scan to find region with smallest |f(x)|.
    3. Use tight bracket from scan for minimize_scalar.
    4. If sign change found, use brentq.
    5. Final polish if close.
    """
    x0 = float(x0)
    if not _is_finite(x0):
        x0 = 0.0

    fc = _CallCounter(f)
    best_x = x0
    best_f = float('inf')

    try:
        f0 = fc(x0)
        if _is_finite(f0):
            best_f = f0
            if abs(f0) <= 1e-8:
                return x0
    except:
        pass

    # Phase 1: Coarse scan to find region with smallest |f(x)|
    # Try multiple scales and directions
    scan_points = []
    a = x0
    fa = best_f
    if not _is_finite(fa):
        found_good = False
        for dx in [1.0, -1.0, 0.5, -0.5, 2.0, -2.0]:
            try:
                x_try = x0 + dx
                f_try = fc(x_try)
                if _is_finite(f_try):
                    a = x_try
                    fa = f_try
                    found_good = True
                    if abs(f_try) <= 1e-8:
                        return x_try
                    break
            except:
                continue
        if not found_good:
            return x0

    # Coarse scan with geometric expansion
    base_step = max(1.0, abs(x0) * 0.1)
    b = None
    fb = None
    scan_min_x = a
    scan_min_val = abs(fa)

    for i in range(12):
        if _elapsed() > _DEADLINE - 15 or fc.count > _CALL_BUDGET - 50:
            break
            
        step = base_step * (2 ** i)
        
        for direction in [1, -1]:
            if fc.count > _CALL_BUDGET - 50:
                break
            try:
                b_try = a + direction * step
                fb_try = fc(b_try)
                if _is_finite(fb_try):
                    if abs(fb_try) < scan_min_val:
                        scan_min_val = abs(fb_try)
                        scan_min_x = b_try
                    if abs(fb_try) <= 1e-8:
                        return b_try
                    if fa * fb_try < 0:
                        b = b_try
                        fb = fb_try
                        break
            except:
                pass
        if b is not None:
            break

    # Phase 2: If bracket found, use brentq
    if b is not None and fb is not None and fa * fb < 0:
        try:
            root = brentq(f, a, b, xtol=1e-12, rtol=1e-12, maxiter=100)
            if _is_finite(root):
                return root
        except Exception:
            pass

    # Phase 3: Minimize f(x)^2 using tight bracket from coarse scan
    def sq_f(x):
        try:
            val = fc(x)
            if not _is_finite(val):
                return 1e100
            return val * val
        except:
            return 1e100

    best_candidate_x = scan_min_x
    best_candidate_val = scan_min_val

    # Use the region around scan_min_x as the primary bracket
    # Expand width if needed
    primary_width = max(1.0, base_step)
    
    for width in [primary_width, primary_width * 5, primary_width * 20]:
        if _elapsed() > _DEADLINE - 10 or fc.count > _CALL_BUDGET - 30:
            break
        
        lower = scan_min_x - width
        upper = scan_min_x + width
        
        try:
            res = minimize_scalar(sq_f, bounds=(lower, upper), method='bounded', 
                                  options={'xatol': 1e-14})
            if res.success and _is_finite(res.x):
                val = abs(res.fun) if _is_finite(res.fun) else 1e100
                # res.fun is already squared, so take sqrt
                val = math.sqrt(res.fun) if _is_finite(res.fun) else 1e100
                if val < best_candidate_val:
                    best_candidate_val = val
                    best_candidate_x = res.x
                    if val <= 1e-8:
                        return best_candidate_x
        except Exception:
            pass

    # Also try original x0 region as backup
    if _elapsed() < _DEADLINE - 10 and fc.count < _CALL_BUDGET - 30:
        for width in [1.0, 5.0]:
            lower = x0 - width
            upper = x0 + width
            try:
                res = minimize_scalar(sq_f, bounds=(lower, upper), method='bounded',
                                      options={'xatol': 1e-14})
                if res.success and _is_finite(res.x):
                    val = math.sqrt(res.fun) if _is_finite(res.fun) else 1e100
                    if val < best_candidate_val:
                        best_candidate_val = val
                        best_candidate_x = res.x
                        if val <= 1e-8:
                            return best_candidate_x
            except Exception:
                pass

    # Final polish: if we're close, try to tighten with a very narrow window
    if best_candidate_val < 1e-4 and best_candidate_val > 1e-8:
        if _elapsed() < _DEADLINE - 5 and fc.count < _CALL_BUDGET - 10:
            try:
                narrow_width = max(1e-6, best_candidate_val * 100)
                lower = best_candidate_x - narrow_width
                upper = best_candidate_x + narrow_width
                res = minimize_scalar(sq_f, bounds=(lower, upper), method='bounded', 
                                      options={'xatol': 1e-15})
                if res.success and _is_finite(res.x):
                    val = math.sqrt(res.fun) if _is_finite(res.fun) else 1e100
                    if val < best_candidate_val:
                        best_candidate_val = val
                        best_candidate_x = res.x
            except:
                pass

    if best_candidate_val <= 1e-8:
        return best_candidate_x

    return best_candidate_x

# EVOLVE-BLOCK-END
