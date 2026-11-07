import math
import time
import numpy as np
from scipy.optimize import brentq

def find_root(f, x0):
    TOL = 1e-8
    MAX_CALLS = 495  # Leave 5 calls margin for safety
    START_TIME = time.time()
    DEADLINE = START_TIME + 115.0  # 120s limit, 5s margin

    calls = 0
    best_x = float(x0)
    best_f = float('inf')
    f_cache = {}

    def safe_f(x):
        nonlocal calls, best_x, best_f
        if calls >= MAX_CALLS:
            return float('inf')
        
        # Time check
        if time.time() > DEADLINE:
            return float('inf')
            
        key = round(float(x), 12)
        if key in f_cache:
            val = f_cache[key]
            if abs(val) < best_f:
                best_f = abs(val)
                best_x = float(x)
            return val

        calls += 1
        try:
            val = f(float(x))
            if not math.isfinite(val):
                f_cache[key] = float('inf')
                return float('inf')
            f_cache[key] = val
        except Exception:
            f_cache[key] = float('inf')
            return float('inf')
        
        if abs(val) < best_f:
            best_f = abs(val)
            best_x = float(x)
            
        return val

    x_start = float(x0)
    
    # 1. Check starting point
    fx0 = safe_f(x_start)
    if abs(fx0) <= TOL:
        return x_start

    # 2. Adaptive Sign-Change Scan
    # We scan outwards from x0 in logarithmic steps to find a sign change.
    # This is more robust for oscillatory functions than local minimization
    # because it doesn't get trapped in local minima of f^2.
    
    # Initial step size
    # If x0 is 0, use 1.0. Otherwise scale by magnitude.
    base_h = 1e-4 * max(1.0, abs(x_start))
    
    # We will try expanding brackets in both directions.
    # Strategy:
    # 1. Check x0 (done)
    # 2. Check x0 + h, x0 - h
    # 3. If no sign change, expand h exponentially.
    # 4. Also try finer steps within the last successful range if we find a sign change.
    
    # To be efficient, we'll scan a grid.
    # Grid points: x0 + k * h * 2^j
    # But simpler: just do a spiral scan.
    
    # Let's try a sequence of points:
    # x0, x0+h, x0-h, x0+2h, x0-2h, x0+4h, x0-4h ...
    # And also check midpoints if sign change is found between x0 and x0+h.
    
    # Actually, a dense local scan first is often better for "close" roots.
    # Let's do a local scan first: 100 points in [x0 - 10*base_h, x0 + 10*base_h]
    # Then expand if needed.
    
    # Local Scan
    local_span = 10 * base_h
    # If x0 is very small, base_h is 1e-4, span is 1e-3.
    # If x0 is large, base_h is large.
    
    # We want to use calls efficiently.
    # Let's use 50 calls for local scan.
    n_local = 50
    if calls + n_local <= MAX_CALLS:
        points = np.linspace(x_start - local_span, x_start + local_span, n_local)
        prev_x = None
        prev_f = None
        for p in points:
            if time.time() > DEADLINE - 2.0:
                break
            if calls >= MAX_CALLS:
                break
            fp = safe_f(p)
            if abs(fp) <= TOL:
                return p
            
            if prev_f is not None and math.isfinite(prev_f) and math.isfinite(fp):
                if prev_f * fp < 0:
                    # Sign change found!
                    # Refine using brentq
                    try:
                        root = brentq(safe_f, prev_x, p, xtol=TOL, rtol=1e-12, maxiter=100)
                        if math.isfinite(root):
                            return root
                    except Exception:
                        pass
            
            prev_x = p
            prev_f = fp

    # If local scan failed, expand outward.
    # We'll try a few larger scales.
    scales = [10, 100, 1000, 10000]
    for scale in scales:
        if time.time() > DEADLINE - 5.0:
            break
        if calls >= MAX_CALLS - 20:
            break
            
        span = scale * base_h
        n_scan = 50
        if calls + n_scan > MAX_CALLS:
            n_scan = MAX_CALLS - calls
            
        if n_scan < 10:
            continue
            
        points = np.linspace(x_start - span, x_start + span, n_scan)
        prev_x = None
        prev_f = None
        for p in points:
            if time.time() > DEADLINE - 2.0:
                break
            if calls >= MAX_CALLS:
                break
            fp = safe_f(p)
            if abs(fp) <= TOL:
                return p
            
            if prev_f is not None and math.isfinite(prev_f) and math.isfinite(fp):
                if prev_f * fp < 0:
                    # Sign change found!
                    try:
                        root = brentq(safe_f, prev_x, p, xtol=TOL, rtol=1e-12, maxiter=100)
                        if math.isfinite(root):
                            return root
                    except Exception:
                        pass
            
            prev_x = p
            prev_f = fp

    # Fallback: If no sign change found, maybe the root is a tangent (double root) or we missed it.
    # In that case, local minimization of f^2 is the backup.
    # But we've already used many calls.
    # Let's just return the best x found so far.
    
    return best_x
