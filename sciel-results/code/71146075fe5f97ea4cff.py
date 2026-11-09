"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import math
import time

def _safe_f(f, x):
    try:
        y = f(x)
        if not math.isfinite(y):
            return float('inf')
        return float(y)
    except Exception:
        return float('inf')

def _newton_refine(f, x0, max_iter=20):
    """Refine x0 using Newton's method with finite-difference derivative.
    Returns refined x and number of f-calls used."""
    x = x0
    calls = 0
    h = 1e-8 * max(1.0, abs(x))
    for _ in range(max_iter):
        fx = _safe_f(f, x)
        calls += 1
        if not math.isfinite(fx):
            break
        if abs(fx) <= 1e-8:
            return x, calls
        # Central difference derivative
        fxp = _safe_f(f, x + h)
        fxm = _safe_f(f, x - h)
        calls += 2
        if not (math.isfinite(fxp) and math.isfinite(fxm)):
            break
        fp = (fxp - fxm) / (2 * h)
        if abs(fp) < 1e-15:
            break
        x_new = x - fx / fp
        if not math.isfinite(x_new):
            break
        if abs(x_new - x) < 1e-14 * max(1.0, abs(x)):
            x = x_new
            break
        x = x_new
    return x, calls

def _find_roots_via_interpolation(f, x0):
    """
    Strategy: Use polynomial interpolation to model f as a polynomial.
    1. Sample f at several points around x0 to build an interpolating polynomial.
    2. Find all roots of the interpolating polynomial analytically/numerically.
    3. Verify candidate roots by evaluating f directly.
    4. Refine the best candidate with Newton if needed.
    Returns (best_x, total_calls).
    """
    total_calls = 0
    start_time = time.time()
    deadline = start_time + 130.0

    x0 = float(x0) if math.isfinite(x0) else 0.0

    # Phase 1: Quick check at x0
    f0 = _safe_f(f, x0)
    total_calls += 1
    if math.isfinite(f0) and abs(f0) <= 1e-8:
        return x0, total_calls

    best_x = x0
    best_abs = abs(f0) if math.isfinite(f0) else float('inf')

    # Phase 2: Build interpolating polynomial from N sample points
    # We'll sample at geometrically spaced points around x0
    # Use 7 points to fit a degree-6 polynomial (enough for most smooth polys)
    n_pts = 7
    # Sample points: x0, x0 ± scale, x0 ± 2*scale, x0 ± 4*scale
    base_scale = max(1.0, abs(x0)) * 0.5
    
    # Try multiple scales to ensure we capture the polynomial structure
    scales_to_try = [0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 50.0]
    
    candidate_roots = []
    
    for scale_factor in scales_to_try:
        if time.time() > deadline - 20.0:
            break
        if total_calls > 450:
            break
            
        s = base_scale * scale_factor
        # Generate 7 sample points
        pts = [x0 + d * s for d in [0, 1, -1, 2, -2, 4, -4]]
        vals = []
        valid = True
        for p in pts:
            v = _safe_f(f, p)
            total_calls += 1
            if not math.isfinite(v):
                valid = False
                break
            vals.append(v)
            if abs(v) < best_abs:
                best_abs = abs(v)
                best_x = p
            if abs(v) <= 1e-8:
                return p, total_calls
        
        if not valid:
            continue
            
        # Check if we have a sign change among the points (indicates a root)
        # Also check if any point is very close to zero
        for i in range(len(pts) - 1):
            if vals[i] * vals[i+1] < 0:
                # Sign change between pts[i] and pts[i+1]
                # Add both as potential bracket endpoints
                candidate_roots.append((pts[i], pts[i+1], vals[i], vals[i+1]))
        
        # Try to fit a polynomial and find its roots
        # Use numpy for polynomial fitting
        try:
            import numpy as np
            # Fit polynomial of degree n_pts-1
            # Sort points for better conditioning
            idx = sorted(range(len(pts)), key=lambda i: pts[i])
            xs = [pts[i] for i in idx]
            ys = [vals[i] for i in idx]
            
            # Shift and scale for numerical stability
            x_mean = sum(xs) / len(xs)
            x_range = max(max(xs) - min(xs), 1e-10)
            xs_norm = [(x - x_mean) / x_range for x in xs]
            
            coeffs = np.polyfit(xs_norm, ys, deg=len(xs) - 1)
            
            # Find roots of the normalized polynomial
            roots_norm = np.roots(coeffs)
            
            # Map back to original scale
            for r_norm in roots_norm:
                if not np.isfinite(r_norm):
                    continue
                r = r_norm * x_range + x_mean
                if math.isfinite(r):
                    # Only consider roots within a reasonable range
                    if abs(r - x0) < 100 * base_scale + 100:
                        candidate_roots.append((r, r, None, None))
        except Exception:
            pass
    
    # Phase 3: Verify candidate roots
    verified_roots = []
    for item in candidate_roots:
        if time.time() > deadline - 10.0:
            break
        if total_calls > 480:
            break
            
        if len(item) == 2 and item[2] is None:
            # Single point candidate
            x_cand = item[0]
            fc = _safe_f(f, x_cand)
            total_calls += 1
            if math.isfinite(fc):
                if abs(fc) <= 1e-8:
                    return x_cand, total_calls
                if abs(fc) < best_abs:
                    best_abs = abs(fc)
                    best_x = x_cand
                verified_roots.append((x_cand, fc))
        else:
            # Bracket [a, b] with fa, fb
            a, b, fa, fb = item
            if fa is None or fb is None:
                continue
            # Use Illinois method to converge
            lo, hi = a, b
            flo, fhi = fa, fb
            for _ in range(30):
                if time.time() > deadline - 5.0:
                    break
                if total_calls > 490:
                    break
                if abs(flo) < abs(fhi):
                    if abs(fhi - flo) > 1e-15:
                        m = hi - fhi * (hi - lo) / (fhi - flo)
                    else:
                        m = (lo + hi) / 2.0
                else:
                    if abs(fhi - flo) > 1e-15:
                        m = lo - flo * (hi - lo) / (fhi - flo)
                    else:
                        m = (lo + hi) / 2.0
                if m <= lo or m >= hi:
                    m = (lo + hi) / 2.0
                fm = _safe_f(f, m)
                total_calls += 1
                if not math.isfinite(fm):
                    m = (lo + hi) / 2.0
                    fm = _safe_f(f, m)
                    total_calls += 1
                    if not math.isfinite(fm):
                        break
                if abs(fm) < best_abs:
                    best_abs = abs(fm)
                    best_x = m
                if abs(fm) <= 1e-8:
                    return m, total_calls
                if flo * fm <= 0:
                    hi, fhi = m, fm
                else:
                    lo, flo = m, fm
                if abs(hi - lo) < 1e-13:
                    break
            # Return midpoint as best estimate
            mid = (lo + hi) / 2.0
            verified_roots.append((mid, None))
    
    # Phase 4: If we have good candidates, refine the best one with Newton
    if verified_roots:
        # Pick the candidate with smallest |f|
        best_cand = None
        best_cand_abs = float('inf')
        for x_c, f_c in verified_roots:
            if f_c is not None and math.isfinite(f_c) and abs(f_c) < best_cand_abs:
                best_cand_abs = abs(f_c)
                best_cand = x_c
        
        if best_cand is not None and best_cand_abs < 1e-4:
            # Refine with Newton
            refined_x, refine_calls = _newton_refine(f, best_cand, max_iter=15)
            total_calls += refine_calls
            fr = _safe_f(f, refined_x)
            total_calls += 1
            if math.isfinite(fr) and abs(fr) <= 1e-8:
                return refined_x, total_calls
            if math.isfinite(fr) and abs(fr) < best_abs:
                best_abs = abs(fr)
                best_x = refined_x
    
    # Phase 5: Fallback - expanding bracket search if nothing worked
    if best_abs > 1e-8 and time.time() < deadline - 5.0:
        a = best_x
        fa = _safe_f(f, a)
        total_calls += 1
        if not math.isfinite(fa):
            a = 0.0
            fa = _safe_f(f, a)
            total_calls += 1
            if not math.isfinite(fa):
                return best_x, total_calls
        
        if abs(fa) < best_abs:
            best_abs = abs(fa)
            best_x = a
        if abs(fa) <= 1e-8:
            return a, total_calls
        
        scale = 1.0
        while time.time() < deadline - 2.0 and scale < 1e8 and total_calls < 495:
            for direction in [1.0, -1.0]:
                if total_calls >= 495:
                    break
                b = a + direction * scale
                fb = _safe_f(f, b)
                total_calls += 1
                if not math.isfinite(fb):
                    continue
                if abs(fb) < best_abs:
                    best_abs = abs(fb)
                    best_x = b
                if abs(fb) <= 1e-8:
                    return b, total_calls
                if fa * fb < 0:
                    # Bracket found, converge
                    lo, hi = a, b
                    flo, fhi = fa, fb
                    for _ in range(40):
                        if total_calls >= 498:
                            break
                        if abs(flo) < abs(fhi):
                            if abs(fhi - flo) > 1e-15:
                                m = hi - fhi * (hi - lo) / (fhi - flo)
                            else:
                                m = (lo + hi) / 2.0
                        else:
                            if abs(fhi - flo) > 1e-15:
                                m = lo - flo * (hi - lo) / (fhi - flo)
                            else:
                                m = (lo + hi) / 2.0
                        if m <= lo or m >= hi:
                            m = (lo + hi) / 2.0
                        fm = _safe_f(f, m)
                        total_calls += 1
                        if not math.isfinite(fm):
                            m = (lo + hi) / 2.0
                            fm = _safe_f(f, m)
                            total_calls += 1
                            if not math.isfinite(fm):
                                break
                        if abs(fm) < best_abs:
                            best_abs = abs(fm)
                            best_x = m
                        if abs(fm) <= 1e-8:
                            return m, total_calls
                        if flo * fm <= 0:
                            hi, fhi = m, fm
                        else:
                            lo, flo = m, fm
                        if abs(hi - lo) < 1e-13:
                            break
                    return (lo + hi) / 2.0, total_calls
            scale *= 2.0
    
    return best_x, total_calls

def find_root(f, x0):
    """
    Find a root x such that |f(x)| <= 1e-8.
    Strategy: Polynomial interpolation-based global search.
    1. Check x0 directly.
    2. Sample f at multiple scales around x0 to build interpolating polynomials.
    3. Find roots of interpolating polynomials and verify with direct evaluation.
    4. Refine best candidate with Newton's method.
    5. Fallback to expanding bracket search if needed.
    """
    start_time = time.time()
    deadline = start_time + 140.0

    x0 = float(x0) if math.isfinite(x0) else 0.0

    try:
        result, _ = _find_roots_via_interpolation(f, x0)
        if math.isfinite(result):
            return result
    except Exception:
        pass

    # Ultimate fallback: simple expanding search
    best_x = x0
    best_abs = float('inf')
    fx0 = _safe_f(f, x0)
    if math.isfinite(fx0):
        best_abs = abs(fx0)
        if best_abs <= 1e-8:
            return x0

    a = x0
    fa = fx0
    scale = 1.0
    while time.time() < deadline - 5.0 and scale < 1e8:
        for d in [1.0, -1.0]:
            if time.time() > deadline - 5.0:
                break
            b = a + d * scale
            fb = _safe_f(f, b)
            if not math.isfinite(fb):
                continue
            if abs(fb) < best_abs:
                best_abs = abs(fb)
                best_x = b
            if abs(fb) <= 1e-8:
                return b
            if fa * fb < 0:
                lo, hi = a, b
                flo, fhi = fa, fb
                for _ in range(50):
                    if time.time() > deadline - 2.0:
                        break
                    m = (lo + hi) / 2.0
                    fm = _safe_f(f, m)
                    if not math.isfinite(fm):
                        break
                    if abs(fm) < best_abs:
                        best_abs = abs(fm)
                        best_x = m
                    if abs(fm) <= 1e-8:
                        return m
                    if flo * fm <= 0:
                        hi, fhi = m, fm
                    else:
                        lo, flo = m, fm
                return (lo + hi) / 2.0
        scale *= 2.0

    return best_x

# EVOLVE-BLOCK-END
