import math
import time

_START_TIME = time.time()
_WALL_LIMIT = 115.0  # safety margin before 120s (evaluator kills at 150s)

def _time_left():
    return _WALL_LIMIT - (time.time() - _START_TIME)

def _is_finite(x):
    return x is not None and isinstance(x, (int, float)) and math.isfinite(x)

def _safe_call(f, x):
    """Call f(x), returning (value, ok). ok=False on exception or non-finite."""
    try:
        val = f(x)
    except Exception:
        return 0.0, False
    if not _is_finite(val):
        return 0.0, False
    return val, True

def _secant_track(f, x_a, fx_a, x_b, fx_b, max_iter=60, deadline=None):
    """
    Run secant method from two initial points.
    Returns (best_x, best_fx_abs, calls_made).
    Stops when |f| <= 1e-8, max_iter, deadline exceeded, or non-finite.
    """
    best_x = x_a if abs(fx_a) <= abs(fx_b) else x_b
    best_abs = min(abs(fx_a), abs(fx_b))
    calls = 2  # initial two points already evaluated
    
    if best_abs <= 1e-8:
        return best_x, best_abs, calls
    
    for _ in range(max_iter):
        if deadline is not None and _time_left() < 1.0:
            break
        
        denom = fx_b - fx_a
        if abs(denom) < 1e-300:
            # Degenerate: midpoint
            x_new = 0.5 * (x_a + x_b)
        else:
            x_new = x_b - fx_b * (x_b - x_a) / denom
        
        if not _is_finite(x_new):
            break
        
        fx_new, ok = _safe_call(f, x_new)
        calls += 1
        if not ok:
            break
        
        abs_new = abs(fx_new)
        if abs_new < best_abs:
            best_x = x_new
            best_abs = abs_new
        
        if abs_new <= 1e-8:
            return x_new, abs_new, calls
        
        # Shift: (a, b) -> (b, new)
        x_a, fx_a = x_b, fx_b
        x_b, fx_new = x_new, fx_new
        fx_b = fx_new
    
    return best_x, best_abs, calls

def find_root(f, x0):
    x = float(x0)
    
    # Evaluate at x0
    fx0, ok0 = _safe_call(f, x)
    if not ok0:
        return x
    if abs(fx0) <= 1e-8:
        return x
    
    best_x = x
    best_abs = abs(fx0)
    total_calls = 1
    
    # Generate a geometric spread of starting points around x0
    # This is the key structural difference: multi-start instead of single trajectory
    scale = max(1.0, abs(x))
    # Offsets: small to large, covering multiple orders of magnitude
    # For polynomials, roots can be at various distances
    offsets = []
    # Near neighbors
    offsets.append(1e-3 * scale)
    offsets.append(-1e-3 * scale)
    offsets.append(1e-2 * scale)
    offsets.append(-1e-2 * scale)
    offsets.append(1e-1 * scale)
    offsets.append(-1e-1 * scale)
    offsets.append(1.0 * scale)
    offsets.append(-1.0 * scale)
    offsets.append(10.0 * scale)
    offsets.append(-10.0 * scale)
    
    # Filter to finite, distinct points
    candidates = []
    seen = set()
    for off in offsets:
        xc = x + off
        if not _is_finite(xc):
            continue
        # Use a quantized key to avoid near-duplicates
        key = round(xc, 12)
        if key in seen:
            continue
        seen.add(key)
        candidates.append(xc)
    
    # We'll launch secant tracks from pairs of nearby candidates
    # Strategy: for each candidate, pair it with x0 (or the previous candidate)
    # to form a secant pair. We process them in order of proximity to x0,
    # stopping early if we find a root.
    
    # Sort candidates by distance from x0
    candidates.sort(key=lambda c: abs(c - x))
    
    # We'll try up to 4 secant tracks, each with a limited budget
    # Track 1: x0 and nearest candidate
    # Track 2: next two candidates
    # etc.
    
    num_tracks = min(4, len(candidates))
    tracks = []
    
    for t in range(num_tracks):
        if _time_left() < 1.0:
            break
        
        if t == 0:
            # First track: x0 and candidates[0]
            xa, fa = x, fx0
            xb = candidates[0]
            fb, ok = _safe_call(f, xb)
            total_calls += 1
            if not ok:
                continue
            if abs(fb) < best_abs:
                best_x = xb
                best_abs = abs(fb)
            if abs(fb) <= 1e-8:
                return xb
            tracks.append((xa, fa, xb, fb))
        else:
            # Subsequent tracks: use pairs of candidates
            idx1 = 1 + (t - 1) * 2
            idx2 = 2 + (t - 1) * 2
            if idx2 >= len(candidates):
                break
            xa = candidates[idx1]
            fa, ok1 = _safe_call(f, xa)
            total_calls += 1
            if not ok1:
                continue
            xb = candidates[idx2]
            fb, ok2 = _safe_call(f, xb)
            total_calls += 1
            if not ok2:
                continue
            if abs(fa) < best_abs:
                best_x = xa
                best_abs = abs(fa)
            if abs(fb) < best_abs:
                best_x = xb
                best_abs = abs(fb)
            if abs(fa) <= 1e-8:
                return xa
            if abs(fb) <= 1e-8:
                return xb
            tracks.append((xa, fa, xb, fb))
    
    # Now run secant iterations on the most promising tracks
    # Sort tracks by their current best residual (lower is better)
    tracks.sort(key=lambda tr: min(abs(tr[1]), abs(tr[3])))
    
    # Allocate remaining budget to tracks in order of promise
    # Each track gets up to 40 iterations, but we stop early if a root is found
    for (xa, fa, xb, fb) in tracks:
        if _time_left() < 1.0:
            break
        if best_abs <= 1e-8:
            break
        
        tx, tbest, tcalls = _secant_track(f, xa, fa, xb, fb, max_iter=40)
        total_calls += tcalls - 2  # subtract the initial 2 already counted
        
        if tbest < best_abs:
            best_x = tx
            best_abs = tbest
        
        if best_abs <= 1e-8:
            break
    
    return best_x
