"""Model-and-optimize root finder for ill-conditioned equations."""
import time
import math

# EVOLVE-BLOCK-START
def find_root(f, x0):
    _t0 = time.time()
    _TIME_LIMIT = 115.0  # safety margin before 120s
    _MAX_EVALS = 450
    _evals = 0

    def _f(x):
        nonlocal _evals
        _evals += 1
        try:
            return float(f(x))
        except Exception:
            return float('nan')

    def _time_ok():
        return (time.time() - _t0) < _TIME_LIMIT

    def _budget_ok():
        return _evals < _MAX_EVALS

    def _safe(x):
        return math.isfinite(x)

    # Best candidate tracking
    best_x = float(x0)
    best_fx = _f(best_x)
    best_abs = abs(best_fx) if _safe(best_fx) else float('inf')

    def _update(x, fx):
        nonlocal best_x, best_fx, best_abs
        if not _safe(fx):
            return False
        af = abs(fx)
        if af < best_abs:
            best_abs = af
            best_x = float(x)
            best_fx = float(fx)
            return af <= 1e-8
        return False

    _update(best_x, best_fx)
    if best_abs <= 1e-8:
        return best_x

    # Maintain a global sorted list of all evaluated points for PLI
    # Each entry: (x, fx)
    _all_points = []

    def _add_point(x, fx):
        """Add a point to the sorted global list and update best."""
        if not _safe(fx):
            return
        _update(x, fx)
        # Insert in sorted order by x
        _all_points.append((x, fx))
        _all_points.sort(key=lambda p: p[0])

    # PLI: find the two points with smallest |f| that form a sign change,
    # then do linear interpolation between them.
    def _pli_step():
        """Perform one PLI step: find best pair and interpolate."""
        if len(_all_points) < 2:
            return None

        # Find the two points with smallest |f| values
        # Prefer pairs that have a sign change
        # Strategy: sort by |f|, try first few pairs for sign change
        sorted_by_abs = sorted(_all_points, key=lambda p: abs(p[1]))
        
        # Try to find a sign-changing pair among the smallest |f| points
        candidates = sorted_by_abs[:min(10, len(_all_points))]
        
        best_pair = None
        best_pair_score = float('inf')
        
        for i in range(len(candidates)):
            for j in range(i + 1, len(candidates)):
                xi, fi = candidates[i]
                xj, fj = candidates[j]
                if (fi <= 0 and fj >= 0) or (fi >= 0 and fj <= 0):
                    # Sign change - good
                    score = abs(fi) + abs(fj)
                    if score < best_pair_score:
                        best_pair_score = score
                        best_pair = (xi, fi, xj, fj)
        
        if best_pair is None:
            # No sign change found; use the two closest in x with smallest |f|
            # Just use the two points with smallest |f|
            if len(candidates) >= 2:
                best_pair = (candidates[0][0], candidates[0][1],
                             candidates[1][0], candidates[1][1])
            else:
                return None

        xi, fi, xj, fj = best_pair
        denom = fj - fi
        if abs(denom) < 1e-30:
            return None
        x_new = xi - fi * (xj - xi) / denom
        if not _safe(x_new):
            return None
        return x_new

    # Secant method step
    def _secant(x1, f1, x2, f2):
        if not all(_safe(v) for v in [x1, f1, x2, f2]):
            return None
        denom = f2 - f1
        if abs(denom) < 1e-30:
            return None
        x_new = x1 - f1 * (x2 - x1) / denom
        if not _safe(x_new):
            return None
        return x_new

    # Bisection
    def _bisect(a, b, fa, fb, max_iter=60):
        for _ in range(max_iter):
            if not _time_ok() or not _budget_ok():
                break
            mid = (a + b) / 2.0
            fmid = _f(mid)
            if not _safe(fmid):
                break
            _update(mid, fmid)
            _all_points.append((mid, fmid))
            _all_points.sort(key=lambda p: p[0])
            if abs(fmid) <= 1e-8:
                return mid
            if (fmid <= 0 and fa <= 0) or (fmid >= 0 and fa >= 0):
                a, fa = mid, fmid
            else:
                b, fb = mid, fmid
            if abs(b - a) < 1e-14 * max(1.0, abs(a), abs(b)):
                break
        return (a + b) / 2.0

    # --- Phase 1: Initial point exploration with PLI ---
    x1 = float(x0)
    f1 = _f(x1)
    if not _safe(f1):
        x1 = best_x
        f1 = best_fx
    _update(x1, f1)
    _add_point(x1, f1)

    if abs(f1) <= 1e-8:
        return x1

    # Try several offsets to find a second point (ideally with opposite sign)
    scale = max(1.0, abs(x1))
    offsets = [0.1, -0.1, 0.5, -0.5, 1.0, -1.0, 2.0, -2.0, 0.01, -0.01]
    x2 = None
    f2 = None
    for off in offsets:
        if not _budget_ok():
            break
        cand = x1 + off * scale
        fc = _f(cand)
        if not _safe(fc):
            continue
        _add_point(cand, fc)
        if abs(fc) <= 1e-8:
            return cand
        # Prefer a point with opposite sign to f1
        if (f1 <= 0 and fc >= 0) or (f1 >= 0 and fc <= 0):
            x2, f2 = cand, fc
            break
        if x2 is None:
            x2, f2 = cand, fc
        elif abs(fc) < abs(f2):
            x2, f2 = cand, fc

    if x2 is None:
        x2 = x1 + 0.1 * scale
        f2 = _f(x2)
        _add_point(x2, f2)
        if not _safe(f2):
            x2 = x1 - 0.1 * scale
            f2 = _f(x2)
            _add_point(x2, f2)

    if abs(f2) <= 1e-8:
        return x2

    # --- Phase 2: PLI-driven root finding ---
    bracket_a = bracket_b = None
    bracket_fa = bracket_fb = None

    for i in range(80):
        if not _time_ok() or not _budget_ok():
            break

        # Try PLI first
        x_new = _pli_step()

        if x_new is None and len(_all_points) >= 2:
            # Fallback to secant on the two points with smallest |f|
            sorted_by_abs = sorted(_all_points, key=lambda p: abs(p[1]))
            x_a, f_a = sorted_by_abs[0]
            x_b, f_b = sorted_by_abs[1]
            x_new = _secant(x_a, f_a, x_b, f_b)

        if x_new is None:
            # Last resort: small step from best point
            x_new = best_x + 0.1 * max(1.0, abs(best_x))

        max_step = 5.0 * max(1.0, abs(best_x))
        if abs(x_new - best_x) > max_step:
            x_new = best_x + max_step * (1 if x_new > best_x else -1)

        if not _safe(x_new):
            x_new = best_x + 0.01 * max(1.0, abs(best_x))
            if not _safe(x_new):
                break

        # Check for duplicate
        if any(abs(x_new - px) < 1e-14 * max(1.0, abs(px)) for px, _ in _all_points):
            x_new = x_new + 1e-10 * max(1.0, abs(x_new))

        f_new = _f(x_new)
        if not _safe(f_new):
            # Try a different direction
            x_new = best_x - 0.5 * (x_new - best_x)
            f_new = _f(x_new)
            if not _safe(f_new):
                break

        _add_point(x_new, f_new)
        if abs(f_new) <= 1e-8:
            return x_new

        # Track bracket
        # Find if any pair in _all_points forms a bracket
        for i_idx in range(len(_all_points)):
            for j_idx in range(i_idx + 1, len(_all_points)):
                xi, fi = _all_points[i_idx]
                xj, fj = _all_points[j_idx]
                if (fi <= 0 and fj >= 0) or (fi >= 0 and fj <= 0):
                    if bracket_a is None or (abs(fi) + abs(fj)) < (abs(bracket_fa) + abs(bracket_fb)):
                        bracket_a, bracket_b = xi, xj
                        bracket_fa, bracket_fb = fi, fj
                        if bracket_a > bracket_b:
                            bracket_a, bracket_b = bracket_b, bracket_a
                            bracket_fa, bracket_fb = bracket_fb, bracket_fa

        if abs(x_new - best_x) < 1e-14 * max(1.0, abs(x_new)):
            break

    # --- Phase 3: Bracket refinement with bisection + PLI ---
    if bracket_a is not None and bracket_b is not None:
        if (bracket_fa <= 0 and bracket_fb >= 0) or (bracket_fa >= 0 and bracket_fb <= 0):
            a, b = bracket_a, bracket_b
            fa, fb = bracket_fa, bracket_fb

            for i in range(60):
                if not _time_ok() or not _budget_ok():
                    break

                # Try PLI within bracket
                x_trial = None
                # Find points within [a, b]
                bracket_points = [(x, fx) for x, fx in _all_points if a <= x <= b]
                if len(bracket_points) >= 2:
                    x_trial = _pli_step()
                    if x_trial is not None and _safe(x_trial) and (a < x_trial < b):
                        pass
                    else:
                        x_trial = (a + b) / 2.0
                else:
                    x_trial = (a + b) / 2.0

                if not _safe(x_trial) or not (a < x_trial < b):
                    x_trial = (a + b) / 2.0
                    if not _safe(x_trial):
                        break

                f_trial = _f(x_trial)
                if not _safe(f_trial):
                    break
                _add_point(x_trial, f_trial)
                if abs(f_trial) <= 1e-8:
                    return x_trial

                if (f_trial <= 0 and fa <= 0) or (f_trial >= 0 and fa >= 0):
                    a, fa = x_trial, f_trial
                else:
                    b, fb = x_trial, f_trial

                if abs(b - a) < 1e-14 * max(1.0, abs(a), abs(b)):
                    break

            res = (a + b) / 2.0
            fres = _f(res)
            _add_point(res, fres)
            if abs(fres) <= 1e-8:
                return res

    # --- Phase 4: Scan for brackets ---
    if not _time_ok() or not _budget_ok():
        return best_x

    base = best_x
    scan_scale = max(1.0, abs(base))
    found = False
    for mult in [0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0, 200.0]:
        if not _time_ok() or not _budget_ok():
            break
        for sign in [-1, 1]:
            a = base - sign * mult * scan_scale
            b = base + sign * mult * scan_scale
            if a == b:
                continue
            fa = _f(a)
            fb = _f(b)
            if not (_safe(fa) and _safe(fb)):
                continue
            _add_point(a, fa)
            _add_point(b, fb)
            if abs(fa) <= 1e-8:
                return a
            if abs(fb) <= 1e-8:
                return b
            if (fa <= 0 and fb >= 0) or (fa >= 0 and fb <= 0):
                res = _bisect(a, b, fa, fb, max_iter=50)
                fres = _f(res)
                _add_point(res, fres)
                if abs(fres) <= 1e-8:
                    return res
                found = True
                break
        if found:
            break

    # --- Phase 5: Final PLI polish ---
    if _time_ok() and _budget_ok():
        for _ in range(30):
            if not _time_ok() or not _budget_ok():
                break
            x_new = _pli_step()
            if x_new is None:
                break
            f_new = _f(x_new)
            if not _safe(f_new):
                break
            _add_point(x_new, f_new)
            if abs(f_new) <= 1e-8:
                return x_new
            if abs(x_new - best_x) < 1e-14 * max(1.0, abs(x_new)):
                break

    return best_x
# EVOLVE-BLOCK-END
