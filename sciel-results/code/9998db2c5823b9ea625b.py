"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import math
import time
from scipy.optimize import brentq


def find_root(f, x0):
    TOL = 1e-8
    MAX_CALLS = 495  # Leave margin below 500
    DEADLINE = time.time() + 140  # 140s out of 150s limit

    calls = 0
    best_x = float(x0)
    best_abs = float('inf')

    def safe_f(x):
        nonlocal calls, best_x, best_abs
        if calls >= MAX_CALLS:
            return float('inf')
        if time.time() > DEADLINE:
            return float('inf')
        calls += 1
        try:
            val = f(float(x))
            if not math.isfinite(val):
                return float('inf')
            abs_val = abs(val)
            if abs_val < best_abs:
                best_abs = abs_val
                best_x = float(x)
            return val
        except Exception:
            return float('inf')

    # Check initial point
    fx0 = safe_f(x0)
    if abs(fx0) <= TOL:
        return x0

    # Phase 1: Scan outward from x0 to find a bracket [a, b] where sign changes
    # Use logarithmic spacing to cover a wide range efficiently
    def find_bracket(fx0, x0):
        """Search for a sign-change bracket by expanding outward from x0."""
        # First, try small steps in both directions
        step = 1e-3
        for _ in range(15):
            if calls >= MAX_CALLS or time.time() > DEADLINE:
                return None
            # Try x0 + step
            x_r = x0 + step
            fx_r = safe_f(x_r)
            if math.isfinite(fx0) and math.isfinite(fx_r) and fx0 * fx_r < 0:
                return (x0, x_r)
            # Try x0 - step
            x_l = x0 - step
            fx_l = safe_f(x_l)
            if math.isfinite(fx0) and math.isfinite(fx_l) and fx0 * fx_l < 0:
                return (x_l, x0)
            step *= 2.0

        # If no bracket found with x0, try scanning from a few offset points
        # Try offsets: 1, -1, 0.5, -0.5, 2, -2, etc.
        offsets = [1.0, -1.0, 0.5, -0.5, 2.0, -2.0, 0.1, -0.1, 5.0, -5.0]
        for off in offsets:
            if calls >= MAX_CALLS or time.time() > DEADLINE:
                return None
            x_start = x0 + off
            fx_start = safe_f(x_start)
            if not math.isfinite(fx_start):
                continue
            if abs(fx_start) <= TOL:
                return (x_start, x_start)
            # Scan from x_start outward
            step = 0.1
            for _ in range(20):
                if calls >= MAX_CALLS or time.time() > DEADLINE:
                    break
                x_next = x_start + step
                fx_next = safe_f(x_next)
                if math.isfinite(fx_start) and math.isfinite(fx_next) and fx_start * fx_next < 0:
                    return (x_start, x_next)
                if math.isfinite(fx_next) and abs(fx_next) <= TOL:
                    return (x_next, x_next)
                # Also try the other direction
                x_next2 = x_start - step
                fx_next2 = safe_f(x_next2)
                if math.isfinite(fx_start) and math.isfinite(fx_next2) and fx_start * fx_next2 < 0:
                    return (x_next2, x_start)
                if math.isfinite(fx_next2) and abs(fx_next2) <= TOL:
                    return (x_next2, x_next2)
                step *= 2.0
            if calls >= MAX_CALLS or time.time() > DEADLINE:
                break

        return None

    bracket = find_bracket(fx0, x0)

    if bracket is not None:
        a, b = bracket
        if a == b:
            # Already found a root
            return a
        try:
            root = brentq(safe_f, a, b, xtol=1e-12, rtol=1e-12, maxiter=100)
            if math.isfinite(root):
                # Verify
                fr = safe_f(root)
                if abs(fr) <= TOL:
                    return root
                # brentq should have found it, but let's return it anyway
                return root
        except Exception:
            pass

    # Phase 2: If bracketing failed, try a more aggressive scan
    # Scan a grid of points and look for sign changes
    def grid_scan():
        if calls >= MAX_CALLS or time.time() > DEADLINE:
            return None
        # Scan from -10 to 10 with adaptive step
        points = []
        vals = []
        # Start with a coarse grid
        for i in range(-50, 51):
            if calls >= MAX_CALLS or time.time() > DEADLINE:
                break
            x = i * 0.1
            fx = safe_f(x)
            points.append(x)
            vals.append(fx)
            if math.isfinite(fx) and abs(fx) <= TOL:
                return x

        # Look for sign changes in the grid
        for i in range(len(points) - 1):
            if not math.isfinite(vals[i]) or not math.isfinite(vals[i+1]):
                continue
            if vals[i] * vals[i+1] < 0:
                try:
                    root = brentq(safe_f, points[i], points[i+1], xtol=1e-12, rtol=1e-12, maxiter=100)
                    if math.isfinite(root):
                        fr = safe_f(root)
                        if abs(fr) <= TOL:
                            return root
                        return root
                except Exception:
                    pass

        return None

    result = grid_scan()
    if result is not None:
        return result

    # Phase 3: Last resort - try Newton's method from x0 with safeguards
    x = float(x0)
    for _ in range(30):
        if calls >= MAX_CALLS or time.time() > DEADLINE:
            break
        fx = safe_f(x)
        if not math.isfinite(fx):
            break
        if abs(fx) <= TOL:
            return x
        h = 1e-7 * max(1.0, abs(x))
        fx_h = safe_f(x + h)
        if not math.isfinite(fx_h):
            break
        d = (fx_h - fx) / h
        if d == 0 or not math.isfinite(d):
            break
        x_new = x - fx / d
        if not math.isfinite(x_new):
            break
        x = x_new

    return best_x
# EVOLVE-BLOCK-END
