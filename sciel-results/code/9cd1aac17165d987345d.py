import math
import time
import numpy as np
from scipy.optimize import minimize_scalar

N = 11
D = math.sqrt(3)

# Precompute base hexagon vertices (flat-top, angle 0)
_BASE_ANGLES = np.deg2rad(np.array([0.0, 60.0, 120.0, 180.0, 240.0, 300.0]))
_BASE_COS = np.cos(_BASE_ANGLES)
_BASE_SIN = np.sin(_BASE_ANGLES)


def _hex_vertices(cx, cy, angle_deg, s=1.0):
    angles = _BASE_ANGLES + np.deg2rad(angle_deg)
    vs = np.column_stack([s * np.cos(angles), s * np.sin(angles)])
    return vs + np.array([cx, cy])


def _point_in_polygon(pt, poly):
    n = len(poly)
    sign = 0
    for i in range(n):
        a = poly[i]
        b = poly[(i + 1) % n]
        cross = (b[0] - a[0]) * (pt[1] - a[1]) - (b[1] - a[1]) * (pt[0] - a[0])
        if cross > 1e-12:
            if sign < 0:
                return False
            sign = 1
        elif cross < -1e-12:
            if sign > 0:
                return False
            sign = -1
    return True


def _seg_seg_dist(p1, p2, p3, p4):
    d1 = p2 - p1
    d2 = p4 - p3
    r = p1 - p3
    a = np.dot(d1, d1)
    e = np.dot(d2, d2)
    f = np.dot(d2, r)
    if a < 1e-12 and e < 1e-12:
        return np.linalg.norm(p1 - p3)
    if a < 1e-12:
        t = np.clip(f / e, 0.0, 1.0)
        return np.linalg.norm(p1 - (p3 + t * d2))
    c = np.dot(d1, r)
    if e < 1e-12:
        s = np.clip(c / a, 0.0, 1.0)
        return np.linalg.norm((p1 + s * d1) - p3)
    b = np.dot(d1, d2)
    denom = a * e - b * b
    if denom > 1e-12:
        s = (b * f - c * e) / denom
        s = np.clip(s, 0.0, 1.0)
        t = (b * s + f) / e
        if t < 0:
            t = 0.0
            s = np.clip(c / a, 0.0, 1.0)
        elif t > 1:
            t = 1.0
            s = np.clip((b + f) / a, 0.0, 1.0)
        return np.linalg.norm((p1 + s * d1) - (p3 + t * d2))
    t = np.clip(f / e, 0.0, 1.0)
    return np.linalg.norm(p1 - (p3 + t * d2))


def _poly_separation(p1, p2):
    for v in p1:
        if _point_in_polygon(v, p2):
            return 0.0
    for v in p2:
        if _point_in_polygon(v, p1):
            return 0.0
    min_dist = float('inf')
    n1, n2 = len(p1), len(p2)
    for i in range(n1):
        a1 = p1[i]
        a2 = p1[(i + 1) % n1]
        for j in range(n2):
            b1 = p2[j]
            b2 = p2[(j + 1) % n2]
            d = _seg_seg_dist(a1, a2, b1, b2)
            if d < min_dist:
                min_dist = d
    return min_dist


def _outer_hex_vertices(cx, cy, side, angle_deg):
    angles = _BASE_ANGLES + np.deg2rad(angle_deg)
    vs = np.column_stack([side * np.cos(angles), side * np.sin(angles)])
    return vs + np.array([cx, cy])


def _build_flower_centers():
    """Build the standard flower packing: 1 center + 6 ring + 4 outer."""
    eps = 1e-10
    centers = np.array([[0.0, 0.0]])
    # Inner ring: 6 hexagons at distance sqrt(3)
    for k in range(6):
        angle = np.deg2rad(k * 60)
        r = D + eps
        centers = np.vstack([centers, [r * np.cos(angle), r * np.sin(angle)]])
    # Outer ring: 4 hexagons at distance 3.0
    for k in range(4):
        angle = np.deg2rad(30 + k * 60)
        r = 3.0 + 2 * eps
        centers = np.vstack([centers, [r * np.cos(angle), r * np.sin(angle)]])
    return centers


def _vectorized_side(centers, inner_angle_deg, outer_angle_deg):
    """
    Fast vectorized computation of minimum outer hexagon side length.
    All inner hexagons have the same orientation (inner_angle_deg).
    The outer hexagon has rotation outer_angle_deg.
    """
    n = centers.shape[0]
    xs = centers[:, 0]
    ys = centers[:, 1]
    inner_rad = np.deg2rad(inner_angle_deg)
    cos_a = np.cos(inner_rad)
    sin_a = np.sin(inner_rad)
    # Rotate base vertices
    vx = _BASE_COS * cos_a - _BASE_SIN * sin_a + xs[:, np.newaxis]
    vy = _BASE_SIN * cos_a + _BASE_COS * sin_a + ys[:, np.newaxis]
    # Outer hexagon edge normals
    normal_angles = np.deg2rad(30.0 + outer_angle_deg + np.arange(6) * 60.0)
    nx = np.cos(normal_angles)
    ny = np.sin(normal_angles)
    # Project all vertices of all hexagons onto all normals
    proj = vx[:, :, np.newaxis] * nx[np.newaxis, np.newaxis, :] + \
           vy[:, :, np.newaxis] * ny[np.newaxis, np.newaxis, :]
    max_per_hex = np.abs(proj).max(axis=1)
    max_over_all = max_per_hex.max(axis=0)
    return 2.0 * float(np.max(max_over_all)) / math.sqrt(3.0)


def _is_valid_packing(centers, inner_angle_deg, side, outer_angle_deg, min_sep=1e-8):
    """Validate that packing is valid: all hexagons inside outer, no overlaps."""
    # Check containment
    outer_v = _outer_hex_vertices(0.0, 0.0, side, outer_angle_deg)
    n = centers.shape[0]
    for i in range(n):
        hex_v = _hex_vertices(centers[i, 0], centers[i, 1], inner_angle_deg)
        for v in hex_v:
            if not _point_in_polygon(v, outer_v):
                return False
    # Check no overlap between inner hexagons
    for i in range(n):
        for j in range(i + 1, n):
            d = np.linalg.norm(centers[i] - centers[j])
            if d < 2.5:
                v1 = _hex_vertices(centers[i, 0], centers[i, 1], inner_angle_deg)
                v2 = _hex_vertices(centers[j, 0], centers[j, 1], inner_angle_deg)
                sep = _poly_separation(v1, v2)
                if sep < min_sep:
                    return False
    return True


def _optimize_outer_angle(centers, inner_angle_deg, time_limit):
    """
    Multi-stage optimization of outer hexagon angle for fixed inner configuration.
    Uses coarse-to-fine grid search followed by minimize_scalar polish.
    """
    best_side = float('inf')
    best_angle = 0.0

    # Stage 1: Coarse grid over [0, 60) degrees
    coarse_steps = np.arange(0, 60, 0.1)
    for a in coarse_steps:
        if time.time() > time_limit:
            break
        s = _vectorized_side(centers, inner_angle_deg, float(a))
        if s < best_side:
            best_side = s
            best_angle = float(a)

    # Stage 2: Medium grid around best
    if time.time() < time_limit:
        med_steps = np.arange(best_angle - 0.5, best_angle + 0.5, 0.01)
        for a in med_steps:
            if time.time() > time_limit:
                break
            s = _vectorized_side(centers, inner_angle_deg, float(a))
            if s < best_side:
                best_side = s
                best_angle = float(a)

    # Stage 3: Fine grid
    if time.time() < time_limit:
        fine_steps = np.arange(best_angle - 0.05, best_angle + 0.05, 0.001)
        for a in fine_steps:
            if time.time() > time_limit:
                break
            s = _vectorized_side(centers, inner_angle_deg, float(a))
            if s < best_side:
                best_side = s
                best_angle = float(a)

    # Stage 4: minimize_scalar polish
    if time.time() < time_limit:
        try:
            res = minimize_scalar(
                lambda a: _vectorized_side(centers, inner_angle_deg, float(a)),
                bounds=(best_angle - 0.01, best_angle + 0.01),
                method='bounded', options={'xatol': 1e-14, 'maxiter': 1000}
            )
            if res.success and res.fun < best_side:
                best_side = res.fun
                best_angle = res.x
        except Exception:
            pass

    # Stage 5: Ultra-tight final polish
    if time.time() < time_limit:
        try:
            res = minimize_scalar(
                lambda a: _vectorized_side(centers, inner_angle_deg, float(a)),
                bounds=(best_angle - 1e-10, best_angle + 1e-10),
                method='bounded', options={'xatol': 1e-16, 'maxiter': 500}
            )
            if res.success and res.fun < best_side:
                best_side = res.fun
                best_angle = res.x
        except Exception:
            pass

    return best_side, best_angle


def _optimize_inner_rotation(centers, outer_angle_deg, time_limit):
    """
    Optimize the common inner hexagon rotation angle for fixed outer angle.
    """
    best_side = float('inf')
    best_inner_angle = 0.0

    # Coarse grid
    coarse_steps = np.arange(0, 60, 0.5)
    for a in coarse_steps:
        if time.time() > time_limit:
            break
        s = _vectorized_side(centers, float(a), outer_angle_deg)
        if s < best_side:
            best_side = s
            best_inner_angle = float(a)

    # Medium grid
    if time.time() < time_limit:
        med_steps = np.arange(best_inner_angle - 0.5, best_inner_angle + 0.5, 0.05)
        for a in med_steps:
            if time.time() > time_limit:
                break
            s = _vectorized_side(centers, float(a), outer_angle_deg)
            if s < best_side:
                best_side = s
                best_inner_angle = float(a)

    # Fine grid
    if time.time() < time_limit:
        fine_steps = np.arange(best_inner_angle - 0.05, best_inner_angle + 0.05, 0.005)
        for a in fine_steps:
            if time.time() > time_limit:
                break
            s = _vectorized_side(centers, float(a), outer_angle_deg)
            if s < best_side:
                best_side = s
                best_inner_angle = float(a)

    # minimize_scalar polish
    if time.time() < time_limit:
        try:
            res = minimize_scalar(
                lambda a: _vectorized_side(centers, float(a), outer_angle_deg),
                bounds=(best_inner_angle - 0.01, best_inner_angle + 0.01),
                method='bounded', options={'xatol': 1e-14, 'maxiter': 1000}
            )
            if res.success and res.fun < best_side:
                best_side = res.fun
                best_inner_angle = res.x
        except Exception:
            pass

    # Ultra-tight
    if time.time() < time_limit:
        try:
            res = minimize_scalar(
                lambda a: _vectorized_side(centers, float(a), outer_angle_deg),
                bounds=(best_inner_angle - 1e-10, best_inner_angle + 1e-10),
                method='bounded', options={'xatol': 1e-16, 'maxiter': 500}
            )
            if res.success and res.fun < best_side:
                best_side = res.fun
                best_inner_angle = res.x
        except Exception:
            pass

    return best_side, best_inner_angle


def _joint_polish(centers, inner_angle_deg, outer_angle_deg, time_limit):
    """
    Joint coordinate-descent polish of inner and outer angles.
    """
    best_inner = inner_angle_deg
    best_outer = outer_angle_deg
    best_side = _vectorized_side(centers, inner_angle_deg, outer_angle_deg)

    for iteration in range(20):
        if time.time() > time_limit:
            break

        # Optimize outer angle with inner fixed
        s_new, a_new = _optimize_outer_angle(centers, best_inner, time_limit)
        if s_new < best_side:
            best_side = s_new
            best_outer = a_new

        # Optimize inner angle with outer fixed
        s_new2, a_new2 = _optimize_inner_rotation(centers, best_outer, time_limit)
        if s_new2 < best_side:
            best_side = s_new2
            best_inner = a_new2

        # Check convergence
        if abs(s_new - best_side) < 1e-15 and abs(s_new2 - best_side) < 1e-15:
            break

    return best_side, best_inner, best_outer


def construct():
    start_time = time.time()
    DEADLINE = 145.0
    safety_margin = 5.0
    n = 11

    # Build flower packing
    centers = _build_flower_centers()

    # Initial: all inner hexagons at 30 degrees, optimize outer angle
    inner_angle_init = 30.0
    best_side, best_outer_angle = _optimize_outer_angle(centers, inner_angle_init, DEADLINE - safety_margin - 30)

    # Validate
    if not _is_valid_packing(centers, inner_angle_init, best_side, best_outer_angle, min_sep=1e-8):
        # Fallback: use a known-good configuration
        inner_angle_init = 0.0
        best_side, best_outer_angle = _optimize_outer_angle(centers, inner_angle_init, DEADLINE - safety_margin - 30)

    # Phase 1: Optimize inner rotation angle
    if time.time() < DEADLINE - safety_margin - 10:
        s1, a1 = _optimize_inner_rotation(centers, best_outer_angle, DEADLINE - safety_margin - 10)
        if s1 < best_side:
            best_side = s1
            inner_angle_init = a1

    # Phase 2: Joint coordinate-descent polish
    if time.time() < DEADLINE - safety_margin - 5:
        s2, a_inner, a_outer = _joint_polish(centers, inner_angle_init, best_outer_angle, DEADLINE - safety_margin - 5)
        if s2 < best_side:
            best_side = s2
            inner_angle_init = a_inner
            best_outer_angle = a_outer

    # Phase 3: Final ultra-precise polish of both angles
    if time.time() < DEADLINE - safety_margin:
        # Final outer angle polish
        try:
            res = minimize_scalar(
                lambda a: _vectorized_side(centers, inner_angle_init, float(a)),
                bounds=(best_outer_angle - 1e-8, best_outer_angle + 1e-8),
                method='bounded', options={'xatol': 1e-16, 'maxiter': 200}
            )
            if res.success and res.fun < best_side:
                best_side = res.fun
                best_outer_angle = res.x
        except Exception:
            pass

        # Final inner angle polish
        try:
            res = minimize_scalar(
                lambda a: _vectorized_side(centers, float(a), best_outer_angle),
                bounds=(inner_angle_init - 1e-8, inner_angle_init + 1e-8),
                method='bounded', options={'xatol': 1e-16, 'maxiter': 200}
            )
            if res.success and res.fun < best_side:
                best_side = res.fun
                inner_angle_init = res.x
        except Exception:
            pass

    # Final validation
    final_side = _vectorized_side(centers, inner_angle_init, best_outer_angle)
    if not _is_valid_packing(centers, inner_angle_init, final_side, best_outer_angle, min_sep=1e-8):
        # Fallback to known-good
        inner_angle_init = 30.0
        final_side, best_outer_angle = _optimize_outer_angle(centers, inner_angle_init, DEADLINE - safety_margin)
        if not _is_valid_packing(centers, inner_angle_init, final_side, best_outer_angle, min_sep=1e-8):
            # Last resort: very conservative
            inner_angle_init = 0.0
            final_side = _vectorized_side(centers, inner_angle_init, 0.0)
            best_outer_angle = 0.0

    best_side = final_side

    # Build output
    inner_rows = []
    for i in range(n):
        inner_rows.append([float(centers[i, 0]), float(centers[i, 1]), float(inner_angle_init)])

    return inner_rows, (0.0, 0.0), float(best_side), float(best_outer_angle)
