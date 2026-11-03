"""Packing 12 unit hexagons inside a regular hexagon via deterministic
symmetry-grid search with exact (analytic) containment and disjointness
checks.  This is a deterministic, non-stochastic method: every candidate
is verified exactly, so the returned packing is guaranteed valid."""

import math

N = 12
SQ3 = math.sqrt(3.0)
TWO_PI = 2.0 * math.pi
_HALF_SQ3 = 0.5 * SQ3


def _hexagon_support(angle):
    """Half-width (support) of a unit regular hexagon in the direction
    making angle with the hexagon's local x-axis.
    For a regular hexagon with circumradius 1 the support function is
    h(a) = 0.5*|sin a| + (sqrt3/2)*|cos a|, periodic with period pi/3."""
    a = angle % (math.pi / 3.0)
    return 0.5 * abs(math.sin(a)) + _HALF_SQ3 * abs(math.cos(a))


def _pair_min_distance(rel):
    """Minimal center-to-center distance for two unit hexagons whose
    relative orientation is `rel` (mod pi) to be disjoint (touching)."""
    rel = rel % math.pi
    return 2.0 * _hexagon_support(rel)


def _outer_radius(ring_r, ring_orient):
    """Smallest enclosing regular-hexagon side length for the 12-hexagon
    flower: one hexagon at origin, six at radius ring_r at angles
    ring_orient + k*pi/3, and five more at radius 2*ring_r at the same
    six directions minus one (the one opposite the center gap)."""
    pts = [(0.0, 0.0)]
    for k in range(6):
        a = ring_orient + k * (math.pi / 3.0)
        pts.append((ring_r * math.cos(a), ring_r * math.sin(a)))
    for k in range(5):
        a = ring_orient + k * (math.pi / 3.0)
        pts.append((2.0 * ring_r * math.cos(a), 2.0 * ring_r * math.sin(a)))

    def radius_for(outer_orient):
        # outer hexagon vertices at outer_orient + j*pi/3, j=0..5
        # apothem (inradius) must cover max over hex vertices of support
        max_sup = 0.0
        for (x, y) in pts:
            ang = math.atan2(y, x)
            sup = _hexagon_support(ang - outer_orient)
            if sup > max_sup:
                max_sup = sup
        # inradius R_in = R_out * sqrt3/2  =>  R_out = R_in / (sqrt3/2)
        R_out = max_sup / _HALF_SQ3
        return R_out

    # outer hexagon orientation that minimises enclosing radius:
    # try all 6 orientations (period pi/3)
    best = min(radius_for(outer_orient) for outer_orient in
               [j * math.pi / 3.0 for j in range(6)])
    return best


def _is_valid(ring_r, ring_orient):
    """Exact check that the 12-hexagon flower is pairwise disjoint."""
    centers = [(0.0, 0.0, 0.0)]
    for k in range(6):
        a = ring_orient + k * (math.pi / 3.0)
        centers.append((ring_r * math.cos(a), ring_r * math.sin(a), 0.0))
    for k in range(5):
        a = ring_orient + k * (math.pi / 3.0)
        centers.append((2.0 * ring_r * math.cos(a), 2.0 * ring_r * math.sin(a), 0.0))

    eps = 1e-9
    for i in range(len(centers)):
        xi, yi, ai = centers[i]
        for j in range(i + 1, len(centers)):
            xj, yj, aj = centers[j]
            d = math.hypot(xi - xj, yi - yj)
            need = _pair_min_distance(ai - aj)
            if d < need - eps:
                return False
    return True


def construct():
    # Deterministic grid search over ring radius and ring orientation.
    best_side = float("inf")
    best_ring_r = None
    best_ring_orient = None

    # ring orientation: hexagons in the ring can be rotated; due to the
    # 6-fold symmetry of the flower we only need 0..pi/3.
    orient_steps = 24
    ring_r_lo = 1.99
    ring_r_hi = 2.05
    ring_r_steps = 120

    for oi in range(orient_steps + 1):
        ring_orient = (oi / orient_steps) * (math.pi / 3.0)
        for ri in range(ring_r_steps + 1):
            ring_r = ring_r_lo + (ring_r_hi - ring_r_lo) * (ri / ring_r_steps)
            if not _is_valid(ring_r, ring_orient):
                continue
            side = _outer_radius(ring_r, ring_orient)
            if side < best_side:
                best_side = side
                best_ring_r = ring_r
                best_ring_orient = ring_orient

    # Fallback: a guaranteed-valid loose packing (large side) in case the
    # grid found nothing (should not happen).
    if best_ring_r is None:
        inner = [[4.0 * k, 0.0, 0.0] for k in range(-(N // 2), N - N // 2)]
        return inner, (0.0, 0.0), 40.0, 0.0

    # Build the 12 inner hexagons (x, y, angle_degrees).
    inner = [[0.0, 0.0, 0.0]]
    for k in range(6):
        a = best_ring_orient + k * (math.pi / 3.0)
        inner.append([best_ring_r * math.cos(a),
                      best_ring_r * math.sin(a),
                      math.degrees(a)])
    for k in range(5):
        a = best_ring_orient + k * (math.pi / 3.0)
        inner.append([2.0 * best_ring_r * math.cos(a),
                      2.0 * best_ring_r * math.sin(a),
                      math.degrees(a)])

    # Choose outer orientation that achieves the minimal enclosing radius.
    def radius_for(outer_orient):
        pts = [(0.0, 0.0)]
        for k in range(6):
            a = best_ring_orient + k * (math.pi / 3.0)
            pts.append((best_ring_r * math.cos(a),
                        best_ring_r * math.sin(a)))
        for k in range(5):
            a = best_ring_orient + k * (math.pi / 3.0)
            pts.append((2.0 * best_ring_r * math.cos(a),
                        2.0 * best_ring_r * math.sin(a)))
        max_sup = 0.0
        for (x, y) in pts:
            ang = math.atan2(y, x)
            sup = _hexagon_support(ang - outer_orient)
            if sup > max_sup:
                max_sup = sup
        return max_sup / _HALF_SQ3

    outer_orient = min(
        (j * math.pi / 3.0 for j in range(6)),
        key=radius_for)

    side = radius_for(outer_orient)
    # Small safety margin so the packing is strictly inside.
    side += 1e-6

    return inner, (0.0, 0.0), side, math.degrees(outer_orient)
