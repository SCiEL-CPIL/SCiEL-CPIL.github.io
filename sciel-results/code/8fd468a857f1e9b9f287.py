"""Kinematic-closure solver: single global rotation + radial scale of a 12-hex honeycomb."""
import math

N = 12
SQ3 = math.sqrt(3.0)

# Honeycomb cluster: 1 central + 6 vertices + 6 edge-midpoints = 12 unit hexagons.
# Relative coordinates (unit hex side = 1).
_VERTS = [(2.0 * math.cos(math.radians(60.0 * k)), 2.0 * math.sin(math.radians(60.0 * k)))
          for k in range(6)]
_MIDS = [(SQ3 * math.cos(math.radians(30.0 + 60.0 * k)), SQ3 * math.sin(math.radians(30.0 + 60.0 * k)))
         for k in range(6)]
_CLUSTER = [(0.0, 0.0)] + _VERTS + _MIDS

# The 6 outward axis directions of a flat-top hexagon (angle 0).
_AXES = [(math.cos(math.radians(60.0 * k)), math.sin(math.radians(60.0 * k))) for k in range(6)]


def _side_for(theta_deg, t):
    """Outer hexagon side length needed to contain the scaled/rotated cluster."""
    th = math.radians(theta_deg)
    ct, st = math.cos(th), math.sin(th)
    r = 0.0
    for (x, y) in _CLUSTER:
        # rotate by theta, scale by t
        rx = t * (ct * x - st * y)
        ry = t * (st * x + ct * y)
        for (ax, ay) in _AXES:
            d = ax * rx + ay * ry
            if d > r:
                r = d
    # inradius r = s * sqrt(3)/2  =>  s = r * 2 / sqrt(3)
    return r * (2.0 / SQ3)


def _best_t(theta_deg, lo, hi):
    """Binary search the smallest scale t in [lo,hi] so the cluster fits in a unit hexagon."""
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        if _side_for(theta_deg, mid) <= 1.0:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def _s_of_theta(theta_deg):
    """Minimal outer side length for a fixed global rotation theta."""
    return 1.0 / _best_t(theta_deg, 0.0, 2.0)


def _ternary(f, a, b, iters=70):
    for _ in range(iters):
        m1 = a + (b - a) / 3.0
        m2 = b - (b - a) / 3.0
        if f(m1) < f(m2):
            b = m2
        else:
            a = m1
    return 0.5 * (a + b)


# EVOLVE-BLOCK-START
def construct():
    # 1) Find the optimal global rotation via ternary search over one 60-degree period.
    best_theta = _ternary(_s_of_theta, 0.0, 60.0, iters=60)

    # 2) Refine scale at the best rotation (binary search for tightest fit).
    best_t = _best_t(best_theta, 0.0, 2.0)

    # 3) Place the 12 inner hexagons (unit side) at the scaled/rotated cluster positions.
    th = math.radians(best_theta)
    ct, st = math.cos(th), math.sin(th)
    inner = []
    for (x, y) in _CLUSTER:
        rx = best_t * (ct * x - st * y)
        ry = best_t * (st * x + ct * y)
        inner.append([rx, ry, 0.0])

    outer_side = _side_for(best_theta, best_t)
    return inner, (0.0, 0.0), outer_side, 0.0
# EVOLVE-BLOCK-END
