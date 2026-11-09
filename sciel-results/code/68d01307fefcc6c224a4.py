"""Hexagon packing: minimize outer hexagon side length for 11 unit hexagons.

Strategy:
  1. Approximate each unit hexagon by its circumcircle (radius 1) and the
     outer hexagon by its circumscribed circle (radius = side length).
     This turns the problem into a fast circle-packing / force-directed
     optimization.
  2. Use a force-directed (repulsion + boundary) heuristic to find a compact
     cluster of 11 unit circles.
  3. Refine with a local search (random restarts + simulated annealing step).
  4. Return the best cluster with the outer side length equal to the
     circumscribed radius of the bounding circle (which equals the side
     length of a regular hexagon whose circumradius is that value).

The outer hexagon side length for a regular hexagon equals its circumradius,
so minimizing the max distance from center to any inner-hexagon vertex
approximates minimizing the outer side length.
"""

import math
import random
import time

N = 11
DEADLINE = 140.0  # safety margin before 150s kill

# Precompute unit hexagon vertices (circumradius 1)
_HEX_VERTS = [(math.cos(math.radians(60 * k)), math.sin(math.radians(60 * k))) for k in range(6)]

def _hex_extents(cx, cy, angle_deg):
    """Return (min_x, max_x, min_y, max_y) of a unit hexagon centered at (cx,cy) rotated."""
    a = math.radians(angle_deg)
    ca, sa = math.cos(a), math.sin(a)
    xs = []
    ys = []
    for (vx, vy) in _HEX_VERTS:
        rx = ca * vx - sa * vy
        ry = sa * vx + ca * vy
        xs.append(cx + rx)
        ys.append(cy + ry)
    return min(xs), max(xs), min(ys), max(ys)

def _max_vertex_dist(cx, cy, angle_deg):
    """Max distance from origin to any vertex of unit hex at (cx,cy) with rotation."""
    a = math.radians(angle_deg)
    ca, sa = math.cos(a), math.sin(a)
    best = 0.0
    for (vx, vy) in _HEX_VERTS:
        rx = ca * vx - sa * vy
        ry = sa * vx + ca * vy
        d = math.hypot(cx + rx, cy + ry)
        if d > best:
            best = d
    return best

def _outer_side_from_positions(positions):
    """Given list of (x, y, angle_deg), compute outer hexagon side length
    (circumradius) needed to contain all hexagons, centered at origin."""
    best = 0.0
    for (cx, cy, ang) in positions:
        d = _max_vertex_dist(cx, cy, ang)
        if d > best:
            best = d
    return best

def _force_layout(n, iters=4000, seed=0):
    """Force-directed packing of n unit circles. Returns list of (x, y)."""
    rng = random.Random(seed)
    # Initialize in a rough hexagonal cluster
    pts = []
    # seed positions in a small disk
    for i in range(n):
        r = math.sqrt(i / max(1, n)) * 2.5
        th = rng.uniform(0, 2 * math.pi)
        pts.append([r * math.cos(th), r * math.sin(th)])

    dt = 0.05
    for it in range(iters):
        # decay
        dt *= 0.999
        forces = [[0.0, 0.0] for _ in range(n)]
        # repulsion between circles (min distance 2.0)
        for i in range(n):
            for j in range(i + 1, n):
                dx = pts[i][0] - pts[j][0]
                dy = pts[i][1] - pts[j][1]
                d = math.hypot(dx, dy)
                if d < 1e-9:
                    dx = 0.01
                    dy = 0.0
                    d = 0.01
                if d < 2.0:
                    # strong repulsion when overlapping
                    f = (2.0 - d) * 10.0
                    fx = f * dx / d
                    fy = f * dy / d
                    forces[i][0] += fx
                    forces[i][1] += fy
                    forces[j][0] -= fx
                    forces[j][1] -= fy
        # attraction toward center (to keep cluster compact)
        for i in range(n):
            r = math.hypot(pts[i][0], pts[i][1])
            if r > 0.1:
                f = r * 0.02
                forces[i][0] -= f * pts[i][0] / r
                forces[i][1] -= f * pts[i][1] / r
        # integrate
        for i in range(n):
            pts[i][0] += forces[i][0] * dt
            pts[i][1] += forces[i][1] * dt
    return pts

def _overlap_penalty(positions):
    """Penalty for overlapping unit hexagons (approximated via circumcircles)."""
    pen = 0.0
    n = len(positions)
    for i in range(n):
        for j in range(i + 1, n):
            dx = positions[i][0] - positions[j][0]
            dy = positions[i][1] - positions[j][1]
            d = math.hypot(dx, dy)
            if d < 2.0:
                pen += (2.0 - d) ** 2
    return pen

def _optimize(positions, steps=3000, seed=0):
    """Local optimization: minimize outer side length subject to no overlap.
    Uses gradient-free random perturbations with annealing."""
    rng = random.Random(seed)
    best = list(positions)
    best_val = _outer_side_from_positions(best) + 100.0 * _overlap_penalty(best)
    temp = 0.5
    for s in range(steps):
        i = rng.randrange(len(positions))
        # perturb position and angle
        new = list(positions)
        new[i] = (
            new[i][0] + rng.gauss(0, 0.3),
            new[i][1] + rng.gauss(0, 0.3),
            new[i][2] + rng.gauss(0, 10.0),
        )
        # keep angle in [0,60)
        new[i] = (new[i][0], new[i][1], new[i][2] % 60.0)
        val = _outer_side_from_positions(new) + 100.0 * _overlap_penalty(new)
        if val < best_val or rng.random() < math.exp(-(val - best_val) / max(temp, 1e-9)):
            positions = new
            if val < best_val:
                best = list(new)
                best_val = val
        temp *= 0.999
    return best, best_val

def construct():
    start = time.time()
    rng = random.Random(42)

    best_positions = None
    best_val = float('inf')

    # Generate initial layouts via force-directed method with different seeds
    seeds = list(range(30))
    rng.shuffle(seeds)

    for seed in seeds:
        if time.time() - start > DEADLINE:
            break
        pts = _force_layout(N, iters=2000, seed=seed)
        # Convert to (x, y, angle) with zero angle initially
        positions = [(x, y, 0.0) for (x, y) in pts]
        # Center the cluster
        cx = sum(p[0] for p in positions) / N
        cy = sum(p[1] for p in positions) / N
        positions = [(p[0] - cx, p[1] - cy, p[2]) for p in positions]
        val = _outer_side_from_positions(positions) + 100.0 * _overlap_penalty(positions)
        if val < best_val:
            best_val = val
            best_positions = list(positions)

    # Refine the best layout with local optimization
    if best_positions is not None and time.time() - start < DEADLINE:
        refined, rval = _optimize(best_positions, steps=4000, seed=123)
        if rval < best_val:
            best_val = rval
            best_positions = refined

    # Additional refinement passes if time allows
    for _ in range(3):
        if time.time() - start > DEADLINE - 5:
            break
        refined, rval = _optimize(best_positions, steps=2000, seed=rng.randint(0, 100000))
        if rval < best_val:
            best_val = rval
            best_positions = refined

    # Final: compute outer side length (circumradius) centered at origin
    outer_side = _outer_side_from_positions(best_positions)

    inner = [[round(p[0], 6), round(p[1], 6), round(p[2], 6)] for p in best_positions]
    return inner, (0.0, 0.0), outer_side, 0.0
