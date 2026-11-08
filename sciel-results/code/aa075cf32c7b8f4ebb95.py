import time
import math
import numpy as np

N = 11

# EVOLVE-BLOCK-START

def _hex_vertices(cx, cy, s, angle_deg):
    """Vertices of a regular hexagon (flat-top at angle 0)."""
    a = math.radians(angle_deg)
    ca, sa = math.cos(a), math.sin(a)
    verts = []
    for i in range(6):
        th = math.radians(60 * i)
        px, py = s * math.cos(th), s * math.sin(th)
        verts.append((cx + ca * px - sa * py, cy + sa * px + ca * py))
    return verts


def _point_in_hex(px, py, cx, cy, s, angle_deg, eps=1e-9):
    """Check if point (px,py) is inside or on the hexagon."""
    a = math.radians(angle_deg)
    ca, sa = math.cos(a), math.sin(a)
    # transform point to hexagon's local frame
    dx, dy = px - cx, py - cy
    lx = ca * dx + sa * dy
    ly = -sa * dx + ca * dy
    # regular hexagon with circumradius s, flat-top orientation
    # |ly| <= s/2  and  |lx| <= s*sqrt(3)/2 - |ly|/sqrt(3)  ... actually:
    # For a regular hexagon with vertices at angles 0,60,...,300:
    # the six half-plane constraints are |lx*cos(60k) + ly*sin(60k)| <= s/2 for k=0..5
    # Wait, that's for inradius. Let me use the standard: inradius = s*sqrt(3)/2
    # Actually for a regular hexagon with side s, the apothem (inradius) = s*sqrt(3)/2
    # The six supporting lines are at distance s*sqrt(3)/2 from center.
    # Constraint: lx*cos(60k) + ly*sin(60k) <= s*sqrt(3)/2  for all k
    apothem = s * math.sqrt(3) / 2
    for k in range(6):
        th = math.radians(60 * k)
        if lx * math.cos(th) + ly * math.sin(th) > apothem + eps:
            return False
    return True


def hexagons_disjoint(inner, center, side, angle, tol=1e-6):
    """Check that all inner hexagons are disjoint and inside the outer hexagon."""
    n = len(inner)
    # Check outer containment
    for (ix, iy, ia) in inner:
        # sample center and vertices of inner hex
        pts = [(ix, iy)]
        for (vx, vy) in _hex_vertices(ix, iy, 1.0, ia):
            pts.append((vx, vy))
        for (px, py) in pts:
            if not _point_in_hex(px, py, center[0], center[1], side, angle, eps=tol):
                return False
    # Check pairwise disjointness
    for i in range(n):
        for j in range(i + 1, n):
            xi, yi, ai = inner[i]
            xj, yj, aj = inner[j]
            # Check if any vertex of i is inside j or vice versa
            for (px, py) in _hex_vertices(xi, yi, 1.0, ai):
                if _point_in_hex(px, py, xj, yj, 1.0, aj, eps=-tol):
                    return False
            for (px, py) in _hex_vertices(xj, yj, 1.0, aj):
                if _point_in_hex(px, py, xi, yi, 1.0, ai, eps=-tol):
                    return False
    return True


def _initial_positions():
    """Hexagonal lattice: center + 6 neighbors + 4 extra."""
    r = math.sqrt(3)  # distance between adjacent hexagon centers for side-1 hexagons
    positions = []
    # center
    positions.append([0.0, 0.0, 0.0])
    # 6 neighbors
    for k in range(6):
        th = math.radians(60 * k)
        positions.append([r * math.cos(th), r * math.sin(th), 0.0])
    # 4 more: extend the lattice
    # second ring partial: pick 4 more positions
    # distance for second ring = 2*r
    r2 = 2 * r
    for k in range(6):
        th = math.radians(60 * k)
        positions.append([r2 * math.cos(th), r2 * math.sin(th), 0.0])
        if len(positions) >= N:
            break
    return positions[:N]


def _minkowski_push(pos_i, pos_j, n_iter=50):
    """Compute a push vector to separate hexagon i from hexagon j.
    Uses the Minkowski difference approach: find the minimum translation
    of hex i so it no longer overlaps hex j."""
    xi, yi, ai = pos_i
    xj, yj, aj = pos_j
    dx, dy = xj - xi, yj - yi
    dist = math.sqrt(dx * dx + dy * dy)
    if dist < 1e-12:
        return (0.1, 0.1)  # random small push
    # The Minkowski sum of two unit hexagons has "radius" up to 2*sqrt(3)/2*... 
    # For simplicity, use a conservative overlap threshold
    # Two unit hexagons (side 1) are disjoint if center distance > 2*apothem = sqrt(3)*... 
    # Actually for same orientation, min center distance for disjointness = 2 * apothem = sqrt(3)
    # For different orientations it's more complex. Use a conservative threshold.
    threshold = 1.9  # conservative
    if dist < threshold:
        # push apart along the line connecting centers
        push = (threshold - dist) / dist
        return (push * dx / 2, push * dy / 2)
    return (0.0, 0.0)


def construct():
    t0 = time.time()
    DEADLINE = 140.0  # safety margin before 150s kill

    # Initial configuration
    positions = _initial_positions()
    ox, oy = 0.0, 0.0
    oangle = 0.0
    # Initial outer side: compute bounding side needed
    # For a hexagonal cluster of radius ~2*sqrt(3), the outer hex side is roughly 2*sqrt(3)+1
    outer_side = 5.0

    # Verify initial
    if not hexagons_disjoint(positions, (ox, oy), outer_side, oangle):
        outer_side = 7.0

    best_positions = [p[:] for p in positions]
    best_side = outer_side
    best_valid = True

    # Force-directed relaxation with shrinking
    n_steps = 2000
    step_size = 0.05
    shrink_per_step = 0.001

    for step in range(n_steps):
        if time.time() - t0 > DEADLINE:
            break

        # 1. Repulsion: push overlapping pairs apart
        for i in range(N):
            for j in range(i + 1, N):
                px, py = _minkowski_push(positions[i], positions[j])
                positions[i][0] -= px
                positions[i][1] -= py
                positions[j][0] += px
                positions[j][1] += py

        # 2. Attraction to outer boundary: pull hexagons that are outside back in
        for i in range(N):
            xi, yi, ai = positions[i]
            # Check if hexagon i protrudes the outer hexagon
            # Simple check: is the center too far from outer center?
            dx, dy = xi - ox, yi - oy
            dist_c = math.sqrt(dx * dx + dy * dy)
            # The max distance from center to a vertex of the outer hex is outer_side (circumradius)
            # The inner hex has circumradius 1, so center should be at most outer_side - 1 from center
            max_dist = outer_side - 1.0 - 0.01
            if dist_c > max_dist:
                # Pull back toward center
                pull = (dist_c - max_dist) / dist_c * 0.5
                positions[i][0] -= pull * dx
                positions[i][1] -= pull * dy

        # 3. Small rotation perturbation (gradient-free)
        # Skip for speed; rotations stay at 0 (hexagonal lattice is optimal for same-orientation)

        # 4. Shrink outer side
        outer_side -= shrink_per_step
        if outer_side < 2.0:
            outer_side = 2.0

        # 5. Periodically check validity and record best
        if step % 50 == 0 or step == n_steps - 1:
            if hexagons_disjoint(positions, (ox, oy), outer_side, oangle):
                if outer_side < best_side:
                    best_side = outer_side
                    best_positions = [p[:] for p in positions]
                    best_valid = True
            else:
                # Try to recover: expand slightly
                outer_side += 0.1

    # Final: if best is valid, return it; otherwise try a safe fallback
    if best_valid:
        return best_positions, (ox, oy), best_side, oangle

    # Fallback: return a known-valid configuration (larger outer hex)
    fallback_positions = _initial_positions()
    for trial_side in [6.0, 7.0, 8.0, 10.0, 15.0, 20.0]:
        if hexagons_disjoint(fallback_positions, (ox, oy), trial_side, oangle):
            return fallback_positions, (ox, oy), trial_side, oangle

    # Last resort: very large outer hex
    return fallback_positions, (ox, oy), 40.0, oangle

# EVOLVE-BLOCK-END
