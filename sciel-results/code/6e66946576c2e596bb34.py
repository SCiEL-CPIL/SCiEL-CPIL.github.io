# EVOLVE-BLOCK-START
import numpy as np
import time

_TIME_LIMIT = 620.0
_START_TIME = None


def _hex_vertices(cx, cy, s, a_deg):
    """Return 6 vertices of a regular hexagon (center cx,cy, side s, angle a_deg)."""
    a = np.deg2rad(a_deg)
    angles = a + np.deg2rad(60.0 * np.arange(6))
    return np.column_stack([cx + s * np.cos(angles), cy + s * np.sin(angles)])


def _hex_edge_normals(verts):
    """Return 6 edge normals (unit, outward) for a hexagon given its 6 vertices."""
    n = np.zeros((6, 2))
    for i in range(6):
        p1 = verts[i]
        p2 = verts[(i + 1) % 6]
        e = p2 - p1
        L = np.linalg.norm(e)
        # outward normal: rotate edge by -90 (for CCW vertices)
        n[i] = np.array([e[1] / L, -e[0] / L])
    return n


def _hexagons_separated(v1, n1, v2, n2, tol=1e-6):
    """Check if two hexagons (vertices v1,v2 and normals n1,n2) are separated."""
    # For each edge normal of hex1, project both hexagons; need gap > tol
    # Actually the condition: on at least one of the 12 normals, projected intervals apart by > tol
    for n in list(n1) + list(n2):
        proj1 = np.dot(v1, n)
        proj2 = np.dot(v2, n)
        min1, max1 = proj1.min(), proj1.max()
        min2, max2 = proj2.min(), proj2.max()
        gap = max(min1 - max2, min2 - max1)
        if gap > tol:
            return True
    return False


def _min_side_for_angle(inner_verts_list, outer_angle_deg):
    """Given inner hexagon vertices (list of 6x2 arrays) and outer hexagon angle,
    compute minimum side length S such that all inner vertices are inside outer hexagon centered at origin."""
    a = np.deg2rad(outer_angle_deg)
    angles = a + np.deg2rad(60.0 * np.arange(6))
    # Outer hexagon edges: for each edge, the constraint is that all points satisfy
    # dot(p, n_i) <= S * cos(30°) where n_i is the outward normal of edge i
    # For a regular hexagon with side S, the apothem (distance from center to edge) = S * cos(30°) = S*sqrt(3)/2
    max_needed = 0.0
    for vi in inner_verts_list:
        for i in range(6):
            n = np.array([np.cos(angles[i] + np.pi / 2), np.sin(angles[i] + np.pi / 2)])
            # outward normal for edge i: perpendicular to edge, pointing away from center
            # Edge i connects vertex i to vertex i+1. The outward normal points away from center.
            # For a regular hexagon, the outward normal of edge i is at angle (a + 60i + 30) deg? 
            # Let's compute: vertex i at angle a+60i, vertex i+1 at angle a+60(i+1).
            # Edge midpoint direction is at angle a + 60i + 30. Outward normal is same direction.
            n_out = np.array([np.cos(a + np.deg2rad(60.0 * i + 30.0)), np.sin(a + np.deg2rad(60.0 * i + 30.0))])
            proj = np.dot(vi, n_out)
            max_needed = max(max_needed, proj.max())
    # apothem = S * sqrt(3)/2, so S = 2*max_needed/sqrt(3)
    return 2.0 * max_needed / np.sqrt(3.0)


def _evaluate(inner_data, outer_angle_deg):
    """Compute the side length needed for given inner arrangement and outer angle."""
    inner_verts_list = []
    for (x, y, ang) in inner_data:
        v = _hex_vertices(x, y, 1.0, ang)
        inner_verts_list.append(v)
    return _min_side_for_angle(inner_verts_list, outer_angle_deg)


def hexagon_packing_11():
    global _START_TIME
    if _START_TIME is None:
        _START_TIME = time.time()

    # Base arrangement: center + 6 ring + 4 corner
    # Ring hexagons at distance 2*sqrt(3) from center, at angles 30,90,150,210,270,330
    # This makes them touch the center hexagon (distance between centers = 2*apothem = 2*sqrt(3))
    r_ring = 2.0 * np.sqrt(3.0)
    ring_angles = [30.0, 90.0, 150.0, 210.0, 270.0, 330.0]

    inner = np.zeros((11, 3))
    inner[0] = [0.0, 0.0, 0.0]  # center
    for i, ang in enumerate(ring_angles):
        inner[i + 1, 0] = r_ring * np.cos(np.deg2rad(ang))
        inner[i + 1, 1] = r_ring * np.sin(np.deg2rad(ang))
        inner[i + 1, 2] = 0.0

    # 4 corner hexagons: placed in the gaps. The gaps between adjacent ring hexagons
    # are at angles 0, 60, 120, 180, 240, 300. We pick 4 of them. For a hexagonal-ish shape,
    # pick the 4 that give the most symmetric/compact result. Try all 4-of-6 combos and pick best.
    gap_angles = [0.0, 60.0, 120.0, 180.0, 240.0, 300.0]
    r_corner = r_ring * np.cos(np.deg2rad(30.0))  # = 3.0
    # Actually the corner hexagon touching two ring hexagons: distance from center
    # The corner hexagon center should be equidistant from two adjacent ring hexagons.
    # Place at the gap angle, at a distance that makes it touch both ring hexagons.
    # Distance from corner to ring hexagon center should be 2*sqrt(3).
    # If corner at angle 0, distance d from center; ring hexagons at 30 and -30 (330), distance r_ring from center.
    # Distance between corner (d,0) and ring (r_ring*cos30, r_ring*sin30):
    # = sqrt((d - r_ring*cos30)^2 + (0 - r_ring*sin30)^2) = 2*sqrt(3)
    # r_ring = 2*sqrt(3), cos30=sqrt(3)/2, sin30=1/2
    # = sqrt((d - 3)^2 + (0 - sqrt(3))^2) = 2*sqrt(3)
    # (d-3)^2 + 3 = 12 => (d-3)^2 = 9 => d-3 = ±3 => d = 6 or 0
    # d=6 is the outer corner, d=0 is center (taken). So corner at distance 6? That seems far.
    # Wait, let me recompute. Actually for the corner to fit in the "bay", it should be closer.
    # Let me reconsider: maybe the 4 extra hexagons are not all in the 6 gaps.
    # Alternative: a 3-4-3 or similar arrangement.
    # Let me just use a known compact arrangement: the "hexagon of hexagons" with side 3 has 1+6+12=19, too many.
    # For 11: center(1) + ring(6) + 4 more. The 4 more can be placed at 4 of the 6 gap positions.
    # To minimize the enclosing hexagon, pick the 4 gaps that are most "spread out" to make it round.
    # Actually to make it compact, we want the 4 extra to fill in the shape. Let's try placing them at
    # angles 0, 120, 240 and one more... that's only 3. We need 4.
    # Let me try: place 4 extra at angles 0, 60, 180, 240? Or 0, 120, 240, and... 
    # Let's just try all C(6,4)=15 combinations and pick the best.

    best_side = np.inf
    best_inner = None
    best_outer_angle = 0.0

    from itertools import combinations
    for combo in combinations(range(6), 4):
        inner_trial = inner.copy()
        for j, gi in enumerate(combo):
            ang = gap_angles[gi]
            # Place corner hexagon at distance that makes it touch the two adjacent ring hexagons
            # From earlier calc, d=6. But let's verify this gives a valid packing.
            # Actually, let me reconsider. Maybe I should place them at a distance where they
            # just touch the ring hexagons. d=6 gives distance 2*sqrt(3) to ring hexagons. Good.
            d = 6.0
            inner_trial[7 + j, 0] = d * np.cos(np.deg2rad(ang))
            inner_trial[7 + j, 1] = d * np.sin(np.deg2rad(ang))
            inner_trial[7 + j, 2] = 0.0

        # Check separation of all pairs
        valid = True
        verts_list = []
        for k in range(11):
            v = _hex_vertices(inner_trial[k, 0], inner_trial[k, 1], 1.0, inner_trial[k, 2])
            n = _hex_edge_normals(v)
            verts_list.append((v, n))
        for i in range(11):
            for j in range(i + 1, 11):
                if not _hexagons_separated(verts_list[i][0], verts_list[i][1], verts_list[j][0], verts_list[j][1]):
                    valid = False
                    break
            if not valid:
                break

        if not valid:
            continue

        # Optimize outer angle
        best_local_side = np.inf
        best_local_angle = 0.0
        for ang in np.arange(0, 60, 0.5):
            s = _evaluate(inner_trial, ang)
            if s < best_local_side:
                best_local_side = s
                best_local_angle = ang

        if best_local_side < best_side:
            best_side = best_local_side
            best_inner = inner_trial.copy()
            best_outer_angle = best_local_angle

        # Time check
        if time.time() - _START_TIME > _TIME_LIMIT - 30:
            break

    if best_inner is None:
        # Fallback: use the base arrangement without corner hexagons (shouldn't happen)
        best_inner = inner.copy()
        best_side = _evaluate(best_inner, 0.0)
        best_outer_angle = 0.0

    # Fine-tune the outer angle
    if time.time() - _START_TIME < _TIME_LIMIT - 10:
        for ang in np.arange(best_outer_angle - 1.0, best_outer_angle + 1.0, 0.05):
            s = _evaluate(best_inner, ang)
            if s < best_side:
                best_side = s
                best_outer_angle = ang

    outer_hex_data = np.array([0.0, 0.0, best_outer_angle])
    return best_inner, outer_hex_data, float(best_side)


# EVOLVE-BLOCK-END
