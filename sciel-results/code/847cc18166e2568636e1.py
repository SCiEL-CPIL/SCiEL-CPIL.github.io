"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
import math

N = 11

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 120.0  # 120s limit, use 120s as safety margin

    def hex_vertices(cx, cy, angle_deg):
        """Return vertices of a regular hexagon with center (cx,cy), side 1, rotation angle_deg."""
        angle = math.radians(angle_deg)
        verts = []
        for i in range(6):
            a = angle + math.pi * i / 3.0
            verts.append((cx + math.cos(a), cy + math.sin(a)))
        return np.array(verts)

    def hex_containment(hex_verts, outer_center, outer_side, outer_angle_deg):
        """Check if hex_verts (array of 6 points) is inside the outer hexagon.
        Outer hexagon: center (outer_center), side length outer_side, rotation outer_angle_deg."""
        ox, oy = outer_center
        oa = math.radians(outer_angle_deg)
        # Vertices of outer hexagon
        outer_verts = []
        for i in range(6):
            a = oa + math.pi * i / 3.0
            outer_verts.append((ox + outer_side * math.cos(a), oy + outer_side * math.sin(a)))
        outer_verts = np.array(outer_verts)

        # Check if all points of inner hex are inside outer hex
        # Use point-in-polygon: for a convex polygon, check cross products
        # Outer hexagon is convex. For each edge, check that point is on the correct side.
        for px, py in hex_verts:
            inside = True
            for i in range(6):
                x1, y1 = outer_verts[i]
                x2, y2 = outer_verts[(i + 1) % 6]
                # Cross product to check side
                cross = (x2 - x1) * (py - y1) - (y2 - y1) * (px - x1)
                # All cross products should have the same sign (for CCW polygon)
                # Determine orientation of outer hex
                # For now, check absolute: if any cross product has wrong sign relative to others
                # Better: check all points against all edges consistently
                # Let's use a different approach: check if point is inside by testing against all edges
                # For a CCW polygon, point is inside if it's to the left of every edge
                # First determine if outer is CCW
                pass
            # Simpler: use numpy to check
        # Use a robust point-in-convex-polygon test
        # For convex polygon with vertices in order, point P is inside if
        # cross(edge_i, P - vertex_i) >= 0 for all i (for CCW) or <= 0 (for CW)
        # Determine orientation
        area2 = 0.0
        for i in range(6):
            x1, y1 = outer_verts[i]
            x2, y2 = outer_verts[(i + 1) % 6]
            area2 += x1 * y2 - x2 * y1
        # area2 > 0 means CCW
        ccw = area2 > 0

        for px, py in hex_verts:
            for i in range(6):
                x1, y1 = outer_verts[i]
                x2, y2 = outer_verts[(i + 1) % 6]
                cross = (x2 - x1) * (py - y1) - (y2 - y1) * (px - x1)
                if ccw and cross < -1e-10:
                    return False
                if not ccw and cross > 1e-10:
                    return False
        return True

    def min_side_for_hexagons(inner_positions, inner_angles, outer_center, outer_angle_deg):
        """Find minimum outer side length such that all inner hexagons fit inside."""
        # Binary search on side length
        lo = 1.0
        hi = 50.0
        for _ in range(60):
            mid = (lo + hi) / 2.0
            all_fit = True
            for (cx, cy), ang in zip(inner_positions, inner_angles):
                hv = hex_vertices(cx, cy, ang)
                if not hex_containment(hv, outer_center, mid, outer_angle_deg):
                    all_fit = False
                    break
            if all_fit:
                hi = mid
            else:
                lo = mid
        return hi

    def check_no_overlap(positions, angles, eps=1e-6):
        """Check that no two unit hexagons overlap."""
        n = len(positions)
        for i in range(n):
            for j in range(i + 1, n):
                # Minimum distance between centers of two non-overlapping unit hexagons
                # depends on their relative orientation. For simplicity, check vertex-vertex
                # and edge-edge intersection.
                # Quick reject: if center distance > 2, no overlap possible (max diameter = 2)
                dx = positions[i][0] - positions[j][0]
                dy = positions[i][1] - positions[j][1]
                if dx * dx + dy * dy > 4.0:
                    continue
                # Full check: get vertices
                vi = hex_vertices(*positions[i], angles[i])
                vj = hex_vertices(*positions[j], angles[j])
                # Check if any vertex of vi is inside vj or vice versa, or if edges intersect
                # Point in hexagon test
                def point_in_hex(p, hexv):
                    # hexv is 6x2 array, CCW or CW
                    px, py = p
                    area2 = 0.0
                    for k in range(6):
                        x1, y1 = hexv[k]
                        x2, y2 = hexv[(k + 1) % 6]
                        area2 += x1 * y2 - x2 * y1
                    ccw = area2 > 0
                    for k in range(6):
                        x1, y1 = hexv[k]
                        x2, y2 = hexv[(k + 1) % 6]
                        cross = (x2 - x1) * (py - y1) - (y2 - y1) * (px - x1)
                        if ccw and cross < -1e-9:
                            return False
                        if not ccw and cross > 1e-9:
                            return False
                    return True

                # Check if any vertex of i is in j
                for p in vi:
                    if point_in_hex(p, vj):
                        return False
                # Check if any vertex of j is in i
                for p in vj:
                    if point_in_hex(p, vi):
                        return False
                # Check edge intersections
                for a in range(6):
                    b = (a + 1) % 6
                    for c in range(6):
                        d = (c + 1) % 6
                        # Check if segment (vi[a],vi[b]) intersects (vj[c],vj[d])
                        x1,y1 = vi[a]; x2,y2 = vi[b]
                        x3,y3 = vj[c]; x4,y4 = vj[d]
                        den = (x2-x1)*(y4-y3)-(y2-y1)*(x4-x3)
                        if abs(den) < 1e-12:
                            continue
                        t = ((x3-x1)*(y4-y3)-(y3-y1)*(x4-x3))/den
                        s = ((x3-x1)*(y2-y1)-(y3-y1)*(x2-x1))/den
                        if 0 < t < 1 and 0 < s < 1:
                            return False
        return True

    # Strategy: Try compact cluster configurations
    # For N=11, try a central hexagon surrounded by 6, then add 4 more in gaps
    # Or a honeycomb-like cluster

    best_score = 40.0
    best_result = None

    # Pre-compute some candidate configurations
    # Configuration 1: Honeycomb cluster (1 center + 6 ring + 4 more)
    # Center at (0,0), angle 0
    # Ring of 6 at distance sqrt(3) in 6 directions
    # Then 4 more in the "bays"

    hex_r = math.sqrt(3)  # center-to-center distance for touching hexagons

    def make_cluster1():
        """1-6-4 cluster: center, 6 around, 4 in gaps"""
        positions = [(0.0, 0.0)]
        angles = [0.0]
        # 6 hexagons around center
        for i in range(6):
            a = math.pi * i / 3.0
            positions.append((hex_r * math.cos(a), hex_r * math.sin(a)))
            angles.append(0.0)
        # 4 more: place at larger radius in alternating gaps
        # Gaps are at angles pi/3*i + pi/6 (between the 6)
        # Distance for second ring touching both adjacent first-ring hexagons
        # For a regular hexagonal packing, the "bay" positions are at distance 2*hex_r
        # but only certain angles work. Let's place 4 at angles 0, 2pi/3, 4pi/3, and... 
        # Actually for 11 hexagons in a compact shape:
        # Layer 0: 1 hex at center
        # Layer 1: 6 hexagons (total 7)
        # Layer 2: 4 hexagons in the "indentations" 
        # The 4 additional go at angles pi/6, 5pi/6, 7pi/6, 11pi/6 at distance 2*hex_r
        for i in range(4):
            a = math.pi / 6 + math.pi * i  # pi/6, 7pi/6, 13pi/6=pi/6+2pi... no
        # Let me reconsider. For a hexagonal number pattern: 1, 7, 19...
        # 11 is between 7 and 19. So take the 7-hex cluster and add 4 more.
        # Add them at the 4 "corners" of the hexagonal shape, at distance 2*sqrt(3)
        # at angles 0, pi/3*2, pi/3*4, and... that's only 3 directions for a hex.
        # Let's just try: 4 additional at angles 0, 2pi/3, 4pi/3, pi at distance 2*hex_r
        # Actually, let's try a different approach: 3-4-3-1 or similar rows

    def make_cluster2():
        """Row-based: 3-4-4 or 3-3-4-1 etc."""
        # Horizontal rows of hexagons
        # Row spacing: for flat-top hexagons, vertical spacing = 1.5, horizontal = sqrt(3)
        # For pointy-top: vertical = sqrt(3), horizontal = 1.5
        # Use pointy-top orientation (angle 0 means pointy-top)
        # Actually, let's use flat-top: angle = 30 degrees gives flat-top
        # For flat-top unit hexagons: width = 2, height = sqrt(3)
        # Horizontal spacing = 1.5 (center to center in same row)
        # Vertical spacing = sqrt(3)
        # Offset rows shifted by 0.75
        positions = []
        angles = []
        # Row 0: 3 hexagons at y=0
        for i in range(3):
            positions.append((1.5 * i, 0.0))
            angles.append(30.0)
        # Row 1: 4 hexagons at y=sqrt(3), offset by 0.75
        for i in range(4):
            positions.append((0.75 + 1.5 * i, math.sqrt(3)))
            angles.append(30.0)
        # Row 2: 3 hexagons at y=2*sqrt(3)
        for i in range(3):
            positions.append((1.5 * i, 2 * math.sqrt(3)))
            angles.append(30.0)
        return positions, angles

    def make_cluster3():
        """Honeycomb 1-6 with 4 extra at 120-degree spaced positions"""
        positions = [(0.0, 0.0)]
        angles = [0.0]
        for i in range(6):
            a = math.pi * i / 3.0
            positions.append((hex_r * math.cos(a), hex_r * math.sin(a)))
            angles.append(0.0)
        # Add 4 more at distance 2*hex_r, angles 0, pi/2, pi, 3pi/2 -- no, that's square
        # For hexagonal symmetry, add at 0, 2pi/3, 4pi/3 (3 positions) plus one more
        # Let's add 3 at 0, 2pi/3, 4pi/3 and 1 at pi/2 (breaks symmetry)
        for i in range(3):
            a = 2 * math.pi * i / 3.0
            positions.append((2 * hex_r * math.cos(a), 2 * hex_r * math.sin(a)))
            angles.append(0.0)
        # 4th one: place at a gap position
        a = math.pi / 3
        positions.append((2 * hex_r * math.cos(a), 2 * hex_r * math.sin(a)))
        angles.append(0.0)
        return positions, angles

    def make_cluster4():
        """Tighter: use rotated hexagons to fit more compactly"""
        # Start with 1-6 cluster, rotate outer 6 by 30 degrees to nest
        positions = [(0.0, 0.0)]
        angles = [0.0]
        for i in range(6):
            a = math.pi * i / 3.0
            # With 30-degree rotation, the touching distance changes
            # For two hexagons with relative angle 30, min center distance is different
            # Just use hex_r for now
            positions.append((hex_r * math.cos(a), hex_r * math.sin(a)))
            angles.append(30.0)
        for i in range(4):
            a = math.pi / 3 + math.pi * i / 2  # spread out
            positions.append((2 * hex_r * math.cos(a), 2 * hex_r * math.sin(a)))
            angles.append(0.0)
        return positions, angles

    # Try various cluster configurations
    clusters = []
    try:
        p2, a2 = make_cluster2()
        if len(p2) == 11:
            clusters.append((p2, a2))
    except:
        pass
    try:
        p3, a3 = make_cluster3()
        if len(p3) == 11:
            clusters.append((p3, a3))
    except:
        pass
    try:
        p4, a4 = make_cluster4()
        if len(p4) == 11:
            clusters.append((p4, a4))
    except:
        pass

    # Also try the simple linear arrangement as fallback
    linear_positions = [(4.0 * k, 0.0) for k in range(-(N // 2), N - N // 2)]
    linear_angles = [0.0] * N
    clusters.append((linear_positions, linear_angles))

    for positions, angles in clusters:
        if time.time() > deadline - 30:
            break
        if len(positions) != N:
            continue
        # Check for overlaps
        if not check_no_overlap(positions, angles):
            continue
        # Find minimum outer side
        # Try different outer center and rotation
        # For now, center at centroid of inner hexagons, rotation 0
        cx = sum(p[0] for p in positions) / N
        cy = sum(p[1] for p in positions) / N
        side = min_side_for_hexagons(positions, angles, (cx, cy), 0.0)
        if side < best_score:
            best_score = side
            best_result = ([(p[0], p[1], a) for p, a in zip(positions, angles)], (cx, cy), side, 0.0)

    # Try to optimize: adjust outer rotation to reduce side length
    if best_result is not None:
        inner_pos, outer_c, side, outer_ang = best_result
        positions = [(p[0], p[1]) for p in inner_pos]
        angles = [p[2] for p in inner_pos]
        # Try several outer rotations
        for test_ang in [0, 30, 60, 15, 45, 90]:
            if time.time() > deadline - 20:
                break
            s = min_side_for_hexagons(positions, angles, outer_c, float(test_ang))
            if s < best_score:
                best_score = s
                best_result = (inner_pos, outer_c, s, float(test_ang))

    if best_result is None:
        # Fallback to seed
        inner = [[4.0 * k, 0.0, 0.0] for k in range(-(N // 2), N - N // 2)]
        return inner, (0.0, 0.0), 40.0, 0.0

    return best_result
# EVOLVE-BLOCK-END
