"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import math
import time
import numpy as np
from scipy.spatial import ConvexHull

N = 11

# EVOLVE-BLOCK-START

def hex_vertices(cx, cy, s, angle_deg):
    """Return the 6 vertices of a regular hexagon with center (cx,cy), side s, rotation angle_deg."""
    a = math.radians(angle_deg)
    verts = []
    for i in range(6):
        theta = a + i * math.pi / 3
        verts.append((cx + s * math.cos(theta), cy + s * math.sin(theta)))
    return verts

def hexagon_centroid_dist_to_edge(cx, cy, s, angle_deg):
    """
    For a regular hexagon with center (cx,cy), side length s, rotation angle_deg,
    compute the minimum distance from its vertices to the edges of the outer hexagon.
    Actually, we need to check if the inner hexagon is inside the outer hexagon.
    
    A point is inside a regular hexagon (centered at origin, side L, rotation 0) if:
    For each of the 6 edges, the point is on the correct side.
    
    Let's use a general approach: check if all 6 vertices of the inner hexagon are inside the outer hexagon.
    """
    pass

def point_in_hexagon(px, py, ocx, ocy, os, oangle_deg):
    """Check if point (px,py) is inside the regular hexagon centered at (ocx,ocy) with side os and rotation oangle_deg."""
    # Translate to origin
    dx = px - ocx
    dy = py - ocy
    # Rotate by -oangle_deg
    a = math.radians(-oangle_deg)
    rx = dx * math.cos(a) - dy * math.sin(a)
    ry = dx * math.sin(a) + dy * math.cos(a)
    
    # Now check if (rx, ry) is inside a regular hexagon centered at origin with side os, rotation 0.
    # A regular hexagon with flat top (rotation 0) has vertices at angles 0, 60, 120, 180, 240, 300.
    # The edges are between these vertices.
    # The hexagon can be described by 6 half-planes.
    # For a hexagon with "pointy top" or "flat top"?
    # Standard: if angle=0, vertex at (s, 0). So it's "pointy right".
    # The edges are:
    # 1. From (s,0) to (s/2, s*sqrt(3)/2): line x + sqrt(3)y <= s*sqrt(3) ? No.
    
    # Let's use the distance to each edge.
    # The 6 edges of a hexagon centered at origin, side s, with vertices at angles k*pi/3:
    # Edge i is between vertex i and vertex i+1.
    # The inward normal of edge i points to angle (i+0.5)*pi/3.
    # The distance from center to each edge is the apothem: a = s * sqrt(3) / 2.
    
    # So a point (x,y) is inside if for all i=0..5:
    # x * cos((i+0.5)*pi/3) + y * sin((i+0.5)*pi/3) <= a
    
    a = os * math.sqrt(3) / 2
    for i in range(6):
        theta = (i + 0.5) * math.pi / 3
        if rx * math.cos(theta) + ry * math.sin(theta) > a + 1e-9:
            return False
    return True

def hexagon_inside_hexagon(inner_verts, ocx, ocy, os, oangle_deg):
    """Check if all vertices of inner hexagon are inside the outer hexagon."""
    for (vx, vy) in inner_verts:
        if not point_in_hexagon(vx, vy, ocx, ocy, os, oangle_deg):
            return False
    return True

def min_outer_side_for_placement(inner_list, ocx, ocy, oangle_deg):
    """
    Given inner hexagons (unit side), find the minimum side length of an outer hexagon
    centered at (ocx, ocy) with rotation oangle_deg that contains all inner hexagons.
    
    This is a convex optimization problem but we can approximate by checking the required
    distance to each of the 6 supporting lines of the outer hexagon.
    
    For the outer hexagon to contain all inner hexagons, each inner vertex must satisfy:
    x * cos(theta_i) + y * sin(theta_i) <= ocx * cos(theta_i) + ocy * sin(theta_i) + a_i
    where theta_i = oangle_deg + (i+0.5)*pi/3, and a_i is the apothem of the outer hexagon.
    
    So a_i >= max over all inner vertices v of: (vx - ocx) * cos(theta_i) + (vy - ocy) * sin(theta_i)
    
    The outer hexagon is valid if all 6 a_i are equal to the apothem a = s * sqrt(3) / 2.
    So we need s * sqrt(3) / 2 >= max(a_0, ..., a_5).
    Thus s >= (2 / sqrt(3)) * max(a_i).
    """
    oangle_rad = math.radians(oangle_deg)
    max_a = 0.0
    for i in range(6):
        theta = oangle_rad + (i + 0.5) * math.pi / 3
        cos_t = math.cos(theta)
        sin_t = math.sin(theta)
        # For each inner hexagon, compute max projection of its vertices onto this direction
        for (cx, cy, iangle) in inner_list:
            iv = hex_vertices(cx, cy, 1.0, iangle)
            for (vx, vy) in iv:
                proj = (vx - ocx) * cos_t + (vy - ocy) * sin_t
                if proj > max_a:
                    max_a = proj
    return (2.0 / math.sqrt(3.0)) * max_a

def check_disjoint(inner_list):
    """Check if all pairs of inner hexagons are disjoint (no overlap)."""
    n = len(inner_list)
    for i in range(n):
        for j in range(i + 1, n):
            if hexagons_overlap(inner_list[i], inner_list[j]):
                return False
    return True

def hexagons_overlap(h1, h2):
    """Check if two unit hexagons overlap. They are disjoint if their centers are far enough apart."""
    # Two regular hexagons of side 1 are disjoint if the distance between centers is >= 2 * (sqrt(3)/2) * 2 ?
    # No. The minimum distance between centers of two disjoint unit hexagons depends on orientation.
    # The minimum distance is the distance between the closest points of the two convex polygons.
    # For efficiency, we can use a conservative bound: if center distance < 2 * inradius = sqrt(3), they might overlap.
    # Actually, the inradius of a unit hexagon is sqrt(3)/2. So if centers are less than 2*sqrt(3)/2 = sqrt(3) apart, they definitely overlap.
    # But they can be disjoint even if closer than that if rotated appropriately? No, the closest points are at least the distance between the two inscribed circles' boundaries.
    # Actually, the minimum distance between two disjoint convex sets is the distance between their closest points.
    # For two unit hexagons, the minimum center distance for disjointness is sqrt(3) (when they share an edge).
    # So if center distance < sqrt(3) - epsilon, they overlap.
    # But to be safe, let's check vertex-edge intersections or use a simpler check.
    
    # Simple check: if any vertex of h1 is inside h2 or any vertex of h2 is inside h1, they overlap.
    # Also, if edges intersect, they overlap.
    
    v1 = hex_vertices(h1[0], h1[1], 1.0, h1[2])
    v2 = hex_vertices(h2[0], h2[1], 1.0, h2[2])
    
    # Check if any vertex of v1 is inside h2
    for (vx, vy) in v1:
        if point_in_hexagon(vx, vy, h2[0], h2[1], 1.0, h2[2]):
            return True
    # Check if any vertex of v2 is inside h1
    for (vx, vy) in v2:
        if point_in_hexagon(vx, vy, h1[0], h1[1], 1.0, h1[2]):
            return True
    # Check for edge intersections (optional, for robustness)
    # For now, the vertex-in-polygon check is sufficient for most cases.
    return False

def construct():
    start_time = time.time()
    deadline = start_time + 120.0  # Safety margin
    
    # Start with a simple grid-like arrangement
    # For N=11, a hexagonal grid arrangement might work well.
    # Let's try a centered hexagonal arrangement.
    
    # Hexagonal grid coordinates:
    # Center at (0,0)
    # Ring 0: 1 hexagon
    # Ring 1: 6 hexagons
    # Ring 2: 12 hexagons
    # Total: 1 + 6 + 12 = 19, too many.
    # So for N=11, we need a custom arrangement.
    
    # Let's try a linear arrangement first as a baseline, then optimize.
    
    # Initial guess: linear arrangement
    inner = [[2.0 * k, 0.0, 0.0] for k in range(-(N // 2), N - N // 2)]
    
    # Compute initial outer side
    outer_side = min_outer_side_for_placement(inner, 0.0, 0.0, 0.0)
    
    # Try to optimize by adjusting positions
    # Use a simple local search: perturb each hexagon's position and angle
    
    best_inner = inner
    best_side = outer_side
    best_center = (0.0, 0.0)
    best_angle = 0.0
    
    # Try different outer hexagon rotations
    for oangle in [0.0, 30.0]:
        # Try to optimize inner positions
        current_inner = [[2.0 * k, 0.0, 0.0] for k in range(-(N // 2), N - N // 2)]
        
        # Local search
        for iteration in range(200):
            if time.time() > deadline:
                break
            
            # Try perturbing each hexagon
            improved = False
            for i in range(N):
                # Try small perturbations
                for dx in [-0.1, 0.0, 0.1]:
                    for dy in [-0.1, 0.0, 0.1]:
                        for da in [-5.0, 0.0, 5.0]:
                            new_inner = [list(h) for h in current_inner]
                            new_inner[i][0] += dx
                            new_inner[i][1] += dy
                            new_inner[i][2] += da
                            
                            # Check disjointness
                            if not check_disjoint(new_inner):
                                continue
                            
                            # Compute outer side
                            side = min_outer_side_for_placement(new_inner, 0.0, 0.0, oangle)
                            
                            if side < best_side:
                                best_side = side
                                best_inner = new_inner
                                best_center = (0.0, 0.0)
                                best_angle = oangle
                                current_inner = new_inner
                                improved = True
            
            if not improved:
                break
    
    # Return the best result
    return best_inner, best_center, best_side, best_angle

# EVOLVE-BLOCK-END
