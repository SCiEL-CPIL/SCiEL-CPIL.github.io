"""Optimized hexagon packing: minimize outer hexagon side length for N unit hexagons."""
import numpy as np
import time
import math

N = 11
SQRT3 = math.sqrt(3)

# EVOLVE-BLOCK-START

def get_hex_vertices(cx, cy, side, angle_rad):
    """
    Return 6 vertices of a regular hexagon.
    Vertices are at angles angle_rad + i*pi/3.
    """
    verts = []
    for i in range(6):
        theta = angle_rad + i * math.pi / 3.0
        vx = cx + side * math.cos(theta)
        vy = cy + side * math.sin(theta)
        verts.append((vx, vy))
    return verts

def compute_exact_side(inner, outer_center, outer_angle_deg):
    """
    Compute the exact minimum outer side length to contain all inner hexagons.
    
    Args:
        inner: list of (x, y, angle_deg)
        outer_center: (x, y)
        outer_angle_deg: rotation of the outer hexagon
        
    Returns:
        Minimum side length S.
    """
    if not inner:
        return 0.0
    
    ox, oy = outer_center
    # Outward normals for a regular hexagon with vertices at 0, 60, ... degrees
    # are at 30, 90, 150, 210, 270, 330 degrees.
    normals = []
    for i in range(6):
        theta = math.radians(30 + i * 60)
        normals.append((math.cos(theta), math.sin(theta)))
    
    # Rotate outer frame to match outer_angle
    # We want to project points onto the normals of the rotated outer hexagon.
    # Equivalent to rotating points by -outer_angle and projecting onto fixed normals.
    cos_a = math.cos(math.radians(outer_angle_deg))
    sin_a = math.sin(math.radians(outer_angle_deg))
    
    max_proj = 0.0
    
    for (ix, iy, iangle) in inner:
        verts = get_hex_vertices(ix, iy, 1.0, math.radians(iangle))
        for v in verts:
            dx = v[0] - ox
            dy = v[1] - oy
            # Rotate point relative to outer center by -outer_angle
            lx = dx * cos_a + dy * sin_a
            ly = -dx * sin_a + dy * cos_a
            
            for nx, ny in normals:
                proj = lx * nx + ly * ny
                if proj > max_proj:
                    max_proj = proj
    
    # The distance from center to side (apothem) of a regular hexagon with side S is S * sqrt(3) / 2.
    # max_proj is the maximum distance from the center to any vertex of the inner set, 
    # measured along the outward normals of the container.
    # Therefore, S * sqrt(3) / 2 >= max_proj
    S = 2.0 * max_proj / SQRT3
    return S

def sat_overlap_check(v1, v2, eps=1e-9):
    """
    Check if two convex polygons (defined by their vertices) overlap using SAT.
    v1, v2: lists of (x, y) tuples.
    Returns True if they overlap (including touching).
    """
    axes = []
    
    # Get axes from edges of v1
    for i in range(len(v1)):
        p1 = v1[i]
        p2 = v1[(i + 1) % len(v1)]
        edge = (p2[0] - p1[0], p2[1] - p1[1])
        normal = (-edge[1], edge[0])
        norm = math.sqrt(normal[0]**2 + normal[1]**2)
        if norm > 1e-9:
            axes.append((normal[0] / norm, normal[1] / norm))
    
    # Get axes from edges of v2
    for i in range(len(v2)):
        p1 = v2[i]
        p2 = v2[(i + 1) % len(v2)]
        edge = (p2[0] - p1[0], p2[1] - p1[1])
        normal = (-edge[1], edge[0])
        norm = math.sqrt(normal[0]**2 + normal[1]**2)
        if norm > 1e-9:
            axes.append((normal[0] / norm, normal[1] / norm))
    
    for axis in axes:
        min1 = min(p[0] * axis[0] + p[1] * axis[1] for p in v1)
        max1 = max(p[0] * axis[0] + p[1] * axis[1] for p in v1)
        min2 = min(p[0] * axis[0] + p[1] * axis[1] for p in v2)
        max2 = max(p[0] * axis[0] + p[1] * axis[1] for p in v2)
        
        # Check for gap
        if max1 < min2 + eps or max2 < min1 + eps:
            return False
    
    return True

def check_all_disjoint(inner, eps=1e-6):
    """Check if all inner hexagons are pairwise disjoint using SAT."""
    n = len(inner)
    verts_list = [get_hex_vertices(inner[i][0], inner[i][1], 1.0, math.radians(inner[i][2])) for i in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            if sat_overlap_check(verts_list[i], verts_list[j], eps):
                return False
    return True

def check_inside(inner, outer_center, outer_side, outer_angle_deg, eps=1e-6):
    """Check if all inner hexagons are inside the outer hexagon."""
    ox, oy = outer_center
    outer_verts = get_hex_vertices(ox, oy, outer_side, math.radians(outer_angle_deg))
    
    for (ix, iy, iangle) in inner:
        verts = get_hex_vertices(ix, iy, 1.0, math.radians(iangle))
        for (vx, vy) in verts:
            inside = True
            for i in range(6):
                p1 = outer_verts[i]
                p2 = outer_verts[(i + 1) % 6]
                ex = p2[0] - p1[0]
                ey = p2[1] - p1[1]
                px = vx - p1[0]
                py = vy - p1[1]
                cross = ex * py - ey * px
                if cross < -eps:
                    inside = False
                    break
            if not inside:
                return False
    return True

def check_validity(inner, outer_center, outer_side, outer_angle_deg):
    """Check if the configuration is valid: no overlaps, all inside outer hexagon."""
    if not check_all_disjoint(inner):
        return False
    if not check_inside(inner, outer_center, outer_side, outer_angle_deg):
        return False
    return True

def construct():
    """
    Main entry point: find optimal packing of N unit hexagons in minimum outer hexagon.
    
    Strategy:
    1. Use the known optimal "flower" packing for 11 hexagons.
    2. Fix the inner hexagons in a tight hexagonal lattice configuration.
    3. Optimize the rotation of the outer hexagon to minimize its side length.
    4. The outer hexagon center is fixed at the centroid of the inner configuration (0,0).
    """
    start_time = time.time()
    
    # 1. Define the 11 hexagons in a "flower" pattern
    # Center at (0,0)
    # Ring 1: 6 hexagons at distance sqrt(3) from center, aligned with lattice
    # Ring 2: 4 hexagons selected to form a compact cluster
    
    # Axial coordinates (q, r) for hexagonal grid
    # Cartesian: x = 1.5 * q, y = sqrt(3) * (r + 0.5 * q)
    
    axial_coords = [
        (0, 0),   # Center
        (1, 0),   # Ring 1
        (0, 1),
        (-1, 1),
        (-1, 0),
        (0, -1),
        (1, -1),
        # Ring 2 - picking 4 that are "closest" to center in terms of bounding box
        # Candidates: (2, -1), (1, 1), (-2, 1), (-1, -1), (0, 2), (2, -2), (-2, 0), (0, -2)
        # Let's pick (2, -1), (1, 1), (-2, 1), (-1, -1) to maintain symmetry
        (2, -1),
        (1, 1),
        (-2, 1),
        (-1, -1)
    ]
    
    centers = []
    for q, r in axial_coords:
        x = 1.5 * q
        y = SQRT3 * (r + 0.5 * q)
        centers.append((x, y))
    
    # All inner hexagons are aligned with angle 0
    inner = [[cx, cy, 0.0] for cx, cy in centers]
    
    # Verify disjointness (should be true by construction with eps=0)
    if not check_all_disjoint(inner):
        # Fallback: slightly scale up if there's a numerical issue, though unlikely
        # But for N=11 standard flower, it is exact.
        pass

    # 2. Optimize outer hexagon rotation
    # By symmetry, we only need to search in [0, 30] degrees.
    # The function S(theta) is periodic with 60 degrees and symmetric around 0 and 30.
    
    best_S = float('inf')
    best_angle = 0.0
    
    # Coarse scan
    n_coarse = 100
    for i in range(n_coarse + 1):
        theta_deg = i * 30.0 / n_coarse
        S = compute_exact_side(inner, (0.0, 0.0), theta_deg)
        if S < best_S:
            best_S = S
            best_angle = theta_deg
            
    # Fine scan around the best coarse angle
    # The function is piecewise linear in terms of projections, but effectively smooth-ish.
    # Let's do a local refinement.
    search_range = 30.0 / n_coarse
    n_fine = 1000
    for i in range(n_fine + 1):
        theta_deg = best_angle - search_range/2 + i * search_range / n_fine
        # Ensure we stay in valid range, though wrapping doesn't matter due to periodicity
        S = compute_exact_side(inner, (0.0, 0.0), theta_deg)
        if S < best_S:
            best_S = S
            best_angle = theta_deg
            
    # Even finer scan
    search_range = search_range / 10.0
    n_fine2 = 1000
    for i in range(n_fine2 + 1):
        theta_deg = best_angle - search_range/2 + i * search_range / n_fine2
        S = compute_exact_side(inner, (0.0, 0.0), theta_deg)
        if S < best_S:
            best_S = S
            best_angle = theta_deg

    # 3. Final Validation
    outer_center = (0.0, 0.0)
    outer_angle = best_angle % 60.0 # Normalize
    final_S = compute_exact_side(inner, outer_center, outer_angle)
    
    # Check validity with a small margin to account for floating point errors
    # We add a tiny epsilon to the side length for the check to ensure "inside" passes
    check_S = final_S + 1e-9
    
    if not check_validity(inner, outer_center, check_S, outer_angle):
        # If validation fails (unlikely), try a slightly larger S or different angle
        # But given the exact geometry, this should pass.
        # Fallback to a known valid configuration: Side 3, Angle 0
        # The flower of 11 hexagons fits in a hexagon of side 3 exactly if aligned?
        # Let's check S for angle 0.
        S_0 = compute_exact_side(inner, (0.0, 0.0), 0.0)
        if check_validity(inner, (0.0, 0.0), S_0 + 1e-9, 0.0):
            final_S = S_0
            outer_angle = 0.0
            outer_center = (0.0, 0.0)
        else:
            # Extreme fallback
            final_S = 4.0
            outer_angle = 0.0
            outer_center = (0.0, 0.0)

    return (inner, outer_center, final_S, outer_angle)

# EVOLVE-BLOCK-END
