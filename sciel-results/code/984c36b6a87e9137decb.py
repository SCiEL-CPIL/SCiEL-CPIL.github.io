"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
from itertools import product
import time

N = 11

# EVOLVE-BLOCK-START
def _hex_vertices(cx, cy, side, angle_deg):
    """Return 6 vertices of a regular hexagon centered at (cx,cy) with given side and rotation."""
    a = np.radians(angle_deg)
    r = side  # circumradius = side for regular hexagon
    angles = a + np.arange(6) * np.pi / 3
    return np.column_stack([cx + r * np.cos(angles), cy + r * np.sin(angles)])

def _point_in_hex(px, py, hx, hy, side, angle_deg):
    """Check if point (px,py) is inside hexagon centered at (hx,hy) with side and rotation."""
    # Transform point to hexagon's local frame
    dx, dy = px - hx, py - hy
    a = -np.radians(angle_deg)
    ca, sa = np.cos(a), np.sin(a)
    lx, ly = ca * dx - sa * dy, sa * dx + ca * dy
    # In local frame, hexagon has vertices at angles 0, 60, ... with radius = side
    # The hexagon is the intersection of 6 half-planes
    # For a regular hexagon with side s, the distance from center to each side (apothem) = s * sqrt(3)/2
    # The half-planes are: |projection onto direction i| <= apothem
    apothem = side * np.sqrt(3) / 2
    for i in range(6):
        theta = i * np.pi / 3
        # Normal direction
        nx, ny = np.cos(theta), np.sin(theta)
        proj = lx * nx + ly * ny
        if proj > apothem:
            return False
    return True

def _hex_hex_overlap_dist(d, angle_diff_deg):
    """
    Return the minimum center-to-center distance for two unit hexagons 
    (one rotated by angle_diff_deg relative to other) to be non-overlapping.
    Actually, we need to check if two unit hexagons at given positions/angles overlap.
    """
    pass

def _check_disjoint(inner):
    """Check if all inner hexagons are pairwise disjoint (non-overlapping interiors)."""
    n = len(inner)
    for i in range(n):
        for j in range(i + 1, n):
            x1, y1, a1 = inner[i]
            x2, y2, a2 = inner[j]
            # Check if the two unit hexagons overlap
            # Sample points on the boundary of each and check
            # More robust: check if any vertex of one is inside the other, and vice versa
            v1 = _hex_vertices(x1, y1, 1.0, a1)
            v2 = _hex_vertices(x2, y2, 1.0, a2)
            # Check if any vertex of hex1 is inside hex2
            for px, py in v1:
                if _point_in_hex(px, py, x2, y2, 1.0, a2):
                    return False
            # Check if any vertex of hex2 is inside hex1
            for px, py in v2:
                if _point_in_hex(px, py, x1, y1, 1.0, a1):
                    return False
            # Also check if edges cross - for now, vertex-in-polygon is a good heuristic
            # For unit hexagons, if no vertex is inside the other, they're likely disjoint
            # but edge crossings can happen. Let's add a midpoint check.
            for i2 in range(6):
                mx = (v1[i2, 0] + v1[(i2+1)%6, 0]) / 2
                my = (v1[i2, 1] + v1[(i2+1)%6, 1]) / 2
                if _point_in_hex(mx, my, x2, y2, 1.0, a2):
                    return False
            for i2 in range(6):
                mx = (v2[i2, 0] + v2[(i2+1)%6, 0]) / 2
                my = (v2[i2, 1] + v2[(i2+1)%6, 1]) / 2
                if _point_in_hex(mx, my, x1, y1, 1.0, a1):
                    return False
    return True

def _min_enclosing_hex_side(inner, center_x, center_y, outer_angle_deg):
    """
    Compute the minimum side length of a regular hexagon centered at (center_x, center_y)
    with rotation outer_angle_deg that contains all inner hexagons.
    """
    apothem_needed = 0.0
    n = len(inner)
    for x, y, a in inner:
        v = _hex_vertices(x, y, 1.0, a)
        for px, py in v:
            # Distance from (center_x, center_y) in the direction of each of the 6 normals
            dx, dy = px - center_x, py - center_y
            for i in range(6):
                theta = np.radians(outer_angle_deg) + i * np.pi / 3
                nx, ny = np.cos(theta), np.sin(theta)
                proj = dx * nx + dy * ny
                if proj > apothem_needed:
                    apothem_needed = proj
    # side = apothem / (sqrt(3)/2)
    return apothem_needed / (np.sqrt(3) / 2)

def _try_packing(lattice_positions, outer_angle_deg, center_x=0.0, center_y=0.0):
    """
    Given a list of (x, y) positions for the 11 hexagons (all with angle 0),
    compute the enclosing hexagon side length.
    """
    inner = [[x, y, 0.0] for x, y in lattice_positions]
    # Check disjointness
    if not _check_disjoint(inner):
        return None
    side = _min_enclosing_hex_side(inner, center_x, center_y, outer_angle_deg)
    return side, inner, (center_x, center_y), outer_angle_deg

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
    best_side = float('inf')
    best_result = None
    
    # Hexagonal lattice basis vectors
    # Axial coordinates (q, r) to pixel:
    # x = size * (3/2 * q)
    # y = size * (sqrt(3)/2 * q + sqrt(3) * r)
    # For unit hexagons touching, center distance = 2 * apothem = 2 * sqrt(3)/2 * 2 = ... 
    # Actually for unit side hexagons, the distance between centers of adjacent hexagons = 2 * apothem = sqrt(3) * 2? 
    # No: apothem = side * sqrt(3)/2 = sqrt(3)/2 for side=1.
    # Adjacent hexagons touch when center distance = 2 * apothem = sqrt(3).
    # Wait, that's not right either. Let me reconsider.
    # For two regular hexagons of side s to touch (share an edge), the center-to-center distance is 2 * apothem = 2 * s * sqrt(3)/2 = s * sqrt(3).
    # So for s=1, distance = sqrt(3) ≈ 1.732.
    
    # Actually, I think I'm confusing myself. Let me use a simpler approach:
    # Place hexagons on a hex grid where adjacent centers are at distance 2*sqrt(3)/2 * 2 = ... 
    # Let me just use the standard hex grid spacing.
    
    # For unit hexagons (side=1), the distance between centers of hexagons sharing an edge is 2 * apothem = sqrt(3).
    # But they also need to be non-overlapping. If centers are at distance >= sqrt(3), they don't overlap (for aligned hexagons).
    
    # Let me generate compact clusters of 11 hexagons on a hex grid.
    # The hex grid positions can be represented in axial coordinates.
    # Position in Cartesian: x = 1.5 * q, y = sqrt(3)/2 * q + sqrt(3) * r (for pointy-top)
    # Or for flat-top: x = 3/2 * q, y = sqrt(3) * (r + q/2)
    
    # Let me use the convention: 
    # q, r are axial coordinates. 
    # Cartesian: x = 3/2 * q, y = sqrt(3) * (r + q/2)
    # This gives adjacent hexagons at distance sqrt(3) apart.
    
    sqrt3 = np.sqrt(3)
    
    def axial_to_cart(q, r):
        return (1.5 * q, sqrt3 * (r + q / 2.0))
    
    # Generate all hex grid positions within a certain radius
    def hex_neighbors(q, r):
        return [(q+1, r), (q-1, r), (q, r+1), (q, r-1), (q+1, r-1), (q-1, r+1)]
    
    def hex_distance(q1, r1, q2, r2):
        dq = q1 - q2
        dr = r1 - r2
        return (abs(dq) + abs(dr) + abs(dq + dr)) // 2
    
    # Generate candidate clusters: BFS from origin, pick 11 positions
    # Start with the 7 hexagons in a "flower" pattern (center + 6 neighbors)
    # Then add 4 more from the next ring
    
    # Ring 0: (0,0)
    # Ring 1: 6 positions at distance 1
    # Ring 2: 12 positions at distance 2
    
    ring0 = [(0, 0)]
    ring1 = [(1,0), (-1,0), (0,1), (0,-1), (1,-1), (-1,1)]
    ring2 = [(2,0), (-2,0), (0,2), (0,-2), (2,-2), (-2,2), (1,1), (-1,-1), (2,-1), (-2,1), (1,-2), (-1,2)]
    
    # Base cluster: ring0 + ring1 = 7 hexagons
    base = set(ring0 + ring1)
    
    # We need 4 more from ring2. Try all combinations of 4 from ring2.
    # There are C(12,4) = 495 combinations.
    
    import itertools
    
    ring2_list = list(ring2)
    
    # For each combination, check if the resulting 11 hexagons are all pairwise non-overlapping
    # and compute the enclosing hexagon side length for various outer rotations.
    
    outer_angles = [0.0, 30.0, 60.0, 90.0, 120.0, 150.0]
    
    for combo in itertools.combinations(range(len(ring2_list)), 4):
        if time.time() > deadline:
            break
        
        cluster = set(base)
        for idx in combo:
            cluster.add(ring2_list[idx])
        
        if len(cluster) != 11:
            continue
        
        positions = [axial_to_cart(q, r) for q, r in cluster]
        
        # Check if all pairs are at distance >= sqrt(3) - epsilon
        # For aligned unit hexagons, minimum non-overlapping distance is sqrt(3)
        valid = True
        for i in range(11):
            for j in range(i+1, 11):
                dx = positions[i][0] - positions[j][0]
                dy = positions[i][1] - positions[j][1]
                dist = np.sqrt(dx*dx + dy*dy)
                if dist < sqrt3 - 1e-6:
                    valid = False
                    break
            if not valid:
                break
        
        if not valid:
            continue
        
        # Try different outer hexagon rotations
        for oa in outer_angles:
            if time.time() > deadline:
                break
            
            # The center of the enclosing hexagon should be at the centroid of the cluster
            cx = np.mean([p[0] for p in positions])
            cy = np.mean([p[1] for p in positions])
            
            inner = [[x, y, 0.0] for x, y in positions]
            
            # Check disjointness with the full check
            if not _check_disjoint(inner):
                continue
            
            side = _min_enclosing_hex_side(inner, cx, cy, oa)
            
            if side < best_side:
                best_side = side
                best_result = (inner, (cx, cy), side, oa)
    
    # If we didn't find anything, return a fallback
    if best_result is None:
        # Fallback: linear arrangement
        inner = [[i * 2 * sqrt3, 0.0, 0.0] for i in range(N)]
        cx = (N - 1) * sqrt3
        cy = 0.0
        side = _min_enclosing_hex_side(inner, cx, cy, 0.0)
        return inner, (cx, cy), side, 0.0
    
    inner, center, side, angle = best_result
    return inner, center, side, angle

# EVOLVE-BLOCK-END
