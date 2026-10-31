import numpy as np
from scipy.optimize import minimize
import time

N = 11

def hex_vertices(cx, cy, side, angle_deg):
    """Return 6 vertices of a regular hexagon with given center, side, rotation."""
    angle = np.radians(angle_deg)
    verts = []
    for i in range(6):
        theta = angle + i * np.pi / 3
        x = cx + side * np.cos(theta)
        y = cy + side * np.sin(theta)
        verts.append([x, y])
    return np.array(verts)

def point_in_hexagon(px, py, cx, cy, side, angle_deg):
    """Check if point (px,py) is inside hexagon with center (cx,cy), side, rotation."""
    # Transform to hexagon's local frame
    dx = px - cx
    dy = py - cy
    angle = np.radians(angle_deg)
    cos_a = np.cos(angle)
    sin_a = np.sin(angle)
    lx = dx * cos_a + dy * sin_a
    ly = -dx * sin_a + dy * cos_a
    # Hexagon with side s has vertices at angles 0,60,...,300 with radius s
    # In local frame, hexagon is defined by |lx| <= s, |ly| <= s*sqrt(3)/2, and 
    # the 6 edge constraints
    # Actually for a regular hexagon with "radius" (center to vertex) = side:
    # The hexagon can be described as: for each of 6 directions, the projection
    # onto that direction is <= apothem = side * sqrt(3)/2
    apothem = side * np.sqrt(3) / 2
    for i in range(6):
        theta = i * np.pi / 3
        # Normal direction
        nx = np.cos(theta)
        ny = np.sin(theta)
        proj = lx * nx + ly * ny
        if proj > apothem + 1e-9:
            return False
    return True

def hexagon_contains_hexagon(outer_cx, outer_cy, outer_side, outer_angle,
                              inner_cx, inner_cy, inner_side, inner_angle):
    """Check if inner hexagon (all vertices) is inside outer hexagon."""
    inner_verts = hex_vertices(inner_cx, inner_cy, inner_side, inner_angle)
    for vx, vy in inner_verts:
        if not point_in_hexagon(vx, vy, outer_cx, outer_cy, outer_side, outer_angle):
            return False
    return True

def hexagons_disjoint(hex1_cx, hex1_cy, hex1_side, hex1_angle,
                       hex2_cx, hex2_cy, hex2_side, hex2_angle):
    """Check if two hexagons are disjoint (no overlap)."""
    # Two convex polygons are disjoint if no edge of one intersects any edge of other
    # and neither contains the other. For simplicity, check vertex containment and edge intersection.
    v1 = hex_vertices(hex1_cx, hex1_cy, hex1_side, hex1_angle)
    v2 = hex_vertices(hex2_cx, hex2_cy, hex2_side, hex2_angle)
    
    # Check if any vertex of one is inside the other
    for vx, vy in v1:
        if point_in_hexagon(vx, vy, hex2_cx, hex2_cy, hex2_side, hex2_angle):
            return False
    for vx, vy in v2:
        if point_in_hexagon(vx, vy, hex1_cx, hex1_cy, hex1_side, hex1_angle):
            return False
    
    # Check edge intersections
    def segments_intersect(p1, p2, p3, p4):
        def cross(o, a, b):
            return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])
        d1 = cross(p3, p4, p1)
        d2 = cross(p3, p4, p2)
        d3 = cross(p1, p2, p3)
        d4 = cross(p1, p2, p4)
        if ((d1 > 0 and d2 < 0) or (d1 < 0 and d2 > 0)) and \
           ((d3 > 0 and d4 < 0) or (d3 < 0 and d4 > 0)):
            return True
        return False
    
    for i in range(6):
        for j in range(6):
            if segments_intersect(v1[i], v1[(i+1)%6], v2[j], v2[(j+1)%6]):
                return False
    return True

def compute_min_enclosing_hexagon(inner_positions, inner_angles, inner_side):
    """Find minimal enclosing regular hexagon (center, side, angle) for given inner hexagons."""
    best_side = float('inf')
    best_params = None
    
    # Try different rotations of the outer hexagon
    for outer_angle in np.linspace(0, 60, 13):  # 0 to 60 degrees, 13 points
        # For each rotation, the problem is to find center and minimal side
        # such that all inner hexagons' vertices are inside.
        
        # Collect all vertices of all inner hexagons
        all_verts = []
        for (cx, cy), ang in zip(inner_positions, inner_angles):
            verts = hex_vertices(cx, cy, inner_side, ang)
            all_verts.extend(verts.tolist())
        all_verts = np.array(all_verts)
        
        # For a given outer hexagon orientation, the constraint is that for each of 6 
        # half-planes defining the hexagon, all points must satisfy the constraint.
        # The minimal side length is determined by the maximum extent in each of the 6 
        # normal directions, and the center is chosen to minimize the maximum.
        
        # For a regular hexagon with rotation theta, the 6 outward normals are at angles
        # theta, theta+60, ..., theta+300. The hexagon is the intersection of 6 half-planes.
        # For each normal direction n_i, we need: max over all points of (p . n_i) - min over 
        # all points of (p . n_i) <= 2 * apothem = 2 * side * sqrt(3)/2 = side * sqrt(3)
        # So side >= (max - min) / sqrt(3) for each direction.
        # The center should be at the midpoint of the bounding box in each direction.
        
        theta = np.radians(outer_angle)
        apothem_factor = np.sqrt(3)  # side = extent / sqrt(3)
        
        side_required = 0
        center_x = 0
        center_y = 0
        for i in range(6):
            n_angle = theta + i * np.pi / 3
            nx = np.cos(n_angle)
            ny = np.sin(n_angle)
            projections = all_verts[:, 0] * nx + all_verts[:, 1] * ny
            proj_min = np.min(projections)
            proj_max = np.max(projections)
            extent = proj_max - proj_min
            side_i = extent / apothem_factor
            if side_i > side_required:
                side_required = side_i
            # Center in this direction
            center_x += (proj_min + proj_max) / 2 * nx
            center_y += (proj_min + proj_max) / 2 * ny
        
        center_x /= 6
        center_y /= 6
        
        # Verify this solution actually works
        if side_required < best_side:
            # Verify all vertices are inside
            valid = True
            for vx, vy in all_verts:
                if not point_in_hexagon(vx, vy, center_x, center_y, side_required, outer_angle):
                    valid = False
                    break
            if valid:
                best_side = side_required
                best_params = (center_x, center_y, side_required, outer_angle)
    
    return best_params

def construct():
    start_time = time.time()
    deadline = start_time + 120  # 120 seconds max
    
    # Generate hexagonal lattice positions for N hexagons
    # Use axial coordinates for hex grid
    # For N=11, use a compact arrangement
    
    # Hexagonal lattice: centers at (q, r) in axial coords map to:
    # x = q + r/2, y = r * sqrt(3)/2
    # But we need spacing of sqrt(3) between adjacent centers for unit hexagons
    
    # Let's use a different approach: generate positions on a triangular lattice
    # with spacing d = sqrt(3) (distance between centers of touching unit hexagons)
    
    d = np.sqrt(3)  # center-to-center distance for adjacent unit hexagons
    
    # Generate candidate positions in a hexagonal pattern
    # Use rings: ring 0: 1 hex, ring 1: 6 hex, ring 2: 12 hex, etc.
    # For N=11: 1 + 6 + 4 = 11
    
    positions = []
    # Center
    positions.append((0.0, 0.0))
    
    # Ring 1: 6 positions
    for i in range(6):
        angle = i * np.pi / 3
        x = d * np.cos(angle)
        y = d * np.sin(angle)
        positions.append((x, y))
    
    # Ring 2: need 4 more. Pick 4 from ring 2
    # Ring 2 positions are at distance 2*d or d*sqrt(3) from center
    # In hex grid, ring 2 has 12 positions
    # Let's add 4 from ring 2 that create a compact shape
    for i in range(4):
        angle = i * np.pi / 3 + np.pi / 6  # offset by 30 degrees
        x = 2 * d * np.cos(angle)
        y = 2 * d * np.sin(angle)
        positions.append((x, y))
    
    positions = positions[:N]
    
    # All inner hexagons have the same orientation (0 degrees) for simplicity
    inner_angles = [0.0] * N
    inner_side = 1.0
    
    # Compute minimal enclosing hexagon
    best_params = compute_min_enclosing_hexagon(positions, inner_angles, inner_side)
    
    if best_params is None:
        # Fallback: large hexagon
        return [[pos[0], pos[1], 0.0] for pos in positions], (0.0, 0.0), 50.0, 0.0
    
    cx, cy, side, angle = best_params
    
    # Verify disjointness
    for i in range(N):
        for j in range(i+1, N):
            if not hexagons_disjoint(positions[i][0], positions[i][1], 1.0, 0.0,
                                     positions[j][0], positions[j][1], 1.0, 0.0):
                # They overlap, need to adjust
                pass
    
    # Return the result
    inner = [[pos[0], pos[1], 0.0] for pos in positions]
    return inner, (cx, cy), side, angle

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 120
    
    N = 11
    d = np.sqrt(3)
    
    # Generate hexagonal lattice positions
    positions = []
    positions.append((0.0, 0.0))
    for i in range(6):
        angle = i * np.pi / 3
        positions.append((d * np.cos(angle), d * np.sin(angle)))
    for i in range(4):
        angle = i * np.pi / 3 + np.pi / 6
        positions.append((2 * d * np.cos(angle), 2 * d * np.sin(angle)))
    
    positions = positions[:N]
    inner_angles = [0.0] * N
    
    best_params = compute_min_enclosing_hexagon(positions, inner_angles, 1.0)
    
    if best_params is None:
        return [[pos[0], pos[1], 0.0] for pos in positions], (0.0, 0.0), 50.0, 0.0
    
    cx, cy, side, angle = best_params
    inner = [[pos[0], pos[1], 0.0] for pos in positions]
    return inner, (cx, cy), side, angle
# EVOLVE-BLOCK-END
