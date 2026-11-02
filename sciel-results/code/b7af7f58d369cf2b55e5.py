import numpy as np
from scipy.optimize import minimize
import time

N = 12

def hex_vertices(cx, cy, side, angle_deg):
    """Get vertices of a regular hexagon."""
    angle = np.radians(angle_deg)
    angles = angle + np.arange(6) * np.pi / 3
    return np.column_stack([cx + side * np.cos(angles), cy + side * np.sin(angles)])

def point_in_hexagon(px, py, hx, hy, side, angle_deg):
    """Check if point (px,py) is inside hexagon centered at (hx,hy) with given side and angle."""
    # Transform point to hexagon's local frame
    dx = px - hx
    dy = py - hy
    angle = np.radians(angle_deg)
    # Rotate back
    cos_a = np.cos(-angle)
    sin_a = np.sin(-angle)
    lx = dx * cos_a - dy * sin_a
    ly = dx * sin_a + dy * cos_a
    # For a regular hexagon with flat top (angle=0), vertices at 0,60,120,180,240,300 degrees
    # The hexagon can be described by 6 half-planes
    # For a unit hexagon (side=1), the support function in direction theta is cos(theta - k*pi/3) * side for the rightmost edge
    # Actually, for a regular hexagon centered at origin with one vertex at angle 0:
    # The hexagon is the intersection of 6 half-planes
    # In local coords, the hexagon with circumradius R=side has:
    # |x*cos(k*pi/3) + y*sin(k*pi/3)| <= side*cos(pi/6) = side*sqrt(3)/2 for k=0..5
    # Wait, that's not right. Let me think again.
    
    # A regular hexagon with side length s has circumradius R=s and inradius r=s*sqrt(3)/2
    # With a vertex at angle 0, the vertices are at (s*cos(k*pi/3), s*sin(k*pi/3)) for k=0..5
    # The edges are between consecutive vertices. The half-plane for each edge:
    # Edge from vertex k to vertex k+1 has normal pointing outward at angle (k+0.5)*pi/3
    # The distance from center to each edge is the inradius = s*sqrt(3)/2
    
    # So the hexagon is: for k=0..5, |x*cos((k+0.5)*pi/3) + y*sin((k+0.5)*pi/3)| <= s*sqrt(3)/2
    # Wait, that's not quite right either. Let me be more careful.
    
    # For a regular hexagon with vertices at angles 0, 60, 120, 180, 240, 300 (degrees) and circumradius R:
    # Vertex k is at (R*cos(k*pi/3), R*sin(k*pi/3))
    # The edge between vertex k and k+1 has a normal pointing outward.
    # The midpoint of edge k is at angle (k+0.5)*pi/3 from center, at distance R*cos(pi/6) = R*sqrt(3)/2
    # So the half-plane constraint is: x*cos((k+0.5)*pi/3) + y*sin((k+0.5)*pi/3) <= R*sqrt(3)/2
    
    # With R = side (since for a regular hexagon, side = circumradius)
    R = side
    inradius = R * np.sqrt(3) / 2
    for k in range(6):
        normal_angle = (k + 0.5) * np.pi / 3
        nx = np.cos(normal_angle)
        ny = np.sin(normal_angle)
        if lx * nx + ly * ny > inradius:
            return False
    return True

def hexagon_distance_to_point(px, py, hx, hy, side, angle_deg):
    """Return negative if point is outside, 0 if on boundary, positive if inside.
    Returns the minimum signed distance from point to hexagon boundary."""
    dx = px - hx
    dy = py - hy
    angle = np.radians(angle_deg)
    cos_a = np.cos(-angle)
    sin_a = np.sin(-angle)
    lx = dx * cos_a - dy * sin_a
    ly = dx * sin_a + dy * cos_a
    
    R = side
    inradius = R * np.sqrt(3) / 2
    min_dist = np.inf
    for k in range(6):
        normal_angle = (k + 0.5) * np.pi / 3
        nx = np.cos(normal_angle)
        ny = np.sin(normal_angle)
        dist = inradius - (lx * nx + ly * ny)
        min_dist = min(min_dist, dist)
    return min_dist

def hexagons_disjoint(h1, h2):
    """Check if two hexagons are disjoint (no overlap). h = (cx, cy, angle_deg) for unit hexagons."""
    cx1, cy1, a1 = h1
    cx2, cy2, a2 = h2
    
    # Get vertices of both hexagons
    v1 = hex_vertices(cx1, cy1, 1.0, a1)
    v2 = hex_vertices(cx2, cy2, 1.0, a2)
    
    # Check if any vertex of one is inside the other
    for px, py in v1:
        if point_in_hexagon(px, py, cx2, cy2, 1.0, a2):
            return False
    for px, py in v2:
        if point_in_hexagon(px, py, cx1, cy1, 1.0, a1):
            return False
    
    # Check edge intersections
    # This is complex; for now, use a simpler check
    # If centers are far enough apart, they're likely disjoint
    dist = np.sqrt((cx1-cx2)**2 + (cy1-cy2)**2)
    if dist > 2.0:  # Max distance between any two points in unit hexagons is 2
        return True
    
    # More careful check: check all edge pairs
    for i in range(6):
        p1 = v1[i]
        p2 = v1[(i+1)%6]
        for j in range(6):
            p3 = v2[j]
            p4 = v2[(j+1)%6]
            if segments_intersect(p1, p2, p3, p4):
                return False
    return True

def segments_intersect(p1, p2, p3, p4):
    """Check if line segment p1-p2 intersects p3-p4."""
    def cross(o, a, b):
        return (a[0]-o[0])*(b[1]-o[1]) - (a[1]-o[1])*(b[0]-o[0])
    
    d1 = cross(p3, p4, p1)
    d2 = cross(p3, p4, p2)
    d3 = cross(p1, p2, p3)
    d4 = cross(p1, p2, p4)
    
    if ((d1 > 0 and d2 < 0) or (d1 < 0 and d2 > 0)) and \
       ((d3 > 0 and d4 < 0) or (d3 < 0 and d4 > 0)):
        return True
    return False

def check_valid(inner, outer_center, outer_side, outer_angle):
    """Check if all inner hexagons are inside the outer hexagon and disjoint."""
    ox, oy = outer_center
    
    # Check each inner hexagon is inside outer
    for cx, cy, angle in inner:
        # Check all vertices of inner hexagon are inside outer
        vertices = hex_vertices(cx, cy, 1.0, angle)
        for px, py in vertices:
            if not point_in_hexagon(px, py, ox, oy, outer_side, outer_angle):
                return False
    
    # Check pairwise disjointness
    for i in range(len(inner)):
        for j in range(i+1, len(inner)):
            if not hexagons_disjoint(inner[i], inner[j]):
                return False
    
    return True

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s limit
    
    # Try a hexagonal packing arrangement
    # Place 12 hexagons in a pattern that fits well in a hexagon
    
    # Strategy: Use a 3x4 grid-like arrangement in hexagonal coordinates
    # Or try to find a good initial arrangement and optimize
    
    # Initial guess: place hexagons in a compact arrangement
    # Hexagonal lattice positions for 12 hexagons
    
    # Let's try a 2-3-3-2-2 arrangement or similar
    # Actually, let's try a more systematic approach
    
    # For a hexagonal grid, positions in axial coordinates (q, r)
    # Convert to Cartesian: x = q + r/2, y = r * sqrt(3)/2
    # But we need to scale for unit hexagons
    
    # In a hexagonal tiling, the distance between centers of adjacent hexagons is 2*inradius = sqrt(3)
    # So if we use a hexagonal lattice with spacing sqrt(3), hexagons will be tangent
    
    # Let's place 12 hexagons in a hexagonal pattern
    # A hexagon of "radius" 1 in hex grid has 7 hexagons (center + 6 neighbors)
    # We need 12, so let's use a larger pattern
    
    # Try: 3 rows of 4, or 4 rows of 3, arranged in hex grid
    positions = []
    
    # Hex grid coordinates for a compact 12-hexagon shape
    # Let's use a 3x4 rectangle in hex grid, offset properly
    # Row 0: 4 hexagons
    # Row 1: 4 hexagons (offset)
    # Row 2: 4 hexagons
    
    # In hex grid with pointy-top, adjacent rows are offset by half a hex width
    # Hex width = 2*inradius = sqrt(3)
    # Hex height = 2*circumradius = 2
    
    spacing = np.sqrt(3)  # Distance between centers of tangent hexagons
    
    # Let's try a 3x4 grid
    for row in range(3):
        for col in range(4):
            x = col * spacing + (row % 2) * spacing / 2
            y = row * 1.5  # Vertical spacing for hex grid is 1.5 * circumradius = 1.5
            positions.append([x, y, 0.0])
    
    # Center the arrangement
    xs = [p[0] for p in positions]
    ys = [p[1] for p in positions]
    cx_mean = (min(xs) + max(xs)) / 2
    cy_mean = (min(ys) + max(ys)) / 2
    positions = [[p[0] - cx_mean, p[1] - cy_mean, p[2]] for p in positions]
    
    # Now find the minimum enclosing hexagon
    # The outer hexagon needs to contain all vertices of all inner hexagons
    
    # For a flat-top hexagon (angle=0), the constraints are:
    # For each point (px, py), we need:
    # |px*cos(k*pi/6) + py*sin(k*pi/6)| <= S*sqrt(3)/2 for k=0..5
    # where S is the side length
    
    # To find minimum S, we maximize over all points and k:
    # S >= 2/sqrt(3) * max_k |px*cos(k*pi/6) + py*sin(k*pi/6)|
    
    # But we can also rotate the outer hexagon to minimize S
    
    # Let's compute the required S for angle=0 first
    all_vertices = []
    for cx, cy, angle in positions:
        verts = hex_vertices(cx, cy, 1.0, angle)
        all_vertices.append(verts)
    
    all_verts = np.vstack(all_vertices)
    
    # For a given outer angle, find minimum S
    def min_side_for_angle(outer_angle_deg):
        angle = np.radians(outer_angle_deg)
        cos_a = np.cos(-angle)
        sin_a = np.sin(-angle)
        
        # Rotate all vertices
        rx = all_verts[:, 0] * cos_a - all_verts[:, 1] * sin_a
        ry = all_verts[:, 0] * sin_a + all_verts[:, 1] * cos_a
        
        max_val = 0
        for k in range(6):
            normal_angle = (k + 0.5) * np.pi / 3
            nx = np.cos(normal_angle)
            ny = np.sin(normal_angle)
            vals = np.abs(rx * nx + ry * ny)
            max_val = max(max_val, np.max(vals))
        
        return 2 * max_val / np.sqrt(3)
    
    # Try different angles
    best_side = np.inf
    best_angle = 0
    for angle_deg in np.linspace(0, 60, 61):
        side = min_side_for_angle(angle_deg)
        if side < best_side:
            best_side = side
            best_angle = angle_deg
    
    # Verify this arrangement is valid
    inner = positions
    outer_center = (0.0, 0.0)
    outer_side = best_side
    outer_angle = best_angle
    
    if check_valid(inner, outer_center, outer_side, outer_angle):
        # Try to improve with optimization
        elapsed = time.time() - start_time
        if elapsed < deadline - 10:
            # Try local optimization of positions
            def objective(params):
                # params: 3*N values (x, y, angle for each hexagon)
                inner_params = params.reshape(N, 3)
                cx_list = inner_params[:, 0]
                cy_list = inner_params[:, 1]
                angle_list = inner_params[:, 2]
                
                # Compute all vertices
                all_verts = []
                for i in range(N):
                    verts = hex_vertices(cx_list[i], cy_list[i], 1.0, angle_list[i])
                    all_verts.append(verts)
                all_verts = np.vstack(all_verts)
                
                # Find min enclosing hexagon side for best angle
                angle = np.radians(outer_angle)
                cos_a = np.cos(-angle)
                sin_a = np.sin(-angle)
                rx = all_verts[:, 0] * cos_a - all_verts[:, 1] * sin_a
                ry = all_verts[:, 0] * sin_a + all_verts[:, 1] * cos_a
                
                max_val = 0
                for k in range(6):
                    normal_angle = (k + 0.5) * np.pi / 3
                    nx = np.cos(normal_angle)
                    ny = np.sin(normal_angle)
                    vals = np.abs(rx * nx + ry * ny)
                    max_val = max(max_val, np.max(vals))
                
                side = 2 * max_val / np.sqrt(3)
                
                # Add penalty for overlaps
                penalty = 0
                for i in range(N):
                    for j in range(i+1, N):
                        dist = np.sqrt((cx_list[i]-cx_list[j])**2 + (cy_list[i]-cy_list[j])**2)
                        if dist < 1.8:  # Potential overlap
                            penalty += (1.8 - dist) ** 2 * 100
                
                return side + penalty
            
            x0 = np.array(inner).flatten()
            try:
                result = minimize(objective, x0, method='L-BFGS-B', options={'maxiter': 100})
                if result.fun < best_side:
                    inner_opt = result.x.reshape(N, 3)
                    # Recompute best side
                    all_verts = []
                    for i in range(N):
                        verts = hex_vertices(inner_opt[i, 0], inner_opt[i, 1], 1.0, inner_opt[i, 2])
                        all_verts.append(verts)
                    all_verts = np.vstack(all_verts)
                    
                    angle = np.radians(outer_angle)
                    cos_a = np.cos(-angle)
                    sin_a = np.sin(-angle)
                    rx = all_verts[:, 0] * cos_a - all_verts[:, 1] * sin_a
                    ry = all_verts[:, 0] * sin_a + all_verts[:, 1] * cos_a
                    
                    max_val = 0
                    for k in range(6):
                        normal_angle = (k + 0.5) * np.pi / 3
                        nx = np.cos(normal_angle)
                        ny = np.sin(normal_angle)
                        vals = np.abs(rx * nx + ry * ny)
                        max_val = max(max_val, np.max(vals))
                    
                    best_side = 2 * max_val / np.sqrt(3)
                    inner = [list(row) for row in inner_opt]
            except:
                pass
    
    # Final check
    if not check_valid(inner, outer_center, outer_side, outer_angle):
        # Fall back to a known valid arrangement
        # Just use a larger hexagon
        inner = [[4.0 * k, 0.0, 0.0] for k in range(-(N // 2), N - N // 2)]
        outer_center = (0.0, 0.0)
        outer_side = 40.0
        outer_angle = 0.0
    
    return inner, outer_center, outer_side, outer_angle
# EVOLVE-BLOCK-END
