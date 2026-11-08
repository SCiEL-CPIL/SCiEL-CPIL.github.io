import numpy as np
from scipy.optimize import minimize
import time
import math

N = 11

# EVOLVE-BLOCK-START
def hex_vertices(cx, cy, side, angle_deg=0.0):
    """Return vertices of a regular hexagon centered at (cx,cy) with given side and rotation."""
    angle = math.radians(angle_deg)
    verts = []
    for i in range(6):
        theta = angle + i * math.pi / 3
        x = cx + side * math.cos(theta)
        y = cy + side * math.sin(theta)
        verts.append([x, y])
    return np.array(verts)

def point_in_hex(px, py, cx, cy, side, angle_deg=0.0):
    """Check if point (px,py) is inside regular hexagon centered at (cx,cy)."""
    angle = math.radians(angle_deg)
    # Transform to hexagon's local frame
    dx = px - cx
    dy = py - cy
    # Rotate back
    cos_a = math.cos(-angle)
    sin_a = math.sin(-angle)
    lx = dx * cos_a - dy * sin_a
    ly = dx * sin_a + dy * cos_a
    
    # In local frame, hexagon has vertices at angles 0, 60, 120, 180, 240, 300
    # The hexagon is defined by |lx| <= side and |ly| <= side*sqrt(3)/2 (approximately)
    # More precisely, for a flat-top hexagon with side s:
    # max(|lx|, |ly|/sqrt(3) + |lx|/2) <= s ... let me use a different approach
    
    # Actually for a pointy-top or flat-top, let me just check against the 6 half-planes
    verts = []
    for i in range(6):
        theta = angle + i * math.pi / 3
        vx = cx + side * math.cos(theta)
        vy = cy + side * math.sin(theta)
        verts.append((vx, vy))
    
    # Check if point is inside polygon using cross products
    sign = 0
    for i in range(6):
        x1, y1 = verts[i]
        x2, y2 = verts[(i + 1) % 6]
        cross = (x2 - x1) * (py - y1) - (y2 - y1) * (px - x1)
        if cross > 0:
            if sign < 0:
                return False
            sign = 1
        elif cross < 0:
            if sign > 0:
                return False
            sign = -1
    return True

def hexagons_disjoint(v1, v2, eps=1e-9):
    """Check if two hexagons (as vertex arrays) are disjoint (no interior overlap)."""
    # Check if any vertex of one is inside the other
    for i in range(6):
        if point_in_hex(v1[i, 0], v1[i, 1], *center_of(v2), side_of(v2), angle_of(v2)):
            return False
    for i in range(6):
        if point_in_hex(v2[i, 0], v2[i, 1], *center_of(v1), side_of(v1), angle_of(v1)):
            return False
    # Check edge intersections
    for i in range(6):
        if segments_intersect(v1[i], v1[(i+1)%6], v2[i], v2[(i+1)%6]):
            return False
    return True

def center_of(verts):
    return (float(np.mean(verts[:, 0])), float(np.mean(verts[:, 1])))

def side_of(verts):
    # Side length is distance between adjacent vertices
    d = np.linalg.norm(verts[1] - verts[0])
    return d

def angle_of(verts):
    # Angle of first vertex from center
    cx, cy = center_of(verts)
    return math.degrees(math.atan2(verts[0, 1] - cy, verts[0, 0] - cx))

def segments_intersect(p1, p2, p3, p4):
    """Check if segment p1-p2 intersects segment p3-p4."""
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

def hexagon_overlap(v1, v2, eps=1e-8):
    """Check if two hexagons have overlapping interiors."""
    # Check if any vertex of one is strictly inside the other
    c1 = center_of(v1)
    c2 = center_of(v2)
    s1 = side_of(v1)
    s2 = side_of(v2)
    a1 = angle_of(v1)
    a2 = angle_of(v2)
    
    for i in range(6):
        if point_in_hex(v1[i, 0], v1[i, 1], c2[0], c2[1], s2, a2):
            return True
    for i in range(6):
        if point_in_hex(v2[i, 0], v2[i, 1], c1[0], c1[1], s1, a1):
            return True
    # Check edge-edge intersections
    for i in range(6):
        for j in range(6):
            if segments_intersect(v1[i], v1[(i+1)%6], v2[j], v2[(j+1)%6]):
                return True
    return False

def all_inside_and_disjoint(inner_positions, outer_center, outer_side, outer_angle):
    """Check all inner hexagons are inside outer and mutually disjoint."""
    outer_verts = hex_vertices(outer_center[0], outer_center[1], outer_side, outer_angle)
    inner_verts = []
    for pos in inner_positions:
        x, y, ang = pos
        v = hex_vertices(x, y, 1.0, ang)
        inner_verts.append(v)
    
    # Check all inside outer
    for v in inner_verts:
        for i in range(6):
            if not point_in_hex(v[i, 0], v[i, 1], outer_center[0], outer_center[1], outer_side, outer_angle):
                return False
    
    # Check mutual disjointness
    for i in range(len(inner_verts)):
        for j in range(i+1, len(inner_verts)):
            if hexagon_overlap(inner_verts[i], inner_verts[j]):
                return False
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin
    
    # Initial arrangement: hexagonal cluster
    # Place hexagons in a compact 2D pattern
    # Hexagonal lattice spacing: for unit hexagons, centers need to be at least 
    # 2 * apothem = 2 * sqrt(3)/2 = sqrt(3) apart for same orientation
    # But with rotation, we can pack tighter
    
    # Start with a 3-2-3-3 or similar arrangement
    # For flat-top hexagons, the horizontal spacing is 3/2 * side = 1.5
    # Vertical spacing is sqrt(3) * side = sqrt(3)
    
    sqrt3 = math.sqrt(3)
    
    # Initial positions: a compact cluster
    # Row 0: 3 hexagons
    # Row 1: 2 hexagons (offset)
    # Row 2: 3 hexagons
    # Row 3: 3 hexagons
    # Total: 11
    
    # Actually let me try a more symmetric arrangement
    # Center + 6 around + 4 more
    
    # Simple approach: grid-like arrangement with hexagonal packing
    # Positions for 11 hexagons in a hexagonal pattern
    
    # Let me use a known good starting point
    # Arrange in rows: 2, 3, 3, 3 or 3, 2, 3, 3
    
    # For flat-top hexagons with side 1:
    # Width = 2, height = sqrt(3)
    # Horizontal spacing between centers in same row: 3 (2 * 1.5)
    # Actually for touching flat-top hexagons: horizontal = 3/2 * 2 = 3? No.
    # Flat-top hex: vertices at 0, 60, 120, 180, 240, 300 degrees
    # Width (left to right) = 2 * side = 2
    # Height (top to bottom) = sqrt(3) * side = sqrt(3)
    # To pack: horizontal spacing = 3/2 * side * 2 = 3*side? 
    # Actually adjacent flat-top hexagons: center-to-center = 3/2 * side horizontally if sharing a vertical edge
    
    # Let me just use a reasonable initial guess and optimize
    # Place them in a roughly circular/cluster arrangement
    
    # Initial: place in a 4-3-4 pattern or similar
    # Row y=0: x = -3, -1.5, 0, 1.5, 3 (5 hexagons) - too many
    # Let me do 3-2-3-3
    
    # For flat-top hexagons (angle=0), vertices at 0,60,120,180,240,300
    # The "width" is 2 (from left vertex to right vertex)
    # The "height" is sqrt(3)
    # Adjacent in x: need center distance >= 3/2 * 2 = 3? No, that's for pointy-top.
    # For flat-top: two hexagons side by side (sharing a vertical edge): center distance = 3/2 * side * 2 = 3
    # Wait, no. Flat-top hex with side 1: leftmost point at x=-1, rightmost at x=1.
    # Two such hexagons touching: centers at x=0 and x=3? That gives gap.
    # Actually the distance between centers of two touching flat-top hexagons sharing a vertical edge is 3/2 * side * 2 = 3? 
    # Let me compute: flat-top hex vertices: (1,0), (0.5, sqrt(3)/2), (-0.5, sqrt(3)/2), (-1,0), (-0.5,-sqrt(3)/2), (0.5,-sqrt(3)/2)
    # Right edge connects (1,0) and (0.5, sqrt(3)/2) and (0.5, -sqrt(3)/2). 
    # For two hexagons to share this edge, the second center is at (1 + 0.5, 0) = (1.5, 0)? 
    # No, the edge is vertical-ish. The midpoint of the right edge is at ((1+0.5)/2, 0) = (0.75, 0).
    # The perpendicular distance from center to this edge... 
    
    # I'm overcomplicating. Let me just use a numerical approach.
    # Start with positions spread out, then optimize.
    
    # Simple initial: place in a line but with some 2D structure
    # Use a hexagonal lattice with spacing sqrt(3) for same-orientation hexagons
    
    # For same-orientation unit hexagons, minimum center distance = sqrt(3) (for pointy-top sharing an edge)
    # or 3/2 * 2 = 3 for flat-top? Let me just use sqrt(3) as a safe initial spacing.
    
    # Actually, for two unit regular hexagons to be tangent (touching at one edge),
    # the center-to-center distance is 2 * apothem = 2 * (sqrt(3)/2) = sqrt(3) ≈ 1.732
    
    # So I can pack them on a triangular lattice with spacing sqrt(3)
    
    # Initial arrangement: triangular lattice
    positions = []
    # 4-3-4 pattern
    # Row 0 (y=0): 4 hexagons
    # Row 1 (y=sqrt(3)/2): 3 hexagons  
    # Row 2 (y=sqrt(3)): 4 hexagons
    
    dx = sqrt3  # horizontal spacing for touching
    dy = sqrt3 / 2  # vertical spacing
    
    # Actually for triangular lattice with hexagon packing:
    # Horizontal spacing = sqrt(3)
    # Vertical spacing = sqrt(3) * sqrt(3)/2 = 3/2? No.
    # Triangular lattice: rows offset by half spacing.
    # If horizontal spacing is d, vertical spacing is d * sqrt(3)/2
    
    d = sqrt3  # center-to-center for touching hexagons
    dy_row = d * math.sqrt(3) / 2  # = sqrt(3) * sqrt(3)/2 = 3/2
    
    # Row 0: 4 hexagons at y=0
    # Row 1: 3 hexagons at y=dy_row, offset by d/2
    # Row 2: 4 hexagons at y=2*dy_row
    
    for i in range(4):
        positions.append([i * d - 1.5 * d, 0.0, 0.0])
    for i in range(3):
        positions.append([i * d - d, dy_row, 0.0])
    for i in range(4):
        positions.append([i * d - 1.5 * d, 2 * dy_row, 0.0])
    
    # That's 11 positions
    
    # Compute bounding hexagon
    def compute_outer_side(positions, outer_angle=0.0):
        """Compute minimum outer hexagon side that contains all positions."""
        # For a regular hexagon, we need to find the minimum side such that
        # all points are inside.
        # The hexagon is defined by 6 half-planes.
        # For a hexagon with center (0,0) and angle 0, the constraints are:
        # |x| <= side, and |y| + |x|/sqrt(3) <= side * sqrt(3)/2 ... 
        # Actually, let me just compute the maximum distance needed.
        
        # For a regular hexagon with side s and center at origin, angle 0:
        # The support function in direction theta is s * cos(theta - k*pi/3) for appropriate k
        # More simply: the hexagon contains point (x,y) iff for all 6 normals n_i, n_i . (x,y) <= s * (apothem factor)
        
        # Let me just binary search on side length
        def fits(side):
            for pos in positions:
                x, y, _ = pos
                # Check if (x,y) is inside hexagon of given side centered at origin
                if not point_in_hex(x, y, 0, 0, side, outer_angle):
                    return False
            return True
        
        lo, hi = 1.0, 20.0
        for _ in range(50):
            mid = (lo + hi) / 2
            if fits(mid):
                hi = mid
            else:
                lo = mid
        return hi
    
    # Objective: minimize outer side length
    def objective(params):
        # params: 33 values (11 * 3: x, y, angle for each)
        # Or we can fix angles to 0 for simplicity first
        elapsed = time.time() - start_time
        if elapsed > deadline - 5.0:
            return 1e10
        
        positions = []
        for i in range(N):
            x = params[3*i]
            y = params[3*i+1]
            ang = params[3*i+2]
            positions.append([x, y, ang])
        
        # Compute outer side
        outer_side = compute_outer_side(positions)
        
        # Check validity
        if not all_inside_and_disjoint(positions, (0.0, 0.0), outer_side, 0.0):
            return 1e10
        
        return outer_side
    
    # Initial params
    x0 = []
    for pos in positions:
        x0.extend([pos[0], pos[1], pos[2]])
    
    # Optimize
    best_result = None
    best_score = compute_outer_side(positions)
    
    # Try optimization
    try:
        result = minimize(objective, x0, method='Nelder-Mead', 
                         options={'maxiter': 200, 'xatol': 0.01, 'fatol': 0.01})
        
        if result.fun < best_score and result.fun < 1e9:
            best_score = result.fun
            best_params = result.x
            best_positions = []
            for i in range(N):
                best_positions.append([best_params[3*i], best_params[3*i+1], best_params[3*i+2]])
            
            # Verify
            if all_inside_and_disjoint(best_positions, (0.0, 0.0), best_score, 0.0):
                best_result = best_positions
    except Exception as e:
        pass
    
    # If optimization didn't help or failed, use initial
    if best_result is None:
        best_result = positions
    
    # Final verification
    final_side = compute_outer_side(best_result)
    if not all_inside_and_disjoint(best_result, (0.0, 0.0), final_side, 0.0):
        # Fallback to simple line arrangement
        best_result = [[4.0 * k, 0.0, 0.0] for k in range(-(N // 2), N - N // 2)]
        final_side = 40.0
    
    return best_result, (0.0, 0.0), final_side, 0.0
# EVOLVE-BLOCK-END
