"""Optimized packing of 12 unit hexagons in a regular hexagon."""
import numpy as np
from scipy.optimize import minimize
import time

N = 12

# EVOLVE-BLOCK-START
def _hex_vertices(cx, cy, side, angle_deg):
    """Return 6 vertices of a regular hexagon centered at (cx,cy) with given side and rotation."""
    a = np.radians(angle_deg)
    verts = []
    for k in range(6):
        theta = a + k * np.pi / 3.0
        verts.append((cx + side * np.cos(theta), cy + side * np.sin(theta)))
    return np.array(verts)

def _point_in_hex(px, py, cx, cy, side, angle_deg):
    """Check if point (px,py) is inside regular hexagon centered at (cx,cy)."""
    # Transform point into hexagon's local frame
    dx = px - cx
    dy = py - cy
    a = np.radians(angle_deg)
    cos_a = np.cos(a)
    sin_a = np.sin(a)
    lx = dx * cos_a + dy * sin_a
    ly = -dx * sin_a + dy * cos_a
    # For a regular hexagon with 'pointy' top (vertex at top), the boundary in local coords:
    # The hexagon with vertices at angles 0,60,...,300 has flat sides.
    # Actually for vertices at 0,60,...: the supporting lines are at angles 30,90,...
    # Let's use the standard: hexagon with flat top has vertices at 30,90,150,210,270,330
    # For our parameterization (vertices at angle + k*60), the inradius is side * sqrt(3)/2
    # The hexagon is the intersection of 6 half-planes.
    # For a hexagon with vertices at angles a+k*60, the edges are perpendicular to directions a+30+k*60
    # Distance from center to each edge = side * sqrt(3)/2
    inradius = side * np.sqrt(3) / 2.0
    for k in range(6):
        edge_angle = a + (k + 0.5) * np.pi / 3.0
        # Normal direction pointing inward
        nx = np.cos(edge_angle)
        ny = np.sin(edge_angle)
        # Project point onto normal
        proj = lx * nx + ly * ny
        if proj > inradius:
            return False
    return True

def _hex_contains_hex(inner_verts, outer_cx, outer_cy, outer_side, outer_angle):
    """Check if all vertices of inner hex are inside outer hexagon."""
    for vx, vy in inner_verts:
        if not _point_in_hex(vx, vy, outer_cx, outer_cy, outer_side, outer_angle):
            return False
    return True

def _min_side_for_config(inner_positions, inner_angles, outer_cx=0.0, outer_cy=0.0, outer_angle=0.0):
    """Find minimum outer side length such that all inner hexagons fit inside outer hexagon at given center/angle."""
    # Binary search on side length
    lo = 0.0
    hi = 100.0
    
    def fits(side):
        for (ix, iy, iang) in zip(inner_positions, inner_angles):
            verts = _hex_vertices(ix, iy, 1.0, iang)
            for vx, vy in verts:
                if not _point_in_hex(vx, vy, outer_cx, outer_cy, side, outer_angle):
                    return False
        return True
    
    if not fits(hi):
        return hi
    
    for _ in range(50):
        mid = (lo + hi) / 2.0
        if fits(mid):
            hi = mid
        else:
            lo = mid
    return hi

def construct():
    t0 = time.time()
    deadline = 145.0  # seconds, safety margin before 150s kill
    
    # Generate candidate inner configurations
    # Honeycomb-like arrangements of 12 hexagons
    # Each hexagon has center (x,y) and rotation angle (degrees)
    
    candidates = []
    
    # Pattern 1: 3-4-3-2 rows (staggered)
    # Row spacing for hexagons of side 1: vertical distance = sqrt(3)
    # Horizontal spacing: 2 for same row, offset by 1 for adjacent rows
    dy = np.sqrt(3)
    
    # Try various compact arrangements
    arrangements = []
    
    # Arrangement A: 4-4-4 in a triangle-ish shape
    # Row 0: 4 hexagons
    # Row 1: 4 hexagons (offset)
    # Row 2: 4 hexagons
    for row_offset in [0.0, 0.5, 1.0]:
        pos = []
        # 3 rows of 4
        for row in range(3):
            for col in range(4):
                x = col * 2.0 + row_offset * 1.0
                y = row * dy
                pos.append((x, y, 0.0))
        arrangements.append(pos)
        
    # Arrangement B: 3-4-3-2
    for row_offset in [0.0, 0.5]:
        pos = []
        counts = [3, 4, 3, 2]
        for row, count in enumerate(counts):
            for col in range(count):
                x = col * 2.0 + row_offset * 1.0 + (4 - count) * 1.0
                y = row * dy
                pos.append((x, y, 0.0))
        arrangements.append(pos)
        
    # Arrangement C: 2-3-4-3
    for row_offset in [0.0, 0.5]:
        pos = []
        counts = [2, 3, 4, 3]
        for row, count in enumerate(counts):
            for col in range(count):
                x = col * 2.0 + row_offset * 1.0 + (4 - count) * 1.0
                y = row * dy
                pos.append((x, y, 0.0))
        arrangements.append(pos)
        
    # Arrangement D: 4-3-3-2
    for row_offset in [0.0, 0.5]:
        pos = []
        counts = [4, 3, 3, 2]
        for row, count in enumerate(counts):
            for col in range(count):
                x = col * 2.0 + row_offset * 1.0 + (4 - count) * 1.0
                y = row * dy
                pos.append((x, y, 0.0))
        arrangements.append(pos)
        
    # Arrangement E: 5-4-3 (triangle)
    for row_offset in [0.0, 0.5]:
        pos = []
        counts = [5, 4, 3]
        for row, count in enumerate(counts):
            for col in range(count):
                x = col * 2.0 + row_offset * 1.0 + (5 - count) * 1.0
                y = row * dy
                pos.append((x, y, 0.0))
        arrangements.append(pos)
        
    # Arrangement F: 3-3-3-3 (4 rows of 3)
    for row_offset in [0.0, 0.5]:
        pos = []
        counts = [3, 3, 3, 3]
        for row, count in enumerate(counts):
            for col in range(count):
                x = col * 2.0 + row_offset * 1.0
                y = row * dy
                pos.append((x, y, 0.0))
        arrangements.append(pos)
        
    # Arrangement G: 2-4-4-2
    for row_offset in [0.0, 0.5]:
        pos = []
        counts = [2, 4, 4, 2]
        for row, count in enumerate(counts):
            for col in range(count):
                x = col * 2.0 + row_offset * 1.0 + (4 - count) * 1.0
                y = row * dy
                pos.append((x, y, 0.0))
        arrangements.append(pos)
        
    # Arrangement H: 1-3-4-3-1
    for row_offset in [0.0, 0.5]:
        pos = []
        counts = [1, 3, 4, 3, 1]
        for row, count in enumerate(counts):
            for col in range(count):
                x = col * 2.0 + row_offset * 1.0 + (4 - count) * 1.0
                y = row * dy
                pos.append((x, y, 0.0))
        arrangements.append(pos)
        
    # Arrangement I: 6-6 (two rows of 6)
    for row_offset in [0.0, 0.5]:
        pos = []
        counts = [6, 6]
        for row, count in enumerate(counts):
            for col in range(count):
                x = col * 2.0 + row_offset * 1.0
                y = row * dy
                pos.append((x, y, 0.0))
        arrangements.append(pos)

    # Filter arrangements that have exactly N hexagons
    valid_arrangements = [arr for arr in arrangements if len(arr) == N]
    
    if not valid_arrangements:
        # Fallback: simple line
        valid_arrangements = [[(2.0 * k, 0.0, 0.0) for k in range(N)]]
    
    best_side = float('inf')
    best_inner = None
    best_center = (0.0, 0.0)
    best_angle = 0.0
    
    # For each arrangement, optimize outer hexagon parameters
    for arr in valid_arrangements:
        if time.time() - t0 > deadline:
            break
            
        inner_positions = np.array([(p[0], p[1]) for p in arr])
        inner_angles = np.array([p[2] for p in arr])
        
        # Center the arrangement
        cx_mean = np.mean(inner_positions[:, 0])
        cy_mean = np.mean(inner_positions[:, 1])
        centered_positions = inner_positions - np.array([cx_mean, cy_mean])
        
        # Try various outer rotations
        best_for_arr = float('inf')
        best_outer_angle = 0.0
        best_outer_cx = 0.0
        best_outer_cy = 0.0
        
        # Coarse search over outer rotation
        for outer_angle in np.linspace(0, 60, 13):
            if time.time() - t0 > deadline:
                break
                
            # Compute minimum side for this rotation
            side = _min_side_for_config(
                centered_positions[:, 0].tolist(),
                centered_positions[:, 1].tolist(),
                inner_angles.tolist(),
                outer_cx=0.0,
                outer_cy=0.0,
                outer_angle=float(outer_angle)
            )
            
            if side < best_for_arr:
                best_for_arr = side
                best_outer_angle = float(outer_angle)
                best_outer_cx = 0.0
                best_outer_cy = 0.0
        
        if best_for_arr < best_side:
            best_side = best_for_arr
            best_inner = [(float(x), float(y), float(a)) for (x, y, a) in zip(
                centered_positions[:, 0], centered_positions[:, 1], inner_angles
            )]
            best_center = (best_outer_cx, best_outer_cy)
            best_angle = best_outer_angle
    
    # Fine-tune the best result with local optimization
    if best_inner is not None and time.time() - t0 < deadline - 10:
        # Try to refine with small perturbations of outer angle
        for delta_a in np.linspace(-3, 3, 13):
            test_angle = (best_angle + delta_a) % 60.0
            test_side = _min_side_for_config(
                [p[0] for p in best_inner],
                [p[1] for p in best_inner],
                [p[2] for p in best_inner],
                outer_cx=best_center[0],
                outer_cy=best_center[1],
                outer_angle=test_angle
            )
            if test_side < best_side:
                best_side = test_side
                best_angle = test_angle
    
    # Return the result
    # Ensure we return a valid format
    if best_inner is None:
        # Absolute fallback
        inner = [[2.0 * k, 0.0, 0.0] for k in range(N)]
        return inner, (0.0, 0.0), 40.0, 0.0
    
    return best_inner, best_center, float(best_side), float(best_angle)
# EVOLVE-BLOCK-END
