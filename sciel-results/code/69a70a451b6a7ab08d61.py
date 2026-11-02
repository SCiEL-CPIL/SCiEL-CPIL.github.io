import numpy as np
import time

N = 11

def _hex_vertices(cx, cy, side, angle_deg):
    """Return vertices of a regular hexagon."""
    angle = np.radians(angle_deg)
    verts = []
    for i in range(6):
        theta = angle + np.pi * i / 3.0
        verts.append([cx + side * np.cos(theta), cy + side * np.sin(theta)])
    return np.array(verts)

def _point_in_hex(px, py, cx, cy, side, angle_deg):
    """Check if point is inside hexagon. Returns True if inside."""
    angle = np.radians(angle_deg)
    dx = px - cx
    dy = py - cy
    cos_a = np.cos(-angle)
    sin_a = np.sin(-angle)
    lx = dx * cos_a - dy * sin_a
    ly = dx * sin_a + dy * cos_a
    
    apothem = side * np.sqrt(3) / 2.0
    for i in range(6):
        theta = np.pi * i / 3.0
        nx = np.cos(theta)
        ny = np.sin(theta)
        d = lx * nx + ly * ny
        if d > apothem + 1e-12:
            return False
    return True

def _is_valid_packing(inner, outer_center, outer_side, outer_angle):
    """Check if all inner hexagons fit inside the outer hexagon."""
    ocx, ocy = outer_center
    for (ix, iy, iangle) in inner:
        verts = _hex_vertices(ix, iy, 1.0, iangle)
        for v in verts:
            if not _point_in_hex(v[0], v[1], ocx, ocy, outer_side, outer_angle):
                return False
    return True

def _min_side_for_arrangement(inner, outer_center, outer_angle):
    """Binary search for minimum outer side length."""
    lo = 0.5
    hi = 50.0
    for _ in range(50):
        mid = (lo + hi) / 2.0
        if _is_valid_packing(inner, outer_center, mid, outer_angle):
            hi = mid
        else:
            lo = mid
    return hi

def construct():
    start_time = time.time()
    deadline = start_time + 135.0
    
    # Honeycomb arrangement: center + 6 neighbors + 4 more
    base_positions = [(0.0, 0.0)]
    for i in range(6):
        theta = np.pi * i / 3.0
        x = np.sqrt(3) * np.cos(theta)
        y = np.sqrt(3) * np.sin(theta)
        base_positions.append((x, y))
    
    # Second ring positions (distance 2*sqrt(3) from center)
    second_ring = []
    for i in range(12):
        theta = np.pi * i / 6.0
        x = 2 * np.sqrt(3) * np.cos(theta)
        y = 2 * np.sqrt(3) * np.sin(theta)
        second_ring.append((x, y))
    
    best_side = float('inf')
    best_inner = None
    best_center = (0.0, 0.0)
    best_angle = 0.0
    
    # Try different subsets of 4 from second ring
    candidate_indices = [
        [0, 2, 4, 6],
        [0, 3, 6, 9],
        [1, 3, 5, 7],
        [0, 2, 5, 7],
        [1, 4, 7, 10],
        [0, 4, 8, 2],
    ]
    
    for idx_set in candidate_indices:
        if time.time() > deadline - 10.0:
            break
        
        positions = list(base_positions)
        for idx in idx_set:
            positions.append(second_ring[idx])
        
        if len(positions) != 11:
            continue
        
        inner = [[p[0], p[1], 0.0] for p in positions]
        
        # Try different outer orientations and centers
        for outer_angle in [0.0, 30.0, 60.0, 15.0, 45.0]:
            if time.time() > deadline - 10.0:
                break
            
            # Sample outer centers
            for ocx in np.linspace(-1.5, 1.5, 7):
                for ocy in np.linspace(-1.5, 1.5, 7):
                    if time.time() > deadline - 10.0:
                        break
                    
                    side = _min_side_for_arrangement(inner, (ocx, ocy), outer_angle)
                    if side < best_side:
                        best_side = side
                        best_inner = inner
                        best_center = (float(ocx), float(ocy))
                        best_angle = outer_angle
    
    if best_inner is None:
        # Fallback
        inner = [[p[0], p[1], 0.0] for p in base_positions[:11]]
        return inner, (0.0, 0.0), 40.0, 0.0
    
    return best_inner, best_center, best_side, best_angle

# EVOLVE-BLOCK-START
# (construct is defined above)
# EVOLVE-BLOCK-END
