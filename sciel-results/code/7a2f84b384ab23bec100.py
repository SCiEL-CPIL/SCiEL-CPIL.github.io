"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 11

# EVOLVE-BLOCK-START
import numpy as np
import time

def hexagon_vertices(cx, cy, side, angle_deg=0.0):
    """Return vertices of a regular hexagon centered at (cx,cy)."""
    angles = np.radians(angle_deg) + np.pi / 6 + np.arange(6) * np.pi / 3
    return np.column_stack([cx + side * np.cos(angles), cy + side * np.sin(angles)])

def point_in_hexagon(px, py, cx, cy, side, angle_deg=0.0):
    """Check if point is inside hexagon."""
    angle = np.radians(angle_deg)
    dx, dy = px - cx, py - cy
    x = dx * np.cos(-angle) - dy * np.sin(-angle)
    y = dx * np.sin(-angle) + dy * np.cos(-angle)
    apothem = side * np.sqrt(3) / 2
    for i in range(6):
        a = np.pi / 6 + i * np.pi / 3
        if x * np.cos(a) + y * np.sin(a) > apothem + 1e-9:
            return False
    return True

def hexagon_fits_in_outer(cx_in, cy_in, side_in, angle_in, cx_out, cy_out, side_out, angle_out):
    """Check if inner hexagon fits inside outer hexagon."""
    inner_verts = hexagon_vertices(cx_in, cy_in, side_in, angle_in)
    for vx, vy in inner_verts:
        if not point_in_hexagon(vx, vy, cx_out, cy_out, side_out, angle_out):
            return False
    return True

def hexagons_disjoint(inner):
    """Check if all unit hexagons are pairwise disjoint."""
    n = len(inner)
    for i in range(n):
        for j in range(i+1, n):
            ix1, iy1, ia1 = inner[i]
            ix2, iy2, ia2 = inner[j]
            dx = ix1 - ix2
            dy = iy1 - iy2
            dist = np.sqrt(dx*dx + dy*dy)
            # For same orientation, min non-overlap distance is sqrt(3)
            # For different orientations, could be less, but for honeycomb they're same orientation
            if dist < np.sqrt(3) - 1e-9:
                # More careful check: see if any vertex is inside the other
                v1 = hexagon_vertices(ix1, iy1, 1.0, ia1)
                v2 = hexagon_vertices(ix2, iy2, 1.0, ia2)
                for vx, vy in v1:
                    if point_in_hexagon(vx, vy, ix2, iy2, 1.0, ia2):
                        return False
                for vx, vy in v2:
                    if point_in_hexagon(vx, vy, ix1, iy1, 1.0, ia1):
                        return False
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 100
    
    best_side = float('inf')
    best_result = None
    
    # Honeycomb arrangements for 11 hexagons
    arrangements = []
    
    # 3-3-3-2 pattern
    arr = []
    for k in range(3):
        arr.append((k * np.sqrt(3), 0.0, 0.0))
    for k in range(3):
        arr.append((k * np.sqrt(3) + np.sqrt(3)/2, 1.5, 0.0))
    for k in range(3):
        arr.append((k * np.sqrt(3), 3.0, 0.0))
    for k in range(2):
        arr.append((k * np.sqrt(3) + np.sqrt(3)/2, 4.5, 0.0))
    arrangements.append(arr)
    
    # 2-3-3-3 pattern
    arr = []
    for k in range(2):
        arr.append((k * np.sqrt(3) + np.sqrt(3)/2, 0.0, 0.0))
    for k in range(3):
        arr.append((k * np.sqrt(3), 1.5, 0.0))
    for k in range(3):
        arr.append((k * np.sqrt(3) + np.sqrt(3)/2, 3.0, 0.0))
    for k in range(3):
        arr.append((k * np.sqrt(3), 4.5, 0.0))
    arrangements.append(arr)
    
    # 3-4-4 pattern
    arr = []
    for k in range(3):
        arr.append((k * np.sqrt(3), 0.0, 0.0))
    for k in range(4):
        arr.append((k * np.sqrt(3) + np.sqrt(3)/2, 1.5, 0.0))
    for k in range(4):
        arr.append((k * np.sqrt(3), 3.0, 0.0))
    arrangements.append(arr)
    
    # 4-3-4 pattern
    arr = []
    for k in range(4):
        arr.append((k * np.sqrt(3), 0.0, 0.0))
    for k in range(3):
        arr.append((k * np.sqrt(3) + np.sqrt(3)/2, 1.5, 0.0))
    for k in range(4):
        arr.append((k * np.sqrt(3), 3.0, 0.0))
    arrangements.append(arr)
    
    # 4-4-3 pattern
    arr = []
    for k in range(4):
        arr.append((k * np.sqrt(3), 0.0, 0.0))
    for k in range(4):
        arr.append((k * np.sqrt(3) + np.sqrt(3)/2, 1.5, 0.0))
    for k in range(3):
        arr.append((k * np.sqrt(3), 3.0, 0.0))
    arrangements.append(arr)
    
    # 5-3-3 pattern
    arr = []
    for k in range(5):
        arr.append((k * np.sqrt(3), 0.0, 0.0))
    for k in range(3):
        arr.append((k * np.sqrt(3) + np.sqrt(3)/2, 1.5, 0.0))
    for k in range(3):
        arr.append((k * np.sqrt(3), 3.0, 0.0))
    arrangements.append(arr)
    
    # 3-5-3 pattern
    arr = []
    for k in range(3):
        arr.append((k * np.sqrt(3), 0.0, 0.0))
    for k in range(5):
        arr.append((k * np.sqrt(3) + np.sqrt(3)/2, 1.5, 0.0))
    for k in range(3):
        arr.append((k * np.sqrt(3), 3.0, 0.0))
    arrangements.append(arr)
    
    # 2-4-5 pattern
    arr = []
    for k in range(2):
        arr.append((k * np.sqrt(3) + np.sqrt(3)/2, 0.0, 0.0))
    for k in range(4):
        arr.append((k * np.sqrt(3), 1.5, 0.0))
    for k in range(5):
        arr.append((k * np.sqrt(3) + np.sqrt(3)/2, 3.0, 0.0))
    arrangements.append(arr)
    
    for arr in arrangements:
        if time.time() > deadline:
            break
        
        xs = [p[0] for p in arr]
        ys = [p[1] for p in arr]
        cx_arr = (min(xs) + max(xs)) / 2
        cy_arr = (min(ys) + max(ys)) / 2
        centered_arr = [(x - cx_arr, y - cy_arr, a) for (x, y, a) in arr]
        
        # Check if hexagons are disjoint
        if not hexagons_disjoint(centered_arr):
            continue
        
        # Try different outer hexagon rotations
        for outer_angle in range(0, 360, 3):
            if time.time() > deadline:
                break
            
            # Binary search for minimum side length
            lo, hi = 2.0, 20.0
            for _ in range(25):
                mid = (lo + hi) / 2
                valid = True
                for (ix, iy, ia) in centered_arr:
                    if not hexagon_fits_in_outer(ix, iy, 1.0, ia, 0, 0, mid, outer_angle):
                        valid = False
                        break
                if valid:
                    hi = mid
                else:
                    lo = mid
            
            if hi < best_side:
                inner = [[x, y, a] for (x, y, a) in centered_arr]
                # Final validation
                all_fit = True
                for (ix, iy, ia) in inner:
                    if not hexagon_fits_in_outer(ix, iy, 1.0, ia, 0, 0, hi, outer_angle):
                        all_fit = False
                        break
                if all_fit:
                    best_side = hi
                    best_result = (inner, (0.0, 0.0), hi, float(outer_angle))
    
    if best_result is not None:
        return best_result
    
    # Fallback
    inner = [[4.0 * k, 0.0, 0.0] for k in range(-(N // 2), N - N // 2)]
    return inner, (0.0, 0.0), 40.0, 0.0
# EVOLVE-BLOCK-END
