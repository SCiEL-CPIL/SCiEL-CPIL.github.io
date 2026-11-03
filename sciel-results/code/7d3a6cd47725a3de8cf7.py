import numpy as np
import time
import random

N = 11
SPACING = np.sqrt(3)  # Minimum center-to-center distance for same-orientation unit hexagons

def hex_vertices(cx, cy, side=1.0, angle_deg=0.0):
    angle = np.radians(angle_deg)
    angles = angle + np.radians(np.arange(0, 360, 60))
    return np.column_stack([cx + side * np.cos(angles), cy + side * np.sin(angles)])

def point_in_hexagon(px, py, cx, cy, side, angle_deg=0.0):
    angle = np.radians(angle_deg)
    dx, dy = px - cx, py - cy
    cos_a, sin_a = np.cos(-angle), np.sin(-angle)
    rx = dx * cos_a - dy * sin_a
    ry = dx * sin_a + dy * cos_a
    inradius = side * np.sqrt(3) / 2
    for i in range(6):
        edge_angle = np.radians(30 + i * 60)
        nx, ny = np.cos(edge_angle), np.sin(edge_angle)
        if rx * nx + ry * ny > inradius:
            return False
    return True

def check_packing(inner, outer_center, outer_side, outer_angle):
    """Assumes all inner hexagons have angle 0."""
    ox, oy = outer_center
    n = len(inner)
    
    # Check non-overlap: pairwise distance >= SPACING
    for i in range(n):
        for j in range(i+1, n):
            dx = inner[i][0] - inner[j][0]
            dy = inner[i][1] - inner[j][1]
            if dx*dx + dy*dy < SPACING*SPACING - 1e-9:
                return False
    
    # Check containment
    for (ix, iy, iangle) in inner:
        verts = hex_vertices(ix, iy, 1.0, iangle)
        for vx, vy in verts:
            if not point_in_hexagon(vx, vy, ox, oy, outer_side, outer_angle):
                return False
    return True

def construct():
    t0 = time.time()
    deadline = t0 + 140
    
    # Initial placement: hexagonal grid
    axial_coords = [(0,0)]
    ring1 = [(1,0), (1,-1), (0,-1), (-1,0), (-1,1), (0,1)]
    axial_coords.extend(ring1)
    # Pick 4 from second ring for symmetry
    axial_coords.extend([(2,0), (0,-2), (-2,0), (0,2)])
    
    positions = []
    for q, r in axial_coords:
        x = SPACING * (q + r/2.0)
        y = SPACING * 1.5 * r
        positions.append((x, y))
    
    xs = [p[0] for p in positions]
    ys = [p[1] for p in positions]
    cx = (min(xs) + max(xs)) / 2
    cy = (min(ys) + max(ys)) / 2
    positions = [(x - cx, y - cy) for x, y in positions]
    
    inner = [[x, y, 0.0] for x, y in positions]
    
    # Find minimal enclosing hexagon
    best_side = 10.0
    best_center = (0.0, 0.0)
    best_angle = 0.0
    
    for angle_deg in [0, 30, 60, 90, 120, 150]:
        for ox in np.linspace(-1, 1, 5):
            for oy in np.linspace(-1, 1, 5):
                lo, hi = 1.0, 10.0
                for _ in range(40):
                    mid = (lo + hi) / 2
                    if check_packing(inner, (ox, oy), mid, angle_deg):
                        hi = mid
                    else:
                        lo = mid
                if hi < best_side and check_packing(inner, (ox, oy), hi, angle_deg):
                    best_side = hi
                    best_center = (ox, oy)
                    best_angle = angle_deg
    
    if time.time() > deadline:
        return inner, best_center, best_side, best_angle
    
    # Local search
    random.seed(42)
    current_inner = [list(p) for p in inner]
    current_side = best_side
    current_center = best_center
    current_angle = best_angle
    
    for iteration in range(20000):
        if time.time() > deadline:
            break
        
        idx = random.randint(0, N-1)
        old = list(current_inner[idx])
        current_inner[idx][0] += random.gauss(0, 0.1)
        current_inner[idx][1] += random.gauss(0, 0.1)
        
        if check_packing(current_inner, current_center, current_side, current_angle):
            # Try to shrink outer hexagon
            lo, hi = max(1.0, current_side * 0.97), current_side
            for _ in range(20):
                mid = (lo + hi) / 2
                if check_packing(current_inner, current_center, mid, current_angle):
                    hi = mid
                else:
                    lo = mid
            if hi < current_side and check_packing(current_inner, current_center, hi, current_angle):
                current_side = hi
        else:
            current_inner[idx] = old
    
    if check_packing(current_inner, current_center, current_side, current_angle):
        return current_inner, current_center, current_side, current_angle
    else:
        return inner, best_center, best_side, best_angle
