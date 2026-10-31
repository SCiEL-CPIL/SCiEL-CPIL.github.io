"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np

N = 11

# EVOLVE-BLOCK-START
def construct():
    sqrt3 = np.sqrt(3)
    
    def compute_bounding_side(inner, outer_center, outer_angle_deg=0.0):
        """Compute minimum side length of outer hexagon containing all inner hexagons."""
        ocx, ocy = outer_center
        theta_outer = np.radians(outer_angle_deg)
        
        # Outer hexagon edge outward normals at angles theta_outer + 30° + k*60°
        r_inner_max = 1.0  # circumradius of unit hexagon (conservative)
        apothem_factor = sqrt3 / 2.0
        
        max_val = 0.0
        for cx, cy, _ in inner:
            dx = cx - ocx
            dy = cy - ocy
            for k in range(6):
                angle = theta_outer + np.pi/6 + k * np.pi / 3.0
                nx = np.cos(angle)
                ny = np.sin(angle)
                proj = dx * nx + dy * ny
                val = (proj + r_inner_max) / apothem_factor
                if val > max_val:
                    max_val = val
        
        return max(1.0, max_val)
    
    def axial_to_pixel(q, r):
        x = sqrt3 * (q + r / 2.0)
        y = 1.5 * r
        return x, y
    
    # Generate candidate arrangements
    candidates = []
    
    # Various hexagonal cluster shapes for 11 hexagons
    
    # Shape 1: Rows 3,3,3,2
    arr = []
    for r, qs in [(0, [-1,0,1]), (1, [-1,0,1]), (2, [-1,0,1]), (3, [-1,0])]:
        for q in qs:
            x, y = axial_to_pixel(q, r)
            arr.append([x, y, 0.0])
    candidates.append(arr)
    
    # Shape 2: Rows 2,3,3,3
    arr = []
    for r, qs in [(0, [0,1]), (1, [-1,0,1]), (2, [-1,0,1]), (3, [-1,0,1])]:
        for q in qs:
            x, y = axial_to_pixel(q, r)
            arr.append([x, y, 0.0])
    candidates.append(arr)
    
    # Shape 3: Rows 2,3,4,2
    arr = []
    for r, qs in [(0, [-1,0]), (1, [-1,0,1]), (2, [-2,-1,0,1]), (3, [-1,0])]:
        for q in qs:
            x, y = axial_to_pixel(q, r)
            arr.append([x, y, 0.0])
    candidates.append(arr)
    
    # Shape 4: Rows 3,4,3,1
    arr = []
    for r, qs in [(0, [-1,0,1]), (1, [-2,-1,0,1]), (2, [-1,0,1]), (3, [0])]:
        for q in qs:
            x, y = axial_to_pixel(q, r)
            arr.append([x, y, 0.0])
    candidates.append(arr)
    
    # Shape 5: Compact - center + 6 + 4 in one direction
    arr = []
    # Center
    x, y = axial_to_pixel(0, 0)
    arr.append([x, y, 0.0])
    # 6 around
    for q, r in [(1,0),(1,-1),(0,-1),(-1,0),(-1,1),(0,1)]:
        x, y = axial_to_pixel(q, r)
        arr.append([x, y, 0.0])
    # 4 more: extend in +x direction
    for q, r in [(2,0),(2,-1),(2,-2),(3,-2)]:
        x, y = axial_to_pixel(q, r)
        arr.append([x, y, 0.0])
    candidates.append(arr)
    
    # Shape 6: Symmetric extension
    arr = []
    # 7-hex cluster
    for q, r in [(0,0),(1,0),(1,-1),(0,-1),(-1,0),(-1,1),(0,1)]:
        x, y = axial_to_pixel(q, r)
        arr.append([x, y, 0.0])
    # Add 4: two on each side
    for q, r in [(2,0),(2,-1),(-2,0),(-2,1)]:
        x, y = axial_to_pixel(q, r)
        arr.append([x, y, 0.0])
    candidates.append(arr)
    
    best_inner = None
    best_center = None
    best_side = float('inf')
    best_angle = 0.0
    
    for arr in candidates:
        if len(arr) != N:
            continue
        
        centers = [(p[0], p[1]) for p in arr]
        cx_mean = np.mean([c[0] for c in centers])
        cy_mean = np.mean([c[1] for c in centers])
        
        # Search over outer center positions
        for dx in np.linspace(-1.5, 1.5, 11):
            for dy in np.linspace(-1.5, 1.5, 11):
                outer_center = (cx_mean + dx, cy_mean + dy)
                side = compute_bounding_side(arr, outer_center, 0.0)
                if side < best_side:
                    best_side = side
                    best_inner = [list(p) for p in arr]
                    best_center = outer_center
                    best_angle = 0.0
    
    if best_inner is None or best_side > 10:
        best_inner = [[4.0 * k, 0.0, 0.0] for k in range(-(N // 2), N - N // 2)]
        best_center = (0.0, 0.0)
        best_side = 40.0
        best_angle = 0.0
    
    return best_inner, best_center, best_side, best_angle
# EVOLVE-BLOCK-END
