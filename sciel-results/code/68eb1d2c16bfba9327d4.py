import math
import time
import numpy as np
from scipy.optimize import differential_evolution, minimize

N = 11

def _hex_vertices(cx, cy, s, angle_deg):
    """
    Returns vertices of a regular hexagon centered at (cx, cy) with side s
    and rotation angle_deg (degrees).
    """
    a = math.radians(angle_deg)
    verts = []
    for i in range(6):
        theta = a + i * math.pi / 3
        x = cx + s * math.cos(theta)
        y = cy + s * math.sin(theta)
        verts.append((x, y))
    return verts

def _point_in_hex_dist_sq(px, py, cx, cy, s, angle_deg):
    """
    Returns squared distance from point (px, py) to the hexagon if outside,
    or 0 if inside.
    """
    a = math.radians(angle_deg)
    dx = px - cx
    dy = py - cy
    # Rotate by -a to align hexagon with axes
    rx = dx * math.cos(-a) - dy * math.sin(-a)
    ry = dx * math.sin(-a) + dy * math.cos(-a)
    
    apothem = s * math.sqrt(3) / 2
    sum_sq_viol = 0.0
    for i in range(3):
        theta_n = math.radians(30 + i * 60)
        proj = rx * math.cos(theta_n) + ry * math.sin(theta_n)
        if proj > apothem:
            sum_sq_viol += (proj - apothem)**2
        if proj < -apothem:
            sum_sq_viol += (-apothem - proj)**2
    return sum_sq_viol

def _pairwise_penalty(p1, p2):
    """
    p1, p2: (cx, cy, angle_deg)
    Returns penalty if they overlap.
    """
    c1x, c1y, a1 = p1
    c2x, c2y, a2 = p2
    
    dx = c1x - c2x
    dy = c1y - c2y
    if dx*dx + dy*dy > 4.0:
        return 0.0
        
    v1 = _hex_vertices(c1x, c1y, 1.0, a1)
    v2 = _hex_vertices(c2x, c2y, 1.0, a2)
    
    penalty = 0.0
    
    for pt in v1:
        penalty += _point_in_hex_dist_sq(pt[0], pt[1], c2x, c2y, 1.0, a2)
        
    for pt in v2:
        penalty += _point_in_hex_dist_sq(pt[0], pt[1], c1x, c1y, 1.0, a1)
        
    if penalty == 0.0:
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
            p1 = v1[i]
            p2 = v1[(i+1)%6]
            for j in range(6):
                p3 = v2[j]
                p4 = v2[(j+1)%6]
                if segments_intersect(p1, p2, p3, p4):
                    mid1 = ((p1[0]+p2[0])/2, (p1[1]+p2[1])/2)
                    mid2 = ((p3[0]+p4[0])/2, (p3[1]+p4[1])/2)
                    penalty += _point_in_hex_dist_sq(mid1[0], mid1[1], c2x, c2y, 1.0, a2)
                    penalty += _point_in_hex_dist_sq(mid2[0], mid2[1], c1x, c1y, 1.0, a1)
                    if penalty == 0.0:
                        penalty += 1.0
                
    return penalty

def _outer_penalty(inner_params, outer_params):
    """
    inner_params: list of (cx, cy, angle)
    outer_params: (cx, cy, s, angle)
    Returns penalty if any inner hexagon is not fully inside the outer hexagon.
    """
    ocx, ocy, os, oang = outer_params
    
    penalty = 0.0
    for ip in inner_params:
        icx, icy, iang = ip
        v_in = _hex_vertices(icx, icy, 1.0, iang)
        for pt in v_in:
            penalty += _point_in_hex_dist_sq(pt[0], pt[1], ocx, ocy, os, oang)
            
    return penalty

def _get_lattice_positions():
    """
    Returns 11 positions in a hexagonal lattice configuration.
    Center + 6 neighbors + 4 from ring 2.
    """
    positions = []
    # Center
    positions.append((0.0, 0.0))
    
    # Ring 1 (6 neighbors)
    # Distance between centers in hex lattice is 2 * apothem = sqrt(3)
    dist = math.sqrt(3)
    for i in range(6):
        a = math.radians(60 * i)
        x = dist * math.cos(a)
        y = dist * math.sin(a)
        positions.append((x, y))
        
    # Ring 2 (4 selected to maintain symmetry and compactness)
    # Ring 2 distance is 2 * sqrt(3)
    dist2 = 2 * math.sqrt(3)
    # Angles: 0, 60, 120, 180, 240, 300
    # Let's pick 0, 120, 180, 300? Or 0, 60, 180, 240?
    # Actually, let's pick 4 that are roughly symmetric.
    # 0, 120, 240 (3-fold symmetry) + one more?
    # Or 0, 60, 180, 240?
    # Let's try 0, 120, 180, 300.
    angles_to_add = [0, 120, 180, 300]
    for deg in angles_to_add:
        a = math.radians(deg)
        x = dist2 * math.cos(a)
        y = dist2 * math.sin(a)
        positions.append((x, y))
        
    return positions

def _calc_min_outer_side(inner_positions, inner_angle):
    """
    Given fixed inner positions and angle, find the minimum outer hexagon
    side length and its optimal center/rotation.
    This is done by sweeping outer rotation and finding the min enclosing hex.
    """
    # For a fixed set of points, the minimum enclosing hexagon side length
    # is determined by the maximum distance from the center to any point
    # projected onto the 3 axes of the hexagon.
    
    # We can optimize the outer hexagon's center (cx, cy) and angle.
    # Let's fix the outer angle to 0 for now and just find center?
    # No, the outer hexagon can rotate.
    
    # Simplification: Assume outer hexagon is aligned with inner lattice (angle 0).
    # Then we just need to find the center (cx, cy) that minimizes the max projection.
    
    # Actually, let's do a small local optimization for outer cx, cy, angle.
    # But for speed, let's just compute the bounding box in the hex coordinate system.
    
    # If we assume outer angle = 0:
    # The constraints are:
    # |x - cx| <= s * 1.5  (Wait, vertex is at s, apothem is s*sqrt(3)/2)
    # For a hexagon with side s, the width is 2s.
    # The projection on the axis (0 deg) is in [-s, s].
    # The projection on the axis (60 deg) is in [-s*sqrt(3)/2 * 2 / sqrt(3)? No.
    
    # Let's use the support function approach.
    # For a hexagon centered at origin with side s and angle 0:
    # h(u) = s * max( cos(theta_i - u) ) where theta_i are vertex angles?
    # No, h(u) for hexagon is piecewise linear.
    
    # Let's just use a simple grid search for outer center and angle.
    # Since N=11, the cluster is small.
    
    # Let's define the objective for outer params given fixed inner.
    def outer_obj(x):
        ocx, ocy, os, oang = x
        pen = 0.0
        for px, py in inner_positions:
            v_in = _hex_vertices(px, py, 1.0, inner_angle)
            for pt in v_in:
                pen += _point_in_hex_dist_sq(pt[0], pt[1], ocx, ocy, os, oang)
        return os + 100.0 * pen

    # Initial guess: center of mass, side = 5, angle = 0
    cx0 = sum(p[0] for p in inner_positions) / len(inner_positions)
    cy0 = sum(p[1] for p in inner_positions) / len(inner_positions)
    
    bounds = [(-10, 10), (-10, 10), (3.0, 10.0), (0, 360)]
    
    # Use Nelder-Mead for speed
    res = minimize(outer_obj, [cx0, cy0, 4.0, 0.0], method='Nelder-Mead', 
                   options={'maxiter': 200, 'xatol': 1e-4, 'fatol': 1e-4})
    
    return res.fun, res.x

def construct():
    start_time = time.time()
    deadline = start_time + 140.0

    # Strategy:
    # 1. Define lattice positions.
    # 2. Optimize global rotation of the cluster (inner_angle).
    # 3. For each angle, find optimal outer hex.
    # 4. Keep the best.

    base_positions = _get_lattice_positions()
    
    best_score = float('inf')
    best_params = None
    best_outer = None
    
    # Sweep inner angle
    # The lattice has 6-fold symmetry, so we only need to check 0 to 60 degrees.
    # Step size 1 degree.
    
    for deg in range(0, 61, 1):
        if time.time() > deadline - 10.0:
            break
            
        # Rotate base positions by deg
        a_rad = math.radians(deg)
        cos_a = math.cos(a_rad)
        sin_a = math.sin(a_rad)
        
        rotated_positions = []
        for x, y in base_positions:
            rx = x * cos_a - y * sin_a
            ry = x * sin_a + y * cos_a
            rotated_positions.append((rx, ry))
            
        # Find best outer hex for this configuration
        score, outer_x = _calc_min_outer_side(rotated_positions, deg)
        
        if score < best_score:
            best_score = score
            best_params = rotated_positions
            best_outer = outer_x
            best_inner_angle = deg

    # Format result
    if best_params is None:
        # Fallback
        best_params = base_positions
        best_inner_angle = 0
        best_outer = [0, 0, 5.0, 0.0]
        
    inner = []
    for x, y in best_params:
        inner.append([x, y, best_inner_angle])
        
    ocx, ocy, os, oang = best_outer
    
    return inner, (ocx, ocy), os, oang

if __name__ == "__main__":
    r = construct()
    print(r)
