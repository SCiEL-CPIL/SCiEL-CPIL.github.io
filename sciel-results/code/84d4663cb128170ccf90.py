"""Packing 12 unit regular hexagons inside a regular hexagon to minimize outer side length."""
import numpy as np
from scipy.optimize import minimize
import time

N = 12

def hexagon_vertices(cx, cy, side, angle_deg):
    angle = np.radians(angle_deg)
    verts = []
    for i in range(6):
        theta = angle + i * np.pi / 3.0
        x = cx + side * np.cos(theta)
        y = cy + side * np.sin(theta)
        verts.append([x, y])
    return np.array(verts)

def point_in_polygon(pt, poly):
    n = len(poly)
    inside = False
    j = n - 1
    for i in range(n):
        xi, yi = poly[i]
        xj, yj = poly[j]
        if ((yi > pt[1]) != (yj > pt[1])) and \
           (pt[0] < (xj - xi) * (pt[1] - yi) / (yj - yi) + xi):
            inside = not inside
        j = i
    return inside

def segment_intersects(a1, a2, b1, b2):
    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])
    c1 = cross(a1, a2, b1)
    c2 = cross(a1, a2, b2)
    c3 = cross(b1, b2, a1)
    c4 = cross(b1, b2, a2)
    if ((c1 > 0 and c2 < 0) or (c1 < 0 and c2 > 0)) and \
       ((c3 > 0 and c4 < 0) or (c3 < 0 and c4 > 0)):
        return True
    return False

def hexagons_overlap(v1, v2, eps=1e-9):
    for pt in v1:
        if point_in_polygon(pt, v2):
            return True
    for pt in v2:
        if point_in_polygon(pt, v1):
            return True
    n1, n2 = len(v1), len(v2)
    for i in range(n1):
        for j in range(n2):
            if segment_intersects(v1[i], v1[(i+1)%n1], v2[j], v2[(j+1)%n2]):
                return True
    return False

def min_enclosing_hexagon_side(verts, angle=0.0):
    """Find minimum side length of regular hexagon with given angle containing all vertices."""
    # For a regular hexagon with side S and angle A, the support function in direction theta is:
    # h(theta) = S * max over vertices of dot(vertex_dir, direction)
    # Actually, for a regular hexagon centered at origin with side S and rotation A,
    # the vertices are at S*(cos(A + k*pi/3), sin(A + k*pi/3)) for k=0..5
    # The support function in direction (cos(phi), sin(phi)) is S * max_k cos(phi - A - k*pi/3)
    # For the hexagon to contain all points, for each point p, we need:
    # For all 6 directions, the projection of p onto the normal of each edge <= S * cos(0) = S
    # Actually, the distance from center to each edge (inradius) = S * sqrt(3)/2
    # So for each of the 6 edge normals, |dot(p - center, normal)| <= S * sqrt(3)/2
    # To find minimum S for given center and angle, we maximize over all points and all 6 normals
    # of |dot(p - center, normal)|, then S = max_val / (sqrt(3)/2)
    
    angle_rad = np.radians(angle)
    normals = []
    for k in range(6):
        theta = angle_rad + k * np.pi / 3.0 + np.pi / 6.0  # Edge normals are offset by 30 degrees
        normals.append(np.array([np.cos(theta), np.sin(theta)]))
    
    max_proj = 0.0
    for p in verts:
        for n in normals:
            proj = abs(np.dot(p, n))
            if proj > max_proj:
                max_proj = proj
    
    return max_proj / (np.sqrt(3) / 2.0)

def initial_configuration():
    dx = np.sqrt(3)
    dy = 1.5
    
    positions = []
    # Row 0: 3 hexagons
    for i in range(3):
        x = (i - 1) * dx
        y = 0
        positions.append((x, y, 0.0))
    # Row 1: 4 hexagons
    for i in range(4):
        x = (i - 1.5) * dx
        y = -dy
        positions.append((x, y, 0.0))
    # Row 2: 3 hexagons
    for i in range(3):
        x = (i - 1) * dx
        y = -2*dy
        positions.append((x, y, 0.0))
    # Row 3: 2 hexagons
    for i in range(2):
        x = (i - 0.5) * dx
        y = -3*dy
        positions.append((x, y, 0.0))
    
    xs = [p[0] for p in positions]
    ys = [p[1] for p in positions]
    cx = (min(xs) + max(xs)) / 2
    cy = (min(ys) + max(ys)) / 2
    
    inner = [[p[0] - cx, p[1] - cy, p[2]] for p in positions]
    
    # Compute all vertices
    all_verts = []
    for x, y, a in inner:
        verts = hexagon_vertices(x, y, 1.0, a)
        all_verts.extend(verts.tolist())
    all_verts = np.array(all_verts)
    
    # Find optimal outer hexagon angle and side
    best_side = np.inf
    best_angle = 0.0
    for angle in np.arange(0, 60, 5):
        s = min_enclosing_hexagon_side(all_verts, angle)
        if s < best_side:
            best_side = s
            best_angle = angle
    
    return inner, (0.0, 0.0), best_side, best_angle

def construct():
    start_time = time.time()
    deadline = start_time + 115.0
    
    inner, center, side, angle = initial_configuration()
    best_side = side
    best_inner = inner
    best_center = center
    best_angle = angle
    
    # Try different arrangements and optimize
    arrangements = []
    
    # Arrangement 1: 3-4-3-2 (already in initial)
    # Arrangement 2: 2-3-4-3
    dx = np.sqrt(3)
    dy = 1.5
    
    pos2 = []
    for i in range(2):
        pos2.append(((i - 0.5) * dx, 0, 0.0))
    for i in range(3):
        pos2.append(((i - 1) * dx, -dy, 0.0))
    for i in range(4):
        pos2.append(((i - 1.5) * dx, -2*dy, 0.0))
    for i in range(3):
        pos2.append(((i - 1) * dx, -3*dy, 0.0))
    arrangements.append(pos2)
    
    # Arrangement 3: 4-3-3-2
    pos3 = []
    for i in range(4):
        pos3.append(((i - 1.5) * dx, 0, 0.0))
    for i in range(3):
        pos3.append(((i - 1) * dx, -dy, 0.0))
    for i in range(3):
        pos3.append(((i - 1) * dx, -2*dy, 0.0))
    for i in range(2):
        pos3.append(((i - 0.5) * dx, -3*dy, 0.0))
    arrangements.append(pos3)
    
    # Arrangement 4: Hexagonal cluster (1+6+5)
    pos4 = [(0, 0, 0.0)]
    for k in range(6):
        theta = k * np.pi / 3.0
        pos4.append((dx * np.cos(theta), dx * np.sin(theta), 0.0))
    # Add 5 more around the second ring
    for k in range(5):
        theta = k * 2 * np.pi / 5.0 + np.pi/5
        r = 2 * dx
        pos4.append((r * np.cos(theta), r * np.sin(theta), 0.0))
    arrangements.append(pos4)
    
    # Process each arrangement
    all_arrangements = [initial_configuration()]
    for pos in arrangements:
        xs = [p[0] for p in pos]
        ys = [p[1] for p in pos]
        cx = (min(xs) + max(xs)) / 2
        cy = (min(ys) + max(ys)) / 2
        inner_arr = [[p[0] - cx, p[1] - cy, p[2]] for p in pos]
        
        all_verts = []
        for x, y, a in inner_arr:
            verts = hexagon_vertices(x, y, 1.0, a)
            all_verts.extend(verts.tolist())
        all_verts = np.array(all_verts)
        
        best_s = np.inf
        best_a = 0.0
        for angle in np.arange(0, 60, 3):
            s = min_enclosing_hexagon_side(all_verts, angle)
            if s < best_s:
                best_s = s
                best_a = angle
        
        all_arrangements.append((inner_arr, (0.0, 0.0), best_s, best_a))
    
    # Find best
    for arr in all_arrangements:
        if arr[2] < best_side:
            best_side = arr[2]
            best_inner = arr[0]
            best_center = arr[1]
            best_angle = arr[3]
    
    # Try local optimization on positions
    if time.time() < deadline - 60:
        def penalty_function(params):
            n = 12
            inner_p = params[:3*n]
            outer_p = params[3*n:]
            cx, cy, S, A = outer_p
            
            if S < 1.0:
                return 1e10
            
            outer_verts = hexagon_vertices(cx, cy, S, A)
            penalty = 0.0
            
            inner_verts_list = []
            for i in range(n):
                x, y, a = inner_p[3*i], inner_p[3*i+1], inner_p[3*i+2]
                verts = hexagon_vertices(x, y, 1.0, a)
                inner_verts_list.append(verts)
                for pt in verts:
                    if not point_in_polygon(pt, outer_verts):
                        penalty += 50.0
            
            for i in range(n):
                for j in range(i+1, n):
                    if hexagons_overlap(inner_verts_list[i], inner_verts_list[j]):
                        penalty += 50.0
            
            return penalty
        
        # Optimize the best arrangement
        p0 = []
        for x, y, a in best_inner:
            p0.extend([x, y, a])
        p0.extend([best_center[0], best_center[1], best_side, best_angle])
        p0 = np.array(p0)
        
        try:
            result = minimize(penalty_function, p0, method='Nelder-Mead',
                            options={'maxiter': 300, 'maxfev': 1500, 'xatol': 1e-3, 'fatol': 1e-3})
            if result.fun < 1.0:
                final_p = result.x
                final_inner = []
                for i in range(12):
                    final_inner.append([final_p[3*i], final_p[3*i+1], final_p[3*i+2]])
                final_side = final_p[3*12+2]
                if final_side < best_side:
                    best_side = final_side
                    best_inner = final_inner
                    best_center = (final_p[3*12], final_p[3*12+1])
                    best_angle = final_p[3*12+3]
        except Exception:
            pass
    
    return best_inner, best_center, best_side, best_angle

# EVOLVE-BLOCK-START
# EVOLVE-BLOCK-END
