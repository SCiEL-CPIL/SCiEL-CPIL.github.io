import math
import time
import numpy as np
import random

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
    This is used for the penalty of inner hexagons being outside the outer one.
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

def _min_dist_sq_hex_hex(c1x, c1y, a1, c2x, c2y, a2):
    """
    Computes the minimum squared distance between two unit hexagons.
    If they overlap, returns 0.
    We check vertices of both hexagons against the other, and edge intersections.
    """
    v1 = _hex_vertices(c1x, c1y, 1.0, a1)
    v2 = _hex_vertices(c2x, c2y, 1.0, a2)
    
    min_d = float('inf')
    
    # Check if any vertex of 1 is inside 2
    for p in v1:
        d = _point_in_hex_dist_sq(p[0], p[1], c2x, c2y, 1.0, a2)
        if d == 0:
            return 0.0
        min_d = min(min_d, d)
        
    # Check if any vertex of 2 is inside 1
    for p in v2:
        d = _point_in_hex_dist_sq(p[0], p[1], c1x, c1y, 1.0, a1)
        if d == 0:
            return 0.0
        min_d = min(min_d, d)
        
    # Check edge intersections
    # If edges intersect, distance is 0.
    # We can use a simple segment intersection check.
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
                return 0.0

    return min_d

def _get_cluster_radius(params):
    """
    params: list of (x, y, angle) for inner hexagons.
    Returns the minimum side length of a regular hexagon centered at origin
    that contains all inner hexagons.
    """
    # The outer hexagon is centered at (0,0) and has rotation 0 for the radius calculation.
    # We need to find the smallest s such that all inner hexagons are inside.
    # A point (px, py) is inside a hexagon of side s centered at origin (rot 0) if:
    # |px| <= s * sqrt(3) / 2 * 2? No.
    # Apothem a = s * sqrt(3) / 2.
    # The hexagon is defined by 3 pairs of parallel lines.
    # For rotation 0 (flat top/bottom? or pointy top?):
    # Standard hex vertices for s=1, rot=0: (1,0), (0.5, sqrt(3)/2), (-0.5, sqrt(3)/2), (-1,0), (-0.5, -sqrt(3)/2), (0.5, -sqrt(3)/2).
    # The lines are:
    # 1. x = 1, x = -1  -> |x| <= s
    # 2. sqrt(3)/2 * x + y/2 = sqrt(3)/2 * 1? 
    # Let's use the support function approach.
    # The distance from center to edge (apothem) is a = s * sqrt(3) / 2.
    # The normal vectors to the edges for a hexagon with vertices at angles 0, 60, ... are at 30, 90, 150...
    # Wait, if vertices are at 0, 60, 120... the edges are perpendicular to 30, 90, 150...
    # Let's verify. Vertex (1,0). Next (0.5, sqrt(3)/2). Midpoint (0.75, sqrt(3)/4).
    # Normal to this edge? Vector is (-0.5, sqrt(3)/2). Normal is (sqrt(3)/2, 0.5) or (1/sqrt(3), 1/3)?
    # Angle of normal is 30 degrees.
    # So the constraints are:
    # For k in 0..2:
    #   proj = px * cos(30 + 60k) + py * sin(30 + 60k)
    #   proj <= apothem
    #   -proj <= apothem
    # apothem = s * sqrt(3) / 2
    
    max_proj = 0.0
    for (px, py, ang) in params:
        # We need to check all vertices of this inner hexagon
        verts = _hex_vertices(px, py, 1.0, ang)
        for (vx, vy) in verts:
            for k in range(3):
                theta = math.radians(30 + 60*k)
                proj = vx * math.cos(theta) + vy * math.sin(theta)
                abs_proj = abs(proj)
                if abs_proj > max_proj:
                    max_proj = abs_proj
                    
    # max_proj is the required apothem.
    # s * sqrt(3) / 2 = max_proj
    s = max_proj * 2 / math.sqrt(3)
    return s

def _objective(params):
    """
    params: flat list [x1, y1, a1, ..., x11, y11, a11, global_rot]
    Returns the side length of the bounding hexagon + penalty for overlaps.
    """
    inner_params = []
    for i in range(N):
        x = params[i*3]
        y = params[i*3+1]
        a = params[i*3+2]
        inner_params.append((x, y, a))
        
    global_rot = params[N*3]
    
    # Apply global rotation to all inner hexagons
    rotated_params = []
    for (x, y, a) in inner_params:
        # Rotate position
        rx = x * math.cos(math.radians(global_rot)) - y * math.sin(math.radians(global_rot))
        ry = x * math.sin(math.radians(global_rot)) + y * math.cos(math.radians(global_rot))
        # Rotate angle
        ra = a + global_rot
        rotated_params.append((rx, ry, ra))
        
    # Calculate bounding hexagon side length
    side = _get_cluster_radius(rotated_params)
    
    # Calculate overlap penalty
    penalty = 0.0
    for i in range(N):
        for j in range(i+1, N):
            d = _min_dist_sq_hex_hex(rotated_params[i][0], rotated_params[i][1], rotated_params[i][2], 
                                     rotated_params[j][0], rotated_params[j][1], rotated_params[j][2])
            if d == 0.0:
                penalty += 100.0
            else:
                # Small penalty if very close?
                # If d < 0.01**2, add penalty
                if d < 0.0001:
                    penalty += (0.0001 - d) * 1000
    
    return side + penalty

def construct():
    start_time = time.time()
    deadline = start_time + 145.0 # Safety margin
    
    # Initial guess: Lattice packing
    # Center
    init_inner = [(0.0, 0.0, 0.0)]
    # Ring 1
    for i in range(6):
        a = math.radians(60*i)
        x = 1.5 * math.cos(a)
        y = 1.5 * math.sin(a)
        init_inner.append((x, y, 0.0))
    # Ring 2 (4 more to make 11)
    # We need 4 more.
    # Let's pick positions that are roughly valid.
    # (3, 0), (-3, 0), (0, 2.6), (0, -2.6)
    init_inner.append((3.0, 0.0, 0.0))
    init_inner.append((-3.0, 0.0, 0.0))
    init_inner.append((0.0, 2.6, 0.0))
    init_inner.append((0.0, -2.6, 0.0))
    
    x0 = []
    for p in init_inner:
        x0.append(p[0])
        x0.append(p[1])
        x0.append(p[2])
    x0.append(0.0) # global_rot
    
    best_params = x0[:]
    best_val = _objective(best_params)
    
    # Simulated Annealing
    T = 1.0
    T_min = 1e-6
    alpha = 0.995
    step_size = 0.5
    
    current_params = x0[:]
    current_val = best_val
    
    dim = N * 3 + 1
    
    while time.time() < deadline:
        # Generate neighbor
        new_params = current_params[:]
        # Perturb a random subset of parameters
        for _ in range(5):
            idx = random.randint(0, dim - 1)
            delta = random.gauss(0, step_size)
            new_params[idx] += delta
            
        # Handle angle wrapping
        for i in range(N):
            if new_params[i*3+2] > 360:
                new_params[i*3+2] -= 360
            elif new_params[i*3+2] < 0:
                new_params[i*3+2] += 360
        if new_params[N*3] > 360:
            new_params[N*3] -= 360
        elif new_params[N*3] < 0:
            new_params[N*3] += 360
            
        new_val = _objective(new_params)
        
        # Acceptance criterion
        if new_val < current_val or random.random() < math.exp(-(new_val - current_val) / T):
            current_params = new_params
            current_val = new_val
            
            if current_val < best_val:
                best_val = current_val
                best_params = current_params[:]
                
        T *= alpha
        if T < T_min:
            T = T_min
            
    # Final polishing with local search (Nelder-Mead) if time permits
    if time.time() < deadline - 10:
        from scipy.optimize import minimize
        # Convert to a function for scipy
        def obj_scipy(x):
            return _objective(x)
            
        res = minimize(obj_scipy, best_params, method='Nelder-Mead', options={'maxiter': 50, 'xatol': 1e-4, 'fatol': 1e-4})
        if res.fun < best_val:
            best_val = res.fun
            best_params = res.x
            
    return _format_result(best_params)

def _format_result(params):
    inner = []
    for i in range(N):
        cx = params[i*3]
        cy = params[i*3+1]
        a = params[i*3+2]
        inner.append([cx, cy, a])
        
    # The outer hexagon is centered at (0,0) with side length best_val and rotation 0
    # Wait, the interface requires: inner, center, side, angle
    # The outer hexagon center is (0,0) in our local frame.
    # But we can shift the whole thing? No, the score is just the side length.
    # The evaluator places the outer hexagon at the given center and rotation.
    # Our inner hexagons are defined relative to the origin.
    # So we return outer center (0,0), side length, angle 0.
    
    # However, we might want to return the actual bounding box center if we shifted?
    # In our model, we forced the outer hexagon to be centered at (0,0).
    # So the inner hexagons are placed such that they fit in a hexagon centered at (0,0).
    # So we return center (0,0), side length, angle 0.
    
    return inner, (0.0, 0.0), best_val, 0.0

if __name__ == "__main__":
    r = construct()
    print(r)
