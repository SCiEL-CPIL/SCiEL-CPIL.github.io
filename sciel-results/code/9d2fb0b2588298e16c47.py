import numpy as np
from scipy.optimize import minimize
import time

N = 12
_START_TIME = time.time()
_DEADLINE = 140.0  # safety margin before 150s kill

def _hex_vertices(cx, cy, side, angle_deg):
    """Return 6 vertices of a regular hexagon with given center, side, rotation."""
    a = np.radians(angle_deg)
    angles = a + np.arange(6) * np.pi / 3
    return np.column_stack([cx + side * np.cos(angles), cy + side * np.sin(angles)])

def _hex_normals(angle_deg):
    """Return 6 outward normal vectors of a regular hexagon with given rotation."""
    a = np.radians(angle_deg)
    # Normals are at 30, 90, 150, 210, 270, 330 degrees relative to hex angle
    # Wait, if vertex is at angle a, the normal to the edge between vertex a and a+60 is at a+30.
    normal_angles = a + np.pi/6 + np.arange(6) * np.pi / 3
    return np.column_stack([np.cos(normal_angles), np.sin(normal_angles)])

def _project_vertices(verts, normal):
    """Project vertices onto a normal vector."""
    return np.dot(verts, normal)

def _hex_min_distance(hex1, hex2):
    """
    Calculate the minimum distance between two hexagons.
    Returns negative value if they overlap.
    hex1, hex2: tuples of (cx, cy, angle_deg)
    """
    cx1, cy1, ang1 = hex1
    cx2, cy2, ang2 = hex2
    
    v1 = _hex_vertices(cx1, cy1, 1.0, ang1)
    v2 = _hex_vertices(cx2, cy2, 1.0, ang2)
    
    n1 = _hex_normals(ang1)
    n2 = _hex_normals(ang2)
    
    min_dist = np.inf
    
    # Check all 12 axes (6 from each hexagon)
    all_normals = np.vstack([n1, n2])
    
    for n in all_normals:
        p1 = _project_vertices(v1, n)
        p2 = _project_vertices(v2, n)
        
        min1 = np.min(p1)
        max1 = np.max(p1)
        min2 = np.min(p2)
        max2 = np.max(p2)
        
        # Calculate gap
        # If intervals overlap, gap is negative (penetration depth)
        if max1 < min2:
            gap = min2 - max1
        elif max2 < min1:
            gap = min1 - max2
        else:
            gap = 0.0 # Overlapping on this axis
        
        if gap < min_dist:
            min_dist = gap
            
    return min_dist

def _outer_side_needed(inner, outer_center, outer_angle):
    """
    Calculate the minimum side length of the outer hexagon to contain all inner hexagons.
    inner: list of (cx, cy, angle_deg)
    outer_center: (cx, cy)
    outer_angle: float
    """
    ocx, ocy = outer_center
    oang = np.radians(outer_angle)
    
    # Outer hexagon normals
    outer_normals = _hex_normals(outer_angle)
    
    max_proj = 0.0
    
    for (cx, cy, ang) in inner:
        verts = _hex_vertices(cx, cy, 1.0, ang)
        # Translate vertices to outer center frame
        verts_local = verts - np.array([ocx, ocy])
        
        # Project onto each outer normal
        projs = np.dot(verts_local, outer_normals.T)
        
        # For each normal, we need the max projection
        # The constraint is max_proj <= S * sqrt(3) / 2
        # So S >= max_proj / (sqrt(3) / 2)
        # We take the max over all normals and all vertices
        max_p = np.max(projs)
        if max_p > max_proj:
            max_proj = max_p
            
    return max_proj * 2.0 / np.sqrt(3)

def _collision_penalty(inner):
    """Calculate smooth penalty for collisions between inner hexagons."""
    penalty = 0.0
    n = len(inner)
    for i in range(n):
        for j in range(i + 1, n):
            dist = _hex_min_distance(inner[i], inner[j])
            if dist < 0:
                # Smooth penalty: square of penetration depth
                penalty += (dist * dist) * 10.0
    return penalty

def construct():
    global _START_TIME
    t0 = time.time()
    
    # Initial configuration: Honeycomb
    inner = []
    for row in range(3):
        for col in range(4):
            x = col * 1.5
            y = row * np.sqrt(3)
            if row % 2 == 1:
                x += 0.75
            inner.append([x, y, 0.0])
            
    # Center
    xs = [p[0] for p in inner]
    ys = [p[1] for p in inner]
    cx = (min(xs) + max(xs)) / 2
    cy = (min(ys) + max(ys)) / 2
    for i in range(len(inner)):
        inner[i][0] -= cx
        inner[i][1] -= cy
        
    outer_center = (0.0, 0.0)
    outer_angle = 0.0
    outer_side = _outer_side_needed(inner, outer_center, outer_angle)
    
    best_side = outer_side
    best_inner = [list(x) for x in inner]
    best_center = outer_center
    best_angle = outer_angle
    
    def objective(params):
        if time.time() - t0 > _DEADLINE - 5:
            return best_side
            
        inner_x = params[0:12]
        inner_y = params[12:24]
        inner_ang = params[24:36]
        ocx = params[36]
        ocy = params[37]
        oang = params[38]
        
        inner_list = [(inner_x[i], inner_y[i], inner_ang[i]) for i in range(12)]
        
        pen = _collision_penalty(inner_list)
        if pen > 1.0:
            return pen + best_side
            
        side = _outer_side_needed(inner_list, (ocx, ocy), oang)
        return side + pen
        
    init_params = np.zeros(39)
    for i in range(12):
        init_params[i] = inner[i][0]
        init_params[12+i] = inner[i][1]
        init_params[24+i] = inner[i][2]
    init_params[36] = 0.0
    init_params[37] = 0.0
    init_params[38] = 0.0
    
    try:
        result = minimize(objective, init_params, method='L-BFGS-B',
                         options={'maxiter': 1000, 'ftol': 1e-8, 'gtol': 1e-8})
        
        res_inner = [(result.x[i], result.x[12+i], result.x[24+i]) for i in range(12)]
        res_center = (result.x[36], result.x[37])
        res_angle = result.x[38]
        
        # Check validity
        valid = True
        for i in range(12):
            for j in range(i+1, 12):
                if _hex_min_distance(res_inner[i], res_inner[j]) < -1e-5:
                    valid = False
                    break
            if not valid: break
            
        if valid:
            res_side = _outer_side_needed(res_inner, res_center, res_angle)
            if res_side < best_side:
                best_side = res_side
                best_inner = res_inner
                best_center = res_center
                best_angle = res_angle
    except Exception:
        pass
        
    # Polish with Nelder-Mead if time permits
    if time.time() - t0 < _DEADLINE - 20:
        try:
            init_params_nm = np.zeros(39)
            for i in range(12):
                init_params_nm[i] = best_inner[i][0]
                init_params_nm[12+i] = best_inner[i][1]
                init_params_nm[24+i] = best_inner[i][2]
            init_params_nm[36] = best_center[0]
            init_params_nm[37] = best_center[1]
            init_params_nm[38] = best_angle
            
            result_nm = minimize(objective, init_params_nm, method='Nelder-Mead',
                               options={'maxiter': 500, 'xatol': 1e-5, 'fatol': 1e-5})
                               
            res_inner_nm = [(result_nm.x[i], result_nm.x[12+i], result_nm.x[24+i]) for i in range(12)]
            res_center_nm = (result_nm.x[36], result_nm.x[37])
            res_angle_nm = result_nm.x[38]
            
            valid_nm = True
            for i in range(12):
                for j in range(i+1, 12):
                    if _hex_min_distance(res_inner_nm[i], res_inner_nm[j]) < -1e-5:
                        valid_nm = False
                        break
                if not valid_nm: break
                
            if valid_nm:
                res_side_nm = _outer_side_needed(res_inner_nm, res_center_nm, res_angle_nm)
                if res_side_nm < best_side:
                    best_side = res_side_nm
                    best_inner = res_inner_nm
                    best_center = res_center_nm
                    best_angle = res_angle_nm
        except Exception:
            pass
            
    # Final verification and safety margin
    if not _check_disjoint(best_inner):
        # Fallback
        best_inner = [list(x) for x in inner]
        best_center = (0.0, 0.0)
        best_angle = 0.0
        best_side = _outer_side_needed(best_inner, best_center, best_angle)
        
    if not _all_inside(best_inner, best_center, best_side, best_angle):
        best_side *= 1.01
        
    return best_inner, best_center, best_side, best_angle

def _check_disjoint(inner):
    n = len(inner)
    for i in range(n):
        for j in range(i + 1, n):
            if _hex_min_distance(inner[i], inner[j]) < -1e-5:
                return False
    return True

def _all_inside(inner, outer_center, outer_side, outer_angle):
    # Check if all vertices are inside
    # We can reuse _outer_side_needed logic: if required side <= given side, it's inside
    required_side = _outer_side_needed(inner, outer_center, outer_angle)
    return required_side <= outer_side + 1e-9
