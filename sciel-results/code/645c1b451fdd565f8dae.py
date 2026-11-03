import numpy as np
from scipy.optimize import minimize
import time

N = 12
_START_TIME = time.time()
_DEADLINE = 140.0  # safety margin before 150s kill

def _hex_vertices(cx, cy, side, angle_deg):
    """Return 6 vertices of a regular hexagon with given center, side, rotation."""
    a = np.radians(angle_deg)
    # Vertices at angles a + i*60 for i=0..5, distance = side from center
    angles = a + np.arange(6) * np.pi / 3
    return np.column_stack([cx + side * np.cos(angles), cy + side * np.sin(angles)])

def _point_in_hexagon(px, py, hex_center, hex_side, hex_angle_deg):
    """Check if point (px,py) is inside regular hexagon (center, side, angle)."""
    # Transform point to hexagon's local frame
    a = np.radians(hex_angle_deg)
    dx = px - hex_center[0]
    dy = py - hex_center[1]
    # Rotate by -a to align hexagon with vertex at angle 0
    cx = dx * np.cos(-a) - dy * np.sin(-a)
    sy = dx * np.sin(-a) + dy * np.cos(-a)
    # In local frame, hexagon has vertices at (S,0), (S/2, S*sqrt(3)/2), etc.
    # The hexagon is defined by |x| <= S, |x/2 + y*sqrt(3)/2| <= S, |-x/2 + y*sqrt(3)/2| <= S
    # Actually, let's use the support function approach:
    # For a regular hexagon with circumradius S, the 6 edge normals are at angles 30,90,150,210,270,330
    # The constraint is: for each normal direction, dot(point, normal) <= S*sin(60) = S*sqrt(3)/2
    # Wait, apothem = S * cos(30) = S * sqrt(3)/2
    # Edge normals at angles: 30, 90, 150, 210, 270, 330 degrees
    normal_angles = np.radians([30, 90, 150, 210, 270, 330])
    for na in normal_angles:
        val = cx * np.cos(na) + sy * np.sin(na)
        if val > hex_side * np.sqrt(3) / 2 + 1e-9:
            return False
    return True

def _all_inside(inner, outer_center, outer_side, outer_angle):
    """Check all inner hexagon vertices are inside outer hexagon."""
    for (cx, cy, ang) in inner:
        verts = _hex_vertices(cx, cy, 1.0, ang)
        for (vx, vy) in verts:
            if not _point_in_hexagon(vx, vy, outer_center, outer_side, outer_angle):
                return False
    return True

def _check_disjoint(inner):
    """Check no two inner hexagons overlap (minimum distance between centers >= 2 for same orientation)."""
    # For two unit hexagons, they're disjoint if no vertex of one is inside the other and vice versa
    # Simple check: distance between centers must be >= some threshold, but orientation matters
    # Use: for each pair, check no vertex of hex i is strictly inside hex j
    n = len(inner)
    for i in range(n):
        vi = _hex_vertices(inner[i][0], inner[i][1], 1.0, inner[i][2])
        for j in range(i + 1, n):
            vj = _hex_vertices(inner[j][0], inner[j][1], 1.0, inner[j][2])
            # Check if any vertex of i is inside j
            for (vx, vy) in vi:
                if _point_in_hexagon(vx, vy, (inner[j][0], inner[j][1]), 1.0, inner[j][2]):
                    return False
            # Check if any vertex of j is inside i
            for (vx, vy) in vj:
                if _point_in_hexagon(vx, vy, (inner[i][0], inner[i][1]), 1.0, inner[i][2]):
                    return False
    return True

def _outer_side_needed(inner, outer_center, outer_angle):
    """Given inner hexagons and outer center/angle, find minimum outer side that contains all."""
    # For each inner vertex, compute the required outer side
    # In the outer hexagon's local frame, the constraint is:
    # For each of 6 edge normals, dot(local_point, normal) <= S * sqrt(3)/2
    # So S >= dot(local_point, normal) / (sqrt(3)/2) for each normal
    # S >= max over all vertices and all normals of dot / (sqrt(3)/2)
    a = np.radians(outer_angle)
    ca, sa = np.cos(-a), np.sin(-a)
    normal_angles = np.radians([30, 90, 150, 210, 270, 330])
    max_val = 0.0
    for (cx, cy, ang) in inner:
        verts = _hex_vertices(cx, cy, 1.0, ang)
        for (vx, vy) in verts:
            dx = vx - outer_center[0]
            dy = vy - outer_center[1]
            lx = dx * ca - dy * sa
            ly = dx * sa + dy * ca
            for na in normal_angles:
                val = lx * np.cos(na) + ly * np.sin(na)
                if val > max_val:
                    max_val = val
    return max_val * 2.0 / np.sqrt(3)

def construct():
    global _START_TIME
    t0 = time.time()
    
    # Initial configuration: honeycomb-like arrangement
    # 3 rows of 4, or 4 rows of 3
    # Let's try a compact arrangement
    
    # Honeycomb: rows offset
    # Row spacing for hexagons: sqrt(3) * side = sqrt(3)
    # Column spacing: 1.5 * side = 1.5 for same row, but adjacent rows offset by 0.75
    
    # Let's place them in a grid pattern
    # 4 columns x 3 rows
    # For flat-top hexagons (vertex up), width = sqrt(3), height = 2
    # For pointy-top (flat top), width = 2, height = sqrt(3)
    
    # Let's use angle 0 (vertex at angle 0, so flat top/bottom)
    # Width (horizontal) = 2, Height (vertical) = sqrt(3)
    # Horizontal spacing: 2 (touching)
    # Vertical spacing: sqrt(3) (touching)
    # But we can offset alternate rows by 1 for honeycomb
    
    # Start with a simple grid
    inner = []
    for row in range(3):
        for col in range(4):
            x = col * 2.0
            y = row * np.sqrt(3)
            if row % 2 == 1:
                x += 1.0
            inner.append([x, y, 0.0])
    
    # Center the configuration
    xs = [p[0] for p in inner]
    ys = [p[1] for p in inner]
    cx = (min(xs) + max(xs)) / 2
    cy = (min(ys) + max(ys)) / 2
    for i in range(len(inner)):
        inner[i][0] -= cx
        inner[i][1] -= cy
    
    # Compute initial outer side
    outer_center = (0.0, 0.0)
    outer_angle = 0.0
    outer_side = _outer_side_needed(inner, outer_center, outer_angle)
    
    best_side = outer_side
    best_inner = [list(x) for x in inner]
    best_center = outer_center
    best_angle = outer_angle
    
    # Optimization: minimize outer_side by adjusting inner positions and angles
    # Variables: for each inner hex, (x, y, angle), plus outer_center_x, outer_center_y, outer_angle
    # That's 12*3 + 3 = 39 variables
    
    def objective(params):
        if time.time() - t0 > _DEADLINE - 5:
            return best_side
        inner_x = params[0:12]
        inner_y = params[12:24]
        inner_ang = params[24:36]
        ocx = params[36]
        ocy = params[37]
        oang = params[38]
        
        # Build inner list
        inner_list = [[inner_x[i], inner_y[i], inner_ang[i]] for i in range(12)]
        
        # Check disjointness (penalty if overlapping)
        penalty = 0.0
        for i in range(12):
            vi = _hex_vertices(inner_x[i], inner_y[i], 1.0, inner_ang[i])
            for j in range(i+1, 12):
                vj = _hex_vertices(inner_x[j], inner_y[j], 1.0, inner_ang[j])
                # Check overlap: any vertex of i inside j or vice versa
                for (vx, vy) in vi:
                    if _point_in_hexagon(vx, vy, (inner_x[j], inner_y[j]), 1.0, inner_ang[j]):
                        penalty += 100.0
                for (vx, vy) in vj:
                    if _point_in_hexagon(vx, vy, (inner_x[i], inner_y[i]), 1.0, inner_ang[i]):
                        penalty += 100.0
                # Also check center-to-center distance as a soft constraint
                d = np.sqrt((inner_x[i]-inner_x[j])**2 + (inner_y[i]-inner_y[j])**2)
                if d < 1.8:  # minimum distance for unit hexagons
                    penalty += (1.8 - d) * 50.0
        
        outer_side = _outer_side_needed(inner_list, (ocx, ocy), oang)
        return outer_side + penalty
    
    # Initial params
    init_params = np.zeros(39)
    for i in range(12):
        init_params[i] = inner[i][0]
        init_params[12+i] = inner[i][1]
        init_params[24+i] = inner[i][2]
    init_params[36] = 0.0
    init_params[37] = 0.0
    init_params[38] = 0.0
    
    # Try optimization with time limit
    try:
        result = minimize(objective, init_params, method='Nelder-Mead',
                         options={'maxiter': 2000, 'xatol': 1e-6, 'fatol': 1e-6})
        if time.time() - t0 < _DEADLINE - 10:
            # Extract result
            res_inner = [[result.x[i], result.x[12+i], result.x[24+i]] for i in range(12)]
            res_center = (result.x[36], result.x[37])
            res_angle = result.x[38]
            
            # Verify disjointness
            if _check_disjoint(res_inner):
                res_side = _outer_side_needed(res_inner, res_center, res_angle)
                if res_side < best_side:
                    best_side = res_side
                    best_inner = res_inner
                    best_center = res_center
                    best_angle = res_angle
    except Exception:
        pass
    
    # Try a few more random restarts if time permits
    np.random.seed(42)
    for restart in range(5):
        if time.time() - t0 > _DEADLINE - 10:
            break
        
        # Random perturbation of initial config
        inner_r = []
        for i in range(12):
            x = inner[i][0] + np.random.randn() * 0.5
            y = inner[i][1] + np.random.randn() * 0.5
            ang = np.random.choice([0, 30, 60, 90, 120, 150])
            inner_r.append([x, y, ang])
        
        # Re-center
        xs = [p[0] for p in inner_r]
        ys = [p[1] for p in inner_r]
        cx = (min(xs) + max(xs)) / 2
        cy = (min(ys) + max(ys)) / 2
        for i in range(len(inner_r)):
            inner_r[i][0] -= cx
            inner_r[i][1] -= cy
        
        # Quick local optimization
        init_params_r = np.zeros(39)
        for i in range(12):
            init_params_r[i] = inner_r[i][0]
            init_params_r[12+i] = inner_r[i][1]
            init_params_r[24+i] = inner_r[i][2]
        init_params_r[36] = 0.0
        init_params_r[37] = 0.0
        init_params_r[38] = 0.0
        
        try:
            result_r = minimize(objective, init_params_r, method='Nelder-Mead',
                               options={'maxiter': 500, 'xatol': 1e-5, 'fatol': 1e-5})
            if time.time() - t0 > _DEADLINE - 5:
                break
            res_inner_r = [[result_r.x[i], result_r.x[12+i], result_r.x[24+i]] for i in range(12)]
            res_center_r = (result_r.x[36], result_r.x[37])
            res_angle_r = result_r.x[38]
            
            if _check_disjoint(res_inner_r):
                res_side_r = _outer_side_needed(res_inner_r, res_center_r, res_angle_r)
                if res_side_r < best_side:
                    best_side = res_side_r
                    best_inner = res_inner_r
                    best_center = res_center_r
                    best_angle = res_angle_r
        except Exception:
            pass
    
    # Final check: make sure the best solution is valid
    if not _check_disjoint(best_inner):
        # Fallback to initial
        best_inner = [list(x) for x in inner]
        best_center = (0.0, 0.0)
        best_angle = 0.0
        best_side = _outer_side_needed(best_inner, best_center, best_angle)
    
    # Ensure all inner hexagons are inside outer
    if not _all_inside(best_inner, best_center, best_side, best_angle):
        # Increase side slightly
        best_side *= 1.01
    
    return best_inner, best_center, best_side, best_angle
