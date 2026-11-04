import numpy as np
from scipy.optimize import minimize
import time

N = 12
DEADLINE = 145.0

def get_hex_vertices(cx, cy, side=1.0, rot_deg=0.0):
    angles = np.radians(np.array([0, 60, 120, 180, 240, 300])) + np.radians(rot_deg)
    x = cx + side * np.cos(angles)
    y = cy + side * np.sin(angles)
    return np.column_stack((x, y))

def sat_overlap(v1, v2):
    axes = []
    for v in (v1, v2):
        for i in range(len(v)):
            p1 = v[i]
            p2 = v[(i+1) % len(v)]
            e = p2 - p1
            n = np.array([-e[1], e[0]])
            norm = np.linalg.norm(n)
            if norm > 1e-12:
                n = n / norm
                axes.append(n)
    
    min_overlap = float('inf')
    for n in axes:
        proj1 = v1 @ n
        min1, max1 = np.min(proj1), np.max(proj1)
        proj2 = v2 @ n
        min2, max2 = np.min(proj2), np.max(proj2)
        
        if max1 < min2 or max2 < min1:
            gap = min2 - max1 if min2 > max1 else min1 - max2
            return gap
        
        overlap = min(max1, max2) - max(min1, min2)
        min_overlap = min(min_overlap, overlap)
    
    return -min_overlap

def outer_hex_side(points, outer_rot_deg):
    # Normals to the sides of the outer hexagon
    # Sides are at angles 30, 90, 150, 210, 270, 330 + rot
    # Distance from center to side is R * cos(30) = R * sqrt(3)/2
    # So R = d / (sqrt(3)/2) = 2d / sqrt(3)
    # Here R is the circumradius, which equals the side length L.
    angles = np.radians(np.array([90, 150, 210, 270, 330, 30]) + outer_rot_deg)
    normals = np.column_stack((np.cos(angles), np.sin(angles)))
    proj = points @ normals.T
    max_proj = np.max(np.abs(proj))
    return (2.0 / np.sqrt(3.0)) * max_proj

def construct():
    start_time = time.time()
    
    def time_left():
        return DEADLINE - (time.time() - start_time)

    # Generate initial configuration: 1-3-4-3-1 grid
    s = np.sqrt(3)
    h = 1.5
    pts = []
    # Row 0 (y=2h): 1 hex
    pts.append([0, 2*h, 0.0])
    # Row 1 (y=h): 3 hexes
    for dx in [-s, 0, s]:
        pts.append([dx, h, 0.0])
    # Row 2 (y=0): 4 hexes
    for dx in [-1.5*s, -0.5*s, 0.5*s, 1.5*s]:
        pts.append([dx, 0, 0.0])
    # Row 3 (y=-h): 3 hexes
    for dx in [-s, 0, s]:
        pts.append([dx, -h, 0.0])
    # Row 4 (y=-2h): 1 hex
    pts.append([0, -2*h, 0.0])
    
    pts = np.array(pts)
    # Center the configuration
    pts[:, :2] -= np.mean(pts[:, :2], axis=0)
    
    init_params = np.concatenate((pts.flatten(), [0.0])) # Last param is outer rotation
    
    def objective(params):
        if time.time() - start_time > DEADLINE - 5:
            return 1e6
        
        xys_rots = params[:3*N].reshape(N, 3)
        outer_rot = params[3*N]
        
        # Calculate vertices
        vertices_list = [get_hex_vertices(x, y, 1.0, r) for x, y, r in xys_rots]
        
        # Calculate overlap penalty
        penalty = 0.0
        for i in range(N):
            for j in range(i+1, N):
                d = sat_overlap(vertices_list[i], vertices_list[j])
                if d < 0:
                    penalty += d**2
        
        # Calculate outer side length
        all_points = np.vstack(vertices_list)
        L = outer_hex_side(all_points, outer_rot)
        
        return L + 100.0 * penalty

    best_L = float('inf')
    best_params = None
    
    # Try a few initial outer rotations
    for orot in [0.0, 30.0, 15.0]:
        init_params[3*N] = orot
        
        # Use Nelder-Mead for robustness
        try:
            res = minimize(
                objective,
                init_params,
                method='Nelder-Mead',
                options={
                    'maxiter': 500,
                    'xatol': 1e-6,
                    'fatol': 1e-6,
                    'adaptive': True
                }
            )
            
            # Check validity
            xys_rots = res.x[:3*N].reshape(N, 3)
            vertices_list = [get_hex_vertices(x, y, 1.0, r) for x, y, r in xys_rots]
            valid = True
            for i in range(N):
                for j in range(i+1, N):
                    d = sat_overlap(vertices_list[i], vertices_list[j])
                    if d < -1e-6:
                        valid = False
                        break
                if not valid: break
            
            if valid:
                L_val = outer_hex_side(np.vstack(vertices_list), res.x[3*N])
                if L_val < best_L:
                    best_L = L_val
                    best_params = res.x.copy()
                    
        except Exception:
            pass
            
        if time_left() < 10:
            break

    if best_params is None:
        # Fallback
        inner = [[float(x), float(y), float(r)] for x, y, r in pts]
        L_val = outer_hex_side(np.vstack([get_hex_vertices(x, y, 1.0, r) for x, y, r in pts]), 0.0)
        return inner, (0.0, 0.0), L_val, 0.0

    xys_rots = best_params[:3*N].reshape(N, 3)
    outer_rot = best_params[3*N]
    
    inner = [[float(x), float(y), float(r)] for x, y, r in xys_rots]
    center = [0.0, 0.0]
    side = float(best_L)
    angle = float(outer_rot)
    
    return inner, center, side, angle
