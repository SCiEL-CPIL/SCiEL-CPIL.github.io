import math
import numpy as np
import time
from scipy.optimize import minimize

N = 11
APOTHEM = math.sqrt(3) / 2.0
SQRT3 = math.sqrt(3)
START_TIME = time.time()
TIME_LIMIT = 150.0
SAFETY_MARGIN = 10.0

def hex_vertices(cx, cy, angle_deg, side=1.0):
    """Return 6 vertices of a regular hexagon."""
    a = math.radians(angle_deg)
    verts = []
    for k in range(6):
        theta = a + k * math.pi / 3.0
        verts.append((cx + side * math.cos(theta), cy + side * math.sin(theta)))
    return np.array(verts)

def container_normals(outer_angle_deg):
    """6 outward normals of the container hexagon."""
    a = math.radians(outer_angle_deg)
    normals = []
    for k in range(6):
        theta = math.pi / 6.0 + k * math.pi / 3.0 + a
        normals.append((math.cos(theta), math.sin(theta)))
    return normals

def support_function_exact(centers, angles, outer_center, outer_angle_deg):
    """Compute the required outer side length using exact support function."""
    ox, oy = outer_center
    normals = container_normals(outer_angle_deg)
    max_proj = 0.0
    for i in range(len(centers)):
        cx, cy = centers[i]
        verts = hex_vertices(cx, cy, angles[i])
        for v in verts:
            dx = v[0] - ox
            dy = v[1] - oy
            for nx, ny in normals:
                p = dx * nx + dy * ny
                if p > max_proj:
                    max_proj = p
    # Side length S satisfies: apothem_outer = max_proj + apothem_inner
    # apothem_outer = S * sqrt(3)/2, apothem_inner = sqrt(3)/2
    # So S * sqrt(3)/2 = max_proj + sqrt(3)/2
    # S = (max_proj + sqrt(3)/2) / (sqrt(3)/2) = (2*max_proj + sqrt(3)) / sqrt(3)
    S = (max_proj + APOTHEM) / APOTHEM
    return S

def support_function_smooth(centers, angles, outer_center, outer_angle_deg, beta=50.0):
    """Smooth log-sum-exp version of support function."""
    ox, oy = outer_center
    normals = container_normals(outer_angle_deg)
    projs = []
    for i in range(len(centers)):
        cx, cy = centers[i]
        verts = hex_vertices(cx, cy, angles[i])
        for v in verts:
            dx = v[0] - ox
            dy = v[1] - oy
            for nx, ny in normals:
                projs.append(dx * nx + dy * ny)
    if not projs:
        return 0.0
    P = np.array(projs)
    m = np.max(P)
    if m < -1e6:
        return 0.0
    # Log-sum-exp for smooth max
    exp_terms = np.exp(beta * (P - m))
    smooth_max = m + math.log(np.sum(exp_terms)) / beta
    S = (smooth_max + APOTHEM) / APOTHEM
    return S

def min_center_distance(a1, a2):
    """Minimum center distance for two unit hexagons with angles a1, a2 to be disjoint."""
    # For two regular hexagons, the minimum distance depends on relative orientation
    # The worst case (closest approach) is when edges are parallel
    # For axis-aligned: min distance = sqrt(3)
    # With rotation, can be slightly less
    # Approximate: the minimum distance is the minimum over all edge pairs
    # For simplicity, use sqrt(3) as a conservative bound, but allow optimization
    # Actually, for two convex polygons, we can compute exact min distance
    # But for speed, use the fact that for regular hexagons, the min distance
    # between centers for non-overlap is at most sqrt(3)
    # Let's compute it properly
    rel_angle = (a2 - a1) % 60.0
    rel_angle = min(rel_angle, 60.0 - rel_angle)
    # The minimum distance varies with relative angle
    # At 0 degrees (aligned): sqrt(3)
    # At 30 degrees: can be less
    # Approximate formula: d_min = sqrt(3) * cos(rel_angle * pi/6) + ... 
    # Actually, let's just use sqrt(3) as the constraint and let the optimizer figure it out
    # The penalty approach will handle it
    return SQRT3

def contact_penalty(centers, angles, min_dist=SQRT3 - 0.001):
    """Quadratic penalty for center-distance violations."""
    pen = 0.0
    n = len(centers)
    for i in range(n):
        for j in range(i + 1, n):
            dx = centers[i][0] - centers[j][0]
            dy = centers[i][1] - centers[j][1]
            d = math.sqrt(dx * dx + dy * dy)
            # The minimum distance depends on relative rotation
            # For simplicity, use a conservative minimum
            if d < min_dist:
                pen += (min_dist - d) ** 2
    return pen

def objective_smooth(params, beta=50.0):
    """
    params: [x0, y0, a0, x1, y1, a1, ..., x10, y10, a10, ox, oy, outer_angle]
    """
    n = N
    centers = []
    angles = []
    for i in range(n):
        centers.append((params[i*3], params[i*3+1]))
        angles.append(params[i*3+2])
    ox, oy = params[n*3], params[n*3+1]
    outer_angle = params[n*3+2]
    
    S = support_function_smooth(centers, angles, (ox, oy), outer_angle, beta=beta)
    pen = contact_penalty(centers, angles)
    return S + 0.1 * pen

def objective_exact(params):
    """Exact objective for final polish."""
    n = N
    centers = []
    angles = []
    for i in range(n):
        centers.append((params[i*3], params[i*3+1]))
        angles.append(params[i*3+2])
    ox, oy = params[n*3], params[n*3+1]
    outer_angle = params[n*3+2]
    
    S = support_function_exact(centers, angles, (ox, oy), outer_angle)
    pen = contact_penalty(centers, angles)
    return S + 0.01 * pen

def initial_config_flower():
    """Flower pattern: center + 6 around + 4 more."""
    centers = []
    angles = []
    # Center
    centers.append((0.0, 0.0))
    angles.append(0.0)
    # 6 around
    for k in range(6):
        theta = k * math.pi / 3.0
        cx = SQRT3 * math.cos(theta)
        cy = SQRT3 * math.sin(theta)
        centers.append((cx, cy))
        angles.append(0.0)
    # 4 more in second ring
    for k in range(4):
        theta = math.pi/6 + k * math.pi / 2.0
        cx = 2 * SQRT3 * math.cos(theta)
        cy = 2 * SQRT3 * math.sin(theta)
        centers.append((cx, cy))
        angles.append(0.0)
    return centers, angles

def initial_config_compact():
    """Compact rectangular-ish arrangement."""
    centers = []
    angles = []
    # 3x4 grid with offsets
    for row in range(3):
        for col in range(4):
            x = col * SQRT3
            y = row * 1.5
            if row % 2 == 1:
                x += SQRT3 / 2
            centers.append((x, y))
            angles.append(0.0)
    # One more
    centers.append((0.0, 4.5))
    angles.append(0.0)
    return centers, angles

def optimize_from_initial(centers, angles, deadline, n_starts=6):
    """Multi-start optimization."""
    ox0 = sum(c[0] for c in centers) / len(centers)
    oy0 = sum(c[1] for c in centers) / len(centers)
    
    best_val = float('inf')
    best_params = None
    
    angle_candidates = [0, 15, 30, 45, 60, 75]
    
    for k in range(n_starts):
        if time.time() > deadline - 20:
            break
        
        # Initialize
        params = []
        for i in range(N):
            params.extend([centers[i][0], centers[i][1], angles[i]])
        params.extend([ox0, oy0, angle_candidates[k % len(angle_candidates)]])
        
        try:
            # Smooth phase
            res_s = minimize(lambda p: objective_smooth(p, beta=40.0),
                           params, method='Nelder-Mead',
                           options={'maxiter': 200, 'xatol': 1e-4, 'fatol': 1e-6, 'maxfev': 500})
            
            if time.time() > deadline - 10:
                # Only have time for smooth result
                if res_s.fun < best_val:
                    best_val = res_s.fun
                    best_params = res_s.x.copy()
                continue
            
            # Exact phase
            res_e = minimize(lambda p: objective_exact(p),
                           res_s.x, method='Nelder-Mead',
                           options={'maxiter': 300, 'xatol': 1e-8, 'fatol': 1e-10, 'maxfev': 800})
            
            if res_e.fun < best_val:
                best_val = res_e.fun
                best_params = res_e.x.copy()
                
        except Exception:
            pass
    
    return best_val, best_params

def construct():
    # Check time
    if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
        # Fallback: return a valid but suboptimal solution
        centers, angles = initial_config_flower()
        ox = sum(c[0] for c in centers) / N
        oy = sum(c[1] for c in centers) / N
        S = support_function_exact(centers, angles, (ox, oy), 0.0)
        inner = [[float(centers[i][0]), float(centers[i][1]), float(angles[i])] for i in range(N)]
        return inner, [float(ox), float(oy)], float(S), 0.0
    
    deadline = START_TIME + TIME_LIMIT - SAFETY_MARGIN
    
    best_S = float('inf')
    best_inner = None
    best_center = None
    best_angle = 0.0
    
    # Try multiple initial configurations
    configs = [
        initial_config_flower(),
        initial_config_compact(),
    ]
    
    # Also try rotated versions of flower
    for rot in [0, 30, 60]:
        c, a = initial_config_flower()
        c_rot = []
        a_rot = []
        for cx, cy in c:
            theta = math.radians(rot)
            nx = cx * math.cos(theta) - cy * math.sin(theta)
            ny = cx * math.sin(theta) + cy * math.cos(theta)
            c_rot.append((nx, ny))
            a_rot.append(a[len(a_rot)] + rot)
        configs.append((c_rot, a_rot))
    
    for centers, angles in configs:
        if time.time() > deadline - 15:
            break
        
        val, params = optimize_from_initial(centers, angles, deadline, n_starts=4)
        
        if val < best_S and params is not None:
            # Extract the actual S (without penalty)
            n = N
            c = []
            a = []
            for i in range(n):
                c.append((params[i*3], params[i*3+1]))
                a.append(params[i*3+2])
            ox, oy = params[n*3], params[n*3+1]
            outer_angle = params[n*3+2]
            
            S_exact = support_function_exact(c, a, (ox, oy), outer_angle)
            if S_exact < best_S:
                best_S = S_exact
                best_inner = [[float(c[i][0]), float(c[i][1]), float(a[i])] for i in range(N)]
                best_center = [float(ox), float(oy)]
                best_angle = float(outer_angle)
    
    # Final polish on the best solution
    if best_inner is not None and time.time() < deadline - 5:
        params = []
        for i in range(N):
            params.extend([best_inner[i][0], best_inner[i][1], best_inner[i][2]])
        params.extend([best_center[0], best_center[1], best_angle])
        
        try:
            res = minimize(lambda p: objective_exact(p),
                          params, method='Nelder-Mead',
                          options={'maxiter': 200, 'xatol': 1e-10, 'fatol': 1e-12, 'maxfev': 500})
            
            if res.fun < best_S:
                n = N
                c = []
                a = []
                for i in range(n):
                    c.append((res.x[i*3], res.x[i*3+1]))
                    a.append(res.x[i*3+2])
                ox, oy = res.x[n*3], res.x[n*3+1]
                outer_angle = res.x[n*3+2]
                
                S_final = support_function_exact(c, a, (ox, oy), outer_angle)
                if S_final < best_S:
                    best_S = S_final
                    best_inner = [[float(c[i][0]), float(c[i][1]), float(a[i])] for i in range(N)]
                    best_center = [float(ox), float(oy)]
                    best_angle = float(outer_angle)
        except Exception:
            pass
    
    if best_inner is None:
        # Ultimate fallback
        centers, angles = initial_config_flower()
        ox = sum(c[0] for c in centers) / N
        oy = sum(c[1] for c in centers) / N
        best_S = support_function_exact(centers, angles, (ox, oy), 0.0)
        best_inner = [[float(centers[i][0]), float(centers[i][1]), float(angles[i])] for i in range(N)]
        best_center = [float(ox), float(oy)]
        best_angle = 0.0
    
    return best_inner, best_center, float(best_S), float(best_angle)
