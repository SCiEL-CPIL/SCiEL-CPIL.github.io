import math
import numpy as np
import time
from scipy.optimize import minimize

N = 11
APOTHEM = math.sqrt(3) / 2.0
START_TIME = time.time()
TIME_LIMIT = 150.0
SAFETY_MARGIN = 5.0
EPS = 1e-9

def _hex_vertices(cx, cy, angle_rad, scale=1.0):
    """Compute 6 vertices of a unit hexagon centered at (cx,cy) rotated by angle_rad."""
    verts = np.zeros((6, 2))
    for i in range(6):
        ang = angle_rad + i * math.pi / 3.0
        verts[i, 0] = cx + scale * math.cos(ang)
        verts[i, 1] = cy + scale * math.sin(ang)
    return verts

def _separation_distance(v1, v2, angle1, angle2):
    """
    Compute the minimum separation between two unit hexagons using Minkowski sum.
    The Minkowski sum of two unit hexagons is a regular hexagon of side 2.
    We project the center difference onto the 6 normals of the Minkowski hexagon.
    """
    dx = v2[0] - v1[0]
    dy = v2[1] - v1[1]
    
    # The Minkowski sum of two regular hexagons (one rotated by angle1, other by angle2)
    # has normals at angles that are the union of both hexagons' normals.
    # For two hexagons with relative rotation delta = angle2 - angle1,
    # the Minkowski sum's support function in direction theta is:
    #   h(theta) = APOTHEM * (|cos(theta - angle1)| + |cos(theta - angle2)|) * 2 ... 
    # Actually, for a unit hexagon, the support function is:
    #   h_hex(theta) = APOTHEM * max_k cos(theta - (angle + k*pi/3))
    # For the Minkowski sum: h = h1 + h2
    # The hexagons are separated iff for ALL directions theta:
    #   |dot(d, n(theta))| >= h1(theta) + h2(theta)
    # where d = v2 - v1
    
    # The critical directions are where the support function is piecewise linear.
    # For a unit hexagon rotated by alpha, the normals to its edges are at:
    #   alpha + pi/6 + k*pi/3 for k=0..5
    # At these directions, h = APOTHEM. Between them, h = APOTHEM * cos(delta_angle) / cos(pi/6)
    
    # For two hexagons, the Minkowski sum has 12 normals (union of both sets).
    # We check separation at all 12 critical directions.
    
    normals = []
    for k in range(6):
        normals.append(angle1 + math.pi/6 + k * math.pi/3)
    for k in range(6):
        normals.append(angle2 + math.pi/6 + k * math.pi/3)
    
    # Remove duplicates (within tolerance)
    unique_normals = []
    for ang in sorted(normals):
        if not unique_normals or abs(ang - unique_normals[-1]) > 1e-6:
            unique_normals.append(ang)
    
    min_sep = float('inf')
    for ang in unique_normals:
        nx = math.cos(ang)
        ny = math.sin(ang)
        proj = dx * nx + dy * ny
        
        # Support function of hex1 in direction ang
        h1 = APOTHEM
        for k in range(6):
            diff = ang - (angle1 + math.pi/6 + k * math.pi/3)
            diff = ((diff + math.pi) % (2*math.pi)) - math.pi
            if abs(diff) <= math.pi/6:
                h1 = APOTHEM * math.cos(diff) / math.cos(math.pi/6)
                break
        
        # Support function of hex2 in direction ang
        h2 = APOTHEM
        for k in range(6):
            diff = ang - (angle2 + math.pi/6 + k * math.pi/3)
            diff = ((diff + math.pi) % (2*math.pi)) - math.pi
            if abs(diff) <= math.pi/6:
                h2 = APOTHEM * math.cos(diff) / math.cos(math.pi/6)
                break
        
        sep = abs(proj) - (h1 + h2)
        if sep < min_sep:
            min_sep = sep
    
    return min_sep

def _separation_distance_fast(v1, v2, angle1, angle2):
    """
    Faster separation using the fact that for two regular hexagons,
    the Minkowski sum is a 12-gon (or hexagon if aligned).
    We use the key insight: if both hexagons have the same orientation,
    the Minkowski sum is a hexagon of side 2, and separation is simply
    the projection onto the 6 edge normals being >= 2*APOTHEM.
    """
    dx = v2[0] - v1[0]
    dy = v2[1] - v1[1]
    
    # Use 12 normals (6 from each hexagon)
    min_sep = float('inf')
    for k in range(6):
        ang1 = angle1 + math.pi/6 + k * math.pi/3
        ang2 = angle2 + math.pi/6 + k * math.pi/3
        for ang in [ang1, ang2]:
            nx = math.cos(ang)
            ny = math.sin(ang)
            proj = abs(dx * nx + dy * ny)
            
            # Support of hex1 in this direction
            diff1 = ang - (angle1 + math.pi/6)
            diff1 = ((diff1 + math.pi/3) % (math.pi/3)) - math.pi/6
            h1 = APOTHEM * math.cos(diff1) / math.cos(math.pi/6)
            
            # Support of hex2 in this direction
            diff2 = ang - (angle2 + math.pi/6)
            diff2 = ((diff2 + math.pi/3) % (math.pi/3)) - math.pi/6
            h2 = APOTHEM * math.cos(diff2) / math.cos(math.pi/6)
            
            sep = proj - (h1 + h2)
            if sep < min_sep:
                min_sep = sep
    
    return min_sep

def _outer_hexagon_side(pts, cx, cy, angle):
    """
    Compute the minimum side length of a regular hexagon centered at (cx,cy)
    with rotation angle that contains all points pts.
    """
    normals = []
    for k in range(6):
        ang = angle + math.pi/6 + k * math.pi/3
        normals.append((math.cos(ang), math.sin(ang)))
    
    max_proj = 0.0
    for nx, ny in normals:
        for px, py in pts:
            proj = abs((px - cx) * nx + (py - cy) * ny)
            if proj > max_proj:
                max_proj = proj
    
    return (max_proj + APOTHEM) / APOTHEM

def construct():
    if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
        # Fallback: simple lattice configuration
        pts = []
        for q in range(-1, 2):
            for r in range(-1, 2):
                if abs(q + r) <= 1:
                    x = math.sqrt(3) * (q + r / 2.0)
                    y = 1.5 * r
                    pts.append((x, y))
        inner = [[float(x), float(y), 0.0] for x, y in pts]
        return inner, [0.0, 0.0], 3.0, 0.0
    
    # Initial configuration: hexagonal lattice arrangement
    # Start with a good lattice-based initial guess
    initial_axial = [
        (0, 0),
        (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
        (2, 0), (-2, 0), (0, 2), (0, -2)
    ]
    
    def axial_to_cartesian(axial_list):
        pts = []
        for q, r in axial_list:
            x = math.sqrt(3) * (q + r / 2.0)
            y = 1.5 * r
            pts.append((x, y))
        return np.array(pts)
    
    init_pts = axial_to_cartesian(initial_axial)
    init_angles = np.zeros(N)
    
    # Parameters: [x1,y1,theta1, x2,y2,theta2, ..., x11,y11,theta11, cx, cy, outer_angle]
    # Total: 3*11 + 3 = 36 parameters
    
    def unpack(params):
        pts = np.zeros((N, 2))
        angles = np.zeros(N)
        for i in range(N):
            pts[i, 0] = params[3*i]
            pts[i, 1] = params[3*i+1]
            angles[i] = params[3*i+2]
        cx = params[33]
        cy = params[34]
        outer_angle = params[35]
        return pts, angles, cx, cy, outer_angle
    
    def objective(params):
        if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
            return 1e10
        
        pts, angles, cx, cy, outer_angle = unpack(params)
        
        # Check non-overlap with smooth penalty
        penalty = 0.0
        for i in range(N):
            for j in range(i+1, N):
                sep = _separation_distance_fast(pts[i], pts[j], angles[i], angles[j])
                if sep < 0:
                    penalty += sep * sep * 1000.0
        
        # Outer hexagon side length
        S = _outer_hexagon_side(pts, cx, cy, outer_angle)
        
        return S + penalty
    
    # Multi-start optimization
    best_S = float('inf')
    best_params = None
    
    # Generate multiple initial configurations
    def make_initial_config(offset_x=0, offset_y=0, angle_offset=0.0, jitter=0.0, seed=0):
        rng = np.random.RandomState(seed)
        pts = init_pts.copy()
        pts[:, 0] += offset_x
        pts[:, 1] += offset_y
        angles = np.full(N, angle_offset)
        
        if jitter > 0:
            pts += rng.uniform(-jitter, jitter, size=(N, 2))
            angles += rng.uniform(-jitter*0.1, jitter*0.1, size=N)
        
        cx = float(np.mean(pts[:, 0]))
        cy = float(np.mean(pts[:, 1]))
        outer_angle = 0.0
        
        params = np.zeros(36)
        for i in range(N):
            params[3*i] = pts[i, 0]
            params[3*i+1] = pts[i, 1]
            params[3*i+2] = angles[i]
        params[33] = cx
        params[34] = cy
        params[35] = outer_angle
        return params
    
    # Try multiple initial configurations
    initial_configs = []
    
    # Original lattice
    initial_configs.append(make_initial_config(seed=0))
    
    # Rotated versions
    for rot in [math.pi/6, math.pi/3, math.pi/12, 5*math.pi/12]:
        initial_configs.append(make_initial_config(angle_offset=rot, seed=100+int(rot*100)))
    
    # Small perturbations
    for seed in range(1, 6):
        initial_configs.append(make_initial_config(jitter=0.05, seed=seed))
    
    # Different lattice arrangements
    alt_axial_1 = [
        (0, 0), (1, 0), (2, 0),
        (0, 1), (1, 1), (2, 1),
        (0, 2), (1, 2), (2, 2),
        (1, 3), (2, 3)
    ]
    alt_pts_1 = axial_to_cartesian(alt_axial_1)
    params_1 = np.zeros(36)
    for i in range(N):
        params_1[3*i] = alt_pts_1[i, 0]
        params_1[3*i+1] = alt_pts_1[i, 1]
        params_1[3*i+2] = 0.0
    params_1[33] = float(np.mean(alt_pts_1[:, 0]))
    params_1[34] = float(np.mean(alt_pts_1[:, 1]))
    initial_configs.append(params_1)
    
    alt_axial_2 = [
        (0, 0), (1, 0),
        (0, 1), (1, 1), (2, 1),
        (0, 2), (1, 2), (2, 2),
        (0, 3), (1, 3), (2, 3)
    ]
    alt_pts_2 = axial_to_cartesian(alt_axial_2)
    params_2 = np.zeros(36)
    for i in range(N):
        params_2[3*i] = alt_pts_2[i, 0]
        params_2[3*i+1] = alt_pts_2[i, 1]
        params_2[3*i+2] = 0.0
    params_2[33] = float(np.mean(alt_pts_2[:, 0]))
    params_2[34] = float(np.mean(alt_pts_2[:, 1]))
    initial_configs.append(params_2)
    
    # Optimize each initial configuration
    for idx, init_params in enumerate(initial_configs):
        if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
            break
        
        try:
            # First pass: Nelder-Mead for robustness
            res = minimize(objective, init_params, method='Nelder-Mead',
                         options={'maxiter': 2000, 'xatol': 1e-8, 'fatol': 1e-8,
                                  'adaptive': True})
            
            if res.fun < best_S:
                best_S = res.fun
                best_params = res.x.copy()
            
            # Second pass: L-BFGS-B for refinement (if we have time)
            if time.time() - START_TIME < TIME_LIMIT - SAFETY_MARGIN - 10:
                # Use the result from Nelder-Mead as starting point
                res2 = minimize(objective, res.x, method='L-BFGS-B',
                              options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-8})
                
                if res2.fun < best_S:
                    best_S = res2.fun
                    best_params = res2.x.copy()
                    
        except Exception:
            pass
    
    # Final polish with very tight tolerances
    if best_params is not None and time.time() - START_TIME < TIME_LIMIT - SAFETY_MARGIN:
        try:
            res_final = minimize(objective, best_params, method='Nelder-Mead',
                               options={'maxiter': 10000, 'xatol': 1e-12, 'fatol': 1e-12,
                                        'adaptive': True})
            if res_final.fun < best_S:
                best_S = res_final.fun
                best_params = res_final.x.copy()
        except Exception:
            pass
    
    # Extract best result
    if best_params is not None:
        pts, angles, cx, cy, outer_angle = unpack(best_params)
        
        # Verify non-overlap
        valid = True
        for i in range(N):
            for j in range(i+1, N):
                sep = _separation_distance_fast(pts[i], pts[j], angles[i], angles[j])
                if sep < -1e-6:
                    valid = False
                    break
            if not valid:
                break
        
        if valid:
            inner = []
            for i in range(N):
                inner.append([float(pts[i, 0]), float(pts[i, 1]), float(math.degrees(angles[i]))])
            
            # Compute actual outer hexagon side
            S = _outer_hexagon_side(pts, cx, cy, outer_angle)
            
            # Apply tiny safety scale
            SCALE = 1.0 - 1e-10
            pts_scaled = np.zeros_like(pts)
            for i in range(N):
                pts_scaled[i, 0] = cx + (pts[i, 0] - cx) * SCALE
                pts_scaled[i, 1] = cy + (pts[i, 1] - cy) * SCALE
            
            inner = []
            for i in range(N):
                inner.append([float(pts_scaled[i, 0]), float(pts_scaled[i, 1]), float(math.degrees(angles[i]))])
            
            S_scaled = _outer_hexagon_side(pts_scaled, cx, cy, outer_angle)
            
            return inner, [float(cx), float(cy)], float(S_scaled), float(math.degrees(outer_angle))
    
    # Fallback: use the best lattice configuration
    pts = init_pts
    angles = np.zeros(N)
    cx = float(np.mean(pts[:, 0]))
    cy = float(np.mean(pts[:, 1]))
    S = _outer_hexagon_side(pts, cx, cy, 0.0)
    
    inner = []
    for i in range(N):
        inner.append([float(pts[i, 0]), float(pts[i, 1]), 0.0])
    
    return inner, [float(cx), float(cy)], float(S), 0.0
