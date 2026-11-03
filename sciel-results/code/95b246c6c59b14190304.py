import math
import time
import numpy as np
from scipy.optimize import linprog, minimize

N = 11
_START_TIME = time.time()
_DEADLINE = _START_TIME + 115.0  # safety margin within 150s limit

def _get_hex_vertices(cx, cy, angle_deg=0.0):
    """Get vertices of a unit regular hexagon centered at (cx, cy) with rotation."""
    angle_rad = math.radians(angle_deg)
    verts = []
    for k in range(6):
        a = angle_rad + math.pi * k / 3
        verts.append((cx + math.cos(a), cy + math.sin(a)))
    return verts

def _min_enclosing_hex_side_lp(vertices, angle_deg=0.0):
    """
    Find minimum side length of a regular hexagon of given angle
    that contains all given vertices.
    Returns (cx, cy, side, angle_deg).
    """
    if not vertices:
        return 0.0, 0.0, 0.0, angle_deg
    
    n_pts = len(vertices)
    
    # Normals of the outer hexagon
    a_rad = math.radians(angle_deg)
    normals = []
    for k in range(6):
        a = a_rad + math.pi * k / 3
        normals.append((math.cos(a), math.sin(a)))
    
    # LP Variables: cx, cy, R (inradius)
    # Minimize R
    c = np.array([0.0, 0.0, 1.0])
    
    A_ub = []
    b_ub = []
    
    # Constraints: n_k . (p - c) <= R  =>  -n_k . c + R >= -n_k . p  =>  n_k . c - R <= n_k . p
    # Wait, standard form is A_ub x <= b_ub.
    # n_kx * cx + n_ky * cy - 1*R <= n_kx * vx + n_ky * vy
    
    for nx, ny in normals:
        for vx, vy in vertices:
            A_ub.append([nx, ny, -1.0])
            b_ub.append(nx * vx + ny * vy)
    
    # R >= 0  =>  -R <= 0
    A_ub.append([0.0, 0.0, -1.0])
    b_ub.append(0.0)
    
    A_ub = np.array(A_ub)
    b_ub = np.array(b_ub)
    
    try:
        result = linprog(c, A_ub=A_ub, b_ub=b_ub, method='highs')
        if result.success:
            cx_opt = result.x[0]
            cy_opt = result.x[1]
            r_opt = result.x[2]
        else:
            # Fallback
            cx_mean = sum(v[0] for v in vertices) / n_pts
            cy_mean = sum(v[1] for v in vertices) / n_pts
            r_opt = 0.0
            for nx, ny in normals:
                for vx, vy in vertices:
                    proj = (vx - cx_mean) * nx + (vy - cy_mean) * ny
                    r_opt = max(r_opt, proj)
            cx_opt = cx_mean
            cy_opt = cy_mean
    except Exception:
        cx_opt = 0.0
        cy_opt = 0.0
        r_opt = 10.0
        
    side = r_opt / (math.sqrt(3) / 2)
    return cx_opt, cy_opt, side, angle_deg

def _check_overlap_penalty(params, N):
    """
    Calculate overlap penalty for a given set of hexagon parameters.
    params: [x1, y1, a1, x2, y2, a2, ...]
    Returns sum of squared penetration depths.
    """
    penalty = 0.0
    hexes = []
    for i in range(N):
        x = params[3*i]
        y = params[3*i+1]
        a = params[3*i+2]
        hexes.append((x, y, a))
        
    for i in range(N):
        for j in range(i+1, N):
            x1, y1, a1 = hexes[i]
            x2, y2, a2 = hexes[j]
            
            # Quick distance check
            dx = x2 - x1
            dy = y2 - y1
            dist_sq = dx*dx + dy*dy
            if dist_sq > 4.0: # Max distance for overlap is 2*1 = 2, so dist > 2 means no overlap
                continue
                
            # SAT check
            a1_rad = math.radians(a1)
            a2_rad = math.radians(a2)
            
            min_margin = float('inf')
            
            # Check 6 normals of hex1
            for k in range(6):
                ang = a1_rad + math.pi/6 + k * math.pi/3
                nx = math.cos(ang)
                ny = math.sin(ang)
                
                # Support of hex1 in direction (nx, ny)
                s1 = max(math.cos(math.atan2(ny, nx) - (a1_rad + j * math.pi/3)) for j in range(6))
                # Support of hex2 in direction (-nx, -ny)
                s2 = max(math.cos(math.atan2(-ny, -nx) - (a2_rad + j * math.pi/3)) for j in range(6))
                
                proj = dx * nx + dy * ny
                margin = s1 + s2 - proj
                min_margin = min(min_margin, margin)
                
            # Check 6 normals of hex2
            for k in range(6):
                ang = a2_rad + math.pi/6 + k * math.pi/3
                nx = math.cos(ang)
                ny = math.sin(ang)
                
                s1 = max(math.cos(math.atan2(ny, nx) - (a1_rad + j * math.pi/3)) for j in range(6))
                s2 = max(math.cos(math.atan2(-ny, -nx) - (a2_rad + j * math.pi/3)) for j in range(6))
                
                proj = dx * nx + dy * ny
                margin = s1 + s2 - proj
                min_margin = min(min_margin, margin)
                
            if min_margin < 0:
                penalty += min_margin * min_margin
                
    return penalty

def _objective(params):
    if time.time() > _DEADLINE - 5.0:
        return 1e10
        
    N_local = len(params) // 3
    hexes = []
    all_vertices = []
    
    for i in range(N_local):
        x = params[3*i]
        y = params[3*i+1]
        a = params[3*i+2]
        hexes.append((x, y, a))
        verts = _get_hex_vertices(x, y, a)
        all_vertices.extend(verts)
        
    # Check overlaps
    overlap_pen = _check_overlap_penalty(params, N_local)
    
    if overlap_pen > 1e-6:
        return 100.0 + overlap_pen
        
    # Find min enclosing hexagon side
    # We can try a few outer angles to be safe, or just 0.
    # Given symmetry, 0 is likely good, but let's try 0, 15, 30, 45.
    best_side = float('inf')
    best_cx = 0.0
    best_cy = 0.0
    
    for ang in [0.0, 15.0, 30.0, 45.0]:
        cx, cy, side, _ = _min_enclosing_hex_side_lp(all_vertices, ang)
        if side < best_side:
            best_side = side
            best_cx = cx
            best_cy = cy
            
    return best_side

def _lattice_init():
    """Generate initial positions on a triangular lattice."""
    sqrt3 = math.sqrt(3)
    u = (sqrt3, 0.0)
    v = (sqrt3/2, 1.5)
    
    points = []
    for i in range(-4, 5):
        for j in range(-4, 5):
            x = i * u[0] + j * v[0]
            y = i * u[1] + j * v[1]
            points.append((x, y))
            
    points.sort(key=lambda p: p[0]*p[0] + p[1]*p[1])
    
    params = []
    for i in range(N):
        x, y = points[i]
        params.extend([x, y, 0.0])
        
    return params

def construct():
    global _START_TIME, _DEADLINE
    
    if time.time() > _DEADLINE - 10.0:
        # Emergency fallback
        p = _lattice_init()
        hexes = []
        all_vertices = []
        for i in range(N):
            x, y, a = p[3*i], p[3*i+1], p[3*i+2]
            hexes.append((x, y, a))
            all_vertices.extend(_get_hex_vertices(x, y, a))
        cx, cy, side, ang = _min_enclosing_hex_side_lp(all_vertices, 0.0)
        return [[p[3*i], p[3*i+1], p[3*i+2]] for i in range(N)], (cx, cy), side, ang

    best_params = None
    best_score = float('inf')
    
    # Try a few initializations
    inits = [
        _lattice_init(),
        _lattice_init(), # Add noise to second one
    ]
    
    # Add noise to the second init
    for i in range(len(inits[1])):
        if i % 3 != 2: # Don't perturb angles too much initially
            inits[1][i] += np.random.uniform(-0.2, 0.2)
        else:
            inits[1][i] += np.random.uniform(-10, 10)
            
    # Add a third init: slightly different lattice offset
    sqrt3 = math.sqrt(3)
    u = (sqrt3, 0.0)
    v = (sqrt3/2, 1.5)
    points = []
    for i in range(-4, 5):
        for j in range(-4, 5):
            x = i * u[0] + j * v[0] + 0.1
            y = i * u[1] + j * v[1] + 0.1
            points.append((x, y))
    points.sort(key=lambda p: p[0]*p[0] + p[1]*p[1])
    params3 = []
    for i in range(N):
        x, y = points[i]
        params3.extend([x, y, 0.0])
    inits.append(params3)
    
    for init in inits:
        if time.time() > _DEADLINE - 20.0:
            break
            
        try:
            res = minimize(_objective, init, method='Nelder-Mead', 
                           options={'xatol': 1e-4, 'fatol': 1e-4, 'maxiter': 2000})
            
            if res.fun < best_score:
                best_score = res.fun
                best_params = res.x
        except Exception:
            pass
            
    if best_params is None:
        best_params = _lattice_init()
        
    # Final polish: Extract results
    hexes = []
    all_vertices = []
    for i in range(N):
        x, y, a = best_params[3*i], best_params[3*i+1], best_params[3*i+2]
        hexes.append((x, y, a))
        all_vertices.extend(_get_hex_vertices(x, y, a))
        
    # Find best outer hexagon
    best_side = float('inf')
    best_cx = 0.0
    best_cy = 0.0
    best_ang = 0.0
    
    # Sweep outer angle more finely for final result
    for ang in np.arange(0, 60, 1.0):
        cx, cy, side, _ = _min_enclosing_hex_side_lp(all_vertices, ang)
        if side < best_side:
            best_side = side
            best_cx = cx
            best_cy = cy
            best_ang = ang
            
    return [[hexes[i][0], hexes[i][1], hexes[i][2]] for i in range(N)], (best_cx, best_cy), best_side, best_ang

# EVOLVE-BLOCK-END
