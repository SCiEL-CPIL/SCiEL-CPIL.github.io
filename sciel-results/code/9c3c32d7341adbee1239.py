import numpy as np
from scipy.optimize import minimize
import time

N = 11

def _hex_vertices(cx, cy, s, angle_deg):
    """Return 6 vertices of a regular hexagon with center (cx,cy), side s, rotation angle_deg."""
    a = np.radians(angle_deg)
    angles = a + np.arange(6) * np.pi / 3
    return np.column_stack([cx + s * np.cos(angles), cy + s * np.sin(angles)])

def _point_in_hex_margin(px, py, cx, cy, R, angle_deg):
    """Return margin (positive if inside) for point in hexagon with center (cx,cy), side R, rotation angle_deg."""
    a = np.radians(angle_deg)
    apothem = R * np.sqrt(3) / 2
    min_margin = np.inf
    for k in range(6):
        theta = a + k * np.pi / 3
        nx = np.cos(theta)
        ny = np.sin(theta)
        d = (px - cx) * nx + (py - cy) * ny
        margin = apothem - d
        if margin < min_margin:
            min_margin = margin
    return min_margin

def _min_dist_hexes(cx1, cy1, a1, cx2, cy2, a2):
    """Approximate minimum distance between two unit hexagons."""
    v1 = _hex_vertices(cx1, cy1, 1.0, a1)
    v2 = _hex_vertices(cx2, cy2, 1.0, a2)
    min_d = np.inf
    for p in v1:
        for q in v2:
            d = np.sqrt((p[0]-q[0])**2 + (p[1]-q[1])**2)
            if d < min_d:
                min_d = d
    return min_d

def _objective(params, N):
    R = params[-1]
    if R <= 0.5:
        return 1e6
    
    total_pen = 0.0
    centers = []
    angles = []
    for i in range(N):
        x = params[3*i]
        y = params[3*i+1]
        a = params[3*i+2]
        centers.append((x, y))
        angles.append(a)
    
    # Inner hexes must be inside outer hex (center origin, side R, angle 0)
    for i in range(N):
        x, y = centers[i]
        a = angles[i]
        verts = _hex_vertices(x, y, 1.0, a)
        for v in verts:
            m = _point_in_hex_margin(v[0], v[1], 0.0, 0.0, R, 0.0)
            if m < 0:
                total_pen += m * m
    
    # Pairwise non-overlap
    for i in range(N):
        for j in range(i+1, N):
            d = _min_dist_hexes(centers[i][0], centers[i][1], angles[i],
                                centers[j][0], centers[j][1], angles[j])
            # Two unit hexagons need center distance > ~2 for same orientation
            # Use threshold of 1.8 to be safe with rotation
            if d < 1.8:
                total_pen += (1.8 - d) ** 2
    
    return R + 50.0 * total_pen

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 135.0
    
    sqrt3 = np.sqrt(3)
    
    # Hexagonal lattice initial guess
    init_positions = [
        (0, 0),
        (2, 0),
        (1, sqrt3),
        (-1, sqrt3),
        (-2, 0),
        (-1, -sqrt3),
        (1, -sqrt3),
        (3, sqrt3),
        (0, 2*sqrt3),
        (-3, sqrt3),
        (0, -2*sqrt3),
    ]
    
    best_params = None
    best_obj = np.inf
    
    np.random.seed(42)
    
    n_restarts = 8
    for restart in range(n_restarts):
        if time.time() > deadline:
            break
        
        if restart == 0:
            x0_list = []
            for x, y in init_positions:
                x0_list.extend([x, y, 0.0])
            x0_list.append(6.0)
            x0 = np.array(x0_list)
        else:
            x0_list = []
            for x, y in init_positions:
                x0_list.extend([x + np.random.normal(0, 1.0), 
                               y + np.random.normal(0, 1.0), 
                               np.random.uniform(-180, 180)])
            x0_list.append(5.0 + np.random.uniform(-1.0, 2.0))
            x0 = np.array(x0_list)
        
        try:
            res = minimize(
                _objective,
                x0,
                args=(N,),
                method='L-BFGS-B',
                options={'maxiter': 300, 'ftol': 1e-9, 'gtol': 1e-7}
            )
            
            if res.fun < best_obj:
                best_obj = res.fun
                best_params = res.x
        except Exception:
            pass
    
    if best_params is None:
        # Fallback to seed
        inner = [[4.0 * k, 0.0, 0.0] for k in range(-(N // 2), N - N // 2)]
        return inner, (0.0, 0.0), 40.0, 0.0
    
    inner = []
    for i in range(N):
        x = float(best_params[3*i])
        y = float(best_params[3*i+1])
        a = float(best_params[3*i+2])
        inner.append([x, y, a])
    
    R = float(best_params[-1])
    center = (0.0, 0.0)
    angle = 0.0
    
    return inner, center, R, angle
# EVOLVE-BLOCK-END
