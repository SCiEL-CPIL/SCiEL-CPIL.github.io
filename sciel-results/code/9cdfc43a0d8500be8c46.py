import numpy as np
from scipy.optimize import minimize
import time

N = 12
SQRT3 = np.sqrt(3)

def get_hex_vertices(x, y, angle=0.0):
    """
    Returns vertices of a unit regular hexagon centered at (x, y).
    Angle 0 corresponds to a vertex at (1, 0).
    """
    angles = np.deg2rad([0, 60, 120, 180, 240, 300]) + np.deg2rad(angle)
    cos_a = np.cos(angles)
    sin_a = np.sin(angles)
    xs = x + cos_a
    ys = y + sin_a
    return np.column_stack((xs, ys))

def s_required(x, y):
    """
    Calculates the minimum side length S of a regular hexagon 
    (centered at origin, vertex at (S,0)) that contains point (x,y).
    """
    abs_x = np.abs(x)
    abs_y = np.abs(y)
    # Constraints:
    # 1. |x| <= S
    # 2. |y| <= sqrt(3)/2 * S  (for x=0) -> S >= 2/sqrt(3) * |y|
    # 3. The side boundaries:
    #    y = sqrt(3)/2 * (S - x)  => x + 2/sqrt(3)y = S  (for x>0, y>0)
    #    So S >= x + 2/sqrt(3)y
    #    By symmetry: S >= |x| + 2/sqrt(3)|y| is NOT correct for all quadrants?
    #    Let's check the line equation:
    #    Line through (S, 0) and (S/2, S*sqrt(3)/2).
    #    Slope = (S*sqrt(3)/2 - 0) / (S/2 - S) = (S*sqrt(3)/2) / (-S/2) = -sqrt(3).
    #    y - 0 = -sqrt(3) (x - S)
    #    y = -sqrt(3)x + S*sqrt(3)
    #    sqrt(3)x + y = S*sqrt(3)
    #    x + y/sqrt(3) = S
    #    So S >= x + y/sqrt(3)
    #    Using absolute values: S >= |x| + |y|/sqrt(3)
    
    s1 = abs_x
    s2 = abs_x + abs_y / SQRT3
    
    return np.maximum(s1, s2)

def objective(params):
    """
    Minimize the outer side length + overlap penalty.
    params: [x1, y1, x2, y2, ..., x12, y12]
    """
    xs = params[0::2]
    ys = params[1::2]
    
    max_s = 0.0
    # Calculate required S for all vertices
    for i in range(N):
        verts = get_hex_vertices(xs[i], ys[i], 0.0)
        vx = verts[:, 0]
        vy = verts[:, 1]
        s_vals = s_required(vx, vy)
        max_s = max(max_s, np.max(s_vals))
        
    # Overlap penalty
    penalty = 0.0
    # Min distance between centers of non-overlapping unit hexagons (aligned) is sqrt(3)
    min_dist = SQRT3
    for i in range(N):
        for j in range(i + 1, N):
            dx = xs[i] - xs[j]
            dy = ys[i] - ys[j]
            d = np.sqrt(dx*dx + dy*dy)
            if d < min_dist:
                penalty += (min_dist - d)**2
                
    return max_s + 100.0 * penalty

def construct():
    start_time = time.time()
    timeout = 120.0  # Safety margin for 150s limit
    
    best_score = float('inf')
    best_params = None
    best_angle = 0.0
    
    # Initializations
    inits = []
    
    # Init 1: 2x6 Grid
    # Hex grid spacing
    dx = SQRT3
    dy = 1.5
    # Center the grid
    # 6 cols, 2 rows
    xs = []
    ys = []
    for r in range(2):
        for c in range(6):
            x = (c - 2.5) * dx
            y = (r - 0.5) * dy
            xs.append(x)
            ys.append(y)
    params0 = np.array(xs + ys)
    inits.append(params0)
    
    # Init 2: Hexagonal Cluster (7 + 5)
    # Center (0,0)
    # 6 neighbors
    # 5 more attached
    xs2 = [0.0]
    ys2 = [0.0]
    for k in range(6):
        angle = np.deg2rad(60 * k)
        xs2.append(SQRT3 * np.cos(angle))
        ys2.append(SQRT3 * np.sin(angle))
    # Add 5 more around the perimeter
    # Just place them roughly
    for k in range(5):
        angle = np.deg2rad(60 * k + 30)
        # Distance 2*SQRT3 from center?
        xs2.append(2.0 * SQRT3 * np.cos(angle))
        ys2.append(2.0 * SQRT3 * np.sin(angle))
    params1 = np.array(xs2 + ys2)
    inits.append(params1)
    
    # Init 3: Random
    np.random.seed(42)
    params2 = np.random.uniform(-5, 5, 24)
    inits.append(params2)

    results = []
    
    for i, p0 in enumerate(inits):
        if time.time() - start_time > timeout - 10:
            break
            
        try:
            res = minimize(objective, p0, method='Nelder-Mead', 
                           options={'xatol': 1e-6, 'fatol': 1e-6, 'maxiter': 1000, 'maxfev': 2000})
            if res.success or res.fun < best_score:
                # Verify no overlaps in the final result
                xs = res.x[0::2]
                ys = res.x[1::2]
                valid = True
                for u in range(N):
                    for v in range(u+1, N):
                        d = np.sqrt((xs[u]-xs[v])**2 + (ys[u]-ys[v])**2)
                        if d < SQRT3 - 1e-5:
                            valid = False
                if valid:
                    # Calculate actual side
                    max_s = 0.0
                    for u in range(N):
                        verts = get_hex_vertices(xs[u], ys[u], 0.0)
                        s_vals = s_required(verts[:, 0], verts[:, 1])
                        max_s = max(max_s, np.max(s_vals))
                    
                    if max_s < best_score:
                        best_score = max_s
                        best_params = res.x
                        best_angle = 0.0
        except Exception as e:
            pass

    if best_params is None:
        # Fallback to initial best guess
        # Use Init 1
        xs = inits[0][0::2]
        ys = inits[0][1::2]
        # Calculate side
        max_s = 0.0
        for u in range(N):
            verts = get_hex_vertices(xs[u], ys[u], 0.0)
            s_vals = s_required(verts[:, 0], verts[:, 1])
            max_s = max(max_s, np.max(s_vals))
        best_score = max_s
        best_params = inits[0]
        best_angle = 0.0

    inner = []
    for i in range(N):
        x = best_params[2*i]
        y = best_params[2*i+1]
        inner.append([x, y, 0.0])
        
    return inner, (0.0, 0.0), best_score, best_angle

