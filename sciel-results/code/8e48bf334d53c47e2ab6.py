import numpy as np
import time
from scipy.optimize import minimize

N = 12
_START = time.time()
_DEADLINE = _START + 115.0  # 115s safety margin out of 120s

def _hex_vertices(cx, cy, side=1.0, angle_deg=0.0):
    """Return 6 vertices of a regular hexagon centered at (cx,cy) with given rotation."""
    ang = np.radians(angle_deg)
    # Regular hexagon vertices at angles 0, 60, 120, 180, 240, 300 degrees (flat-top)
    base = np.array([np.cos(np.radians(60*k)), np.sin(np.radians(60*k))] for k in range(6))
    c, s = np.cos(ang), np.sin(ang)
    R = np.array([[c, -s], [s, c]])
    return base @ R.T * side + np.array([cx, cy])

def _hex_vertices_flat(cx, cy, side=1.0, angle_deg=0.0):
    """Flat-top hexagon: vertices at 0,60,...,300 degrees."""
    ang = np.radians(angle_deg)
    base = np.array([[np.cos(np.radians(60*k)), np.sin(np.radians(60*k))] for k in range(6)])
    c, s = np.cos(ang), np.sin(ang)
    R = np.array([[c, -s], [s, c]])
    return (base @ R.T) * side + np.array([cx, cy])

def _all_inner_vertices(inner):
    """Compute all vertices of all inner hexagons. inner: list of [x,y,angle_deg]."""
    verts = []
    for x, y, a in inner:
        v = _hex_vertices_flat(x, y, 1.0, a)
        verts.append(v)
    return np.vstack(verts)  # shape (72, 2)

def _support_function(pts, direction):
    """Max dot product of pts with direction (unit vector)."""
    return np.max(pts @ direction)

def _min_outer_side(pts, center, angle_deg):
    """Find minimum side length of a regular hexagon (rotated by angle_deg, centered at center)
    that contains all points in pts."""
    ang = np.radians(angle_deg)
    # Hexagon with flat-top orientation at angle ang:
    # The 6 outward normals (perpendicular to edges) are at angles ang+90, ang+150, ...
    # Actually for a regular hexagon, the support in 6 directions gives the side length.
    # A regular hexagon with side s has inradius = s*sqrt(3)/2.
    # The 6 face normals are at angles: angle_deg + 90 + 60*k for k=0..5
    # Wait, let me think more carefully.
    # A flat-top regular hexagon centered at origin has vertices at angles 0,60,120,180,240,300.
    # The edges are between consecutive vertices. The outward normals to the edges are at
    # angles 30, 90, 150, 210, 270, 330 (midpoint angles of adjacent vertex angles).
    # Actually: edge from angle 0 to 60 has normal at 30 degrees.
    # So normals are at 30+60k for k=0..5.
    # Inradius = s * sqrt(3)/2, so s = 2*inradius/sqrt(3)
    # The hexagon contains all points iff for each normal direction n_k:
    #   max_i (p_i - center) . n_k <= inradius = s * sqrt(3)/2
    # So s >= 2/sqrt(3) * max_k max_i (p_i - center) . n_k
    
    normals = []
    for k in range(6):
        theta = np.radians(angle_deg + 30 + 60*k)
        normals.append(np.array([np.cos(theta), np.sin(theta)]))
    
    rel = pts - np.array(center)
    max_support = 0.0
    for n in normals:
        sup = np.max(rel @ n)
        if sup > max_support:
            max_support = sup
    
    side = 2.0 * max_support / np.sqrt(3.0)
    return max(side, 0.0)

def _objective(params, pts_cache=None):
    """Objective: minimize the outer hexagon side length.
    params: flat array of [x0,y0,a0, x1,y1,a1, ..., x11,y11,a11]
    Returns the side length (or large penalty if invalid)."""
    inner = [[params[3*i], params[3*i+1], params[3*i+2]] for i in range(N)]
    pts = _all_inner_vertices(inner)
    # We'll optimize center and angle separately in outer loop
    # For now, just return a dummy (this is used in inner SA)
    return 0.0

def _sa_optimize_inner(init_inner, outer_center, outer_angle, max_iters=8000, T0=0.8, T1=0.001):
    """Simulated annealing to optimize inner hexagon positions.
    Objective: minimize outer side length given fixed outer center and angle."""
    params = np.array([v for hex in init_inner for v in hex], dtype=np.float64)
    cur_inner = [[params[3*i], params[3*i+1], params[3*i+2]] for i in range(N)]
    
    def _eval(inner):
        pts = _all_inner_vertices(inner)
        return _min_outer_side(pts, outer_center, outer_angle)
    
    cur_obj = _eval(cur_inner)
    best_inner = [h[:] for h in cur_inner]
    best_obj = cur_obj
    
    T = T0
    cooling = (T1 / T0) ** (1.0 / max(max_iters, 1))
    
    for it in range(max_iters):
        if time.time() > _DEADLINE - 5:
            break
        if it % 500 == 0 and it > 0:
            T = max(T * cooling ** 500, T1)
        
        # Propose a move: pick random hexagon, perturb position and/or angle
        new_inner = [h[:] for h in cur_inner]
        idx = np.random.randint(N)
        new_inner[idx][0] += np.random.normal(0, 0.3)
        new_inner[idx][1] += np.random.normal(0, 0.3)
        if np.random.random() < 0.3:
            new_inner[idx][2] += np.random.normal(0, 10.0)
        
        new_obj = _eval(new_inner)
        
        delta = new_obj - cur_obj
        if delta < 0 or np.random.random() < np.exp(-delta / max(T, 1e-10)):
            cur_inner = new_inner
            cur_obj = new_obj
            if cur_obj < best_obj:
                best_obj = cur_obj
                best_inner = [h[:] for h in cur_inner]
        
        T *= cooling
    
    return best_inner, best_obj

def _optimize_outer_side(pts, init_center=None, init_angle=0.0):
    """Optimize outer hexagon center and angle to minimize side length containing all pts."""
    if init_center is None:
        init_center = (0.0, 0.0)
    
    def obj(vars):
        cx, cy, ang = vars[0], vars[1], vars[2]
        return _min_outer_side(pts, (cx, cy), ang)
    
    x0 = np.array([init_center[0], init_center[1], np.radians(init_angle)])
    
    # Try multiple starting angles
    best_side = float('inf')
    best_vars = None
    for ang0 in [0, 30, 60, 15, 45]:
        x0 = np.array([init_center[0], init_center[1], np.radians(ang0)])
        res = minimize(obj, x0, method='Nelder-Mead',
                      options={'maxiter': 500, 'xatol': 1e-5, 'fatol': 1e-5, 'maxfev': 1000})
        if res.fun < best_side:
            best_side = res.fun
            best_vars = res.x
    
    if best_vars is None:
        return init_center, init_angle, best_side
    
    return (best_vars[0], best_vars[1]), np.degrees(best_vars[2]), best_side

def construct():
    t0 = time.time()
    
    # Generate diverse initial configurations
    configs = []
    
    # Config 1: Linear row (seed-like but tighter)
    cfg1 = []
    for k in range(N):
        x = (k - (N-1)/2.0) * 2.0
        y = 0.0
        cfg1.append([x, y, 0.0])
    configs.append(cfg1)
    
    # Config 2: 3x4 grid (staggered)
    cfg2 = []
    for row in range(3):
        for col in range(4):
            x = (col - 1.5) * 2.0
            y = (row - 1.0) * np.sqrt(3)
            cfg2.append([x, y, 0.0])
    configs.append(cfg2)
    
    # Config 3: Center + ring of 6 + outer ring of 5
    cfg3 = [[0.0, 0.0, 0.0]]
    for k in range(6):
        ang = k * 60.0
        r = 2.0
        cfg3.append([r*np.cos(np.radians(ang)), r*np.sin(np.radians(ang)), 0.0])
    for k in range(5):
        ang = 30.0 + k * 60.0 + 12.0  # stagger
        r = 3.464  # 2*sqrt(3)
        cfg3.append([r*np.cos(np.radians(ang)), r*np.sin(np.radians(ang)), 0.0])
    configs.append(cfg3)
    
    # Config 4: 4x3 grid
    cfg4 = []
    for row in range(4):
        for col in range(3):
            x = (col - 1.0) * 2.0
            y = (row - 1.5) * np.sqrt(3)
            cfg4.append([x, y, 0.0])
    configs.append(cfg4)
    
    # Config 5: Hexagonal packing (center + 6 at distance 2 + 5 more)
    cfg5 = [[0.0, 0.0, 0.0]]
    # Inner ring at r=2
    for k in range(6):
        ang = 60.0 * k
        cfg5.append([2.0*np.cos(np.radians(ang)), 2.0*np.sin(np.radians(ang)), 0.0])
    # Place remaining 5 around
    for k in range(5):
        ang = 30.0 + 60.0*k
        r = 3.5
        cfg5.append([r*np.cos(np.radians(ang)), r*np.sin(np.radians(ang)), 0.0])
    configs.append(cfg5)
    
    # Config 6: Two rows of 6
    cfg6 = []
    for k in range(6):
        x = (k - 2.5) * 2.0
        cfg6.append([x, -np.sqrt(3)/2, 0.0])
        cfg6.append([x + 1.0, np.sqrt(3)/2, 0.0])
    configs.append(cfg6)
    
    best_score = float('inf')
    best_result = None
    
    for cfg_idx, inner_init in enumerate(configs):
        if time.time() > _DEADLINE - 10:
            break
        
        # Try different outer orientations
        for outer_angle in [0.0, 30.0, 15.0]:
            if time.time() > _DEADLINE - 8:
                break
            
            # Run SA on inner positions
            iters = max(3000, min(8000, int((_DEADLINE - time.time()) * 200)))
            sa_inner, sa_obj = _sa_optimize_inner(inner_init, (0.0, 0.0), outer_angle,
                                                   max_iters=iters, T0=0.5, T1=0.0005)
            
            if sa_obj >= 1000:
                continue
            
            # Now optimize outer center and angle
            pts = _all_inner_vertices(sa_inner)
            center, angle, side = _optimize_outer_side(pts, (0.0, 0.0), outer_angle)
            
            if side < best_score:
                best_score = side
                best_inner = [h[:] for h in sa_inner]
                best_result = (best_inner, center, side, angle)
        
        # If we have a good result, do a second round of refinement
        if best_score < 20 and time.time() < _DEADLINE - 15:
            # Refine with the best found so far
            if best_result is not None:
                refine_inner, refine_obj = _sa_optimize_inner(
                    best_result[0], best_result[1], best_result[3],
                    max_iters=5000, T0=0.2, T1=0.0001)
                if refine_obj < best_score:
                    pts = _all_inner_vertices(refine_inner)
                    center, angle, side = _optimize_outer_side(pts, best_result[1], best_result[3])
                    if side < best_score:
                        best_score = side
                        best_result = (refine_inner, center, side, angle)
    
    # Fallback
    if best_result is None:
        inner = [[(i % 4 - 1.5) * 2.0, (i // 4 - 1.0) * np.sqrt(3), 0.0] for i in range(N)]
        return inner, (0.0, 0.0), 10.0, 0.0
    
    return best_result
