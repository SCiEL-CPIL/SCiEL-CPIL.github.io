import numpy as np
from scipy.optimize import minimize
import time

N = 11
DEADLINE = 145.0

def _hex_vertices(cx, cy, s, angle_rad):
    verts = []
    for k in range(6):
        theta = angle_rad + k * np.pi / 3.0
        verts.append([cx + s * np.cos(theta), cy + s * np.sin(theta)])
    return np.array(verts)

def _point_in_hex_dist(p, cx, cy, s, angle_rad):
    """Return signed distance from point p to hexagon boundary (positive = outside)."""
    dx, dy = p[0] - cx, p[1] - cy
    # Transform to hexagon's local frame
    ca, sa = np.cos(-angle_rad), np.sin(-angle_rad)
    lx = dx * ca - dy * sa
    ly = dx * sa + dy * ca
    # In local frame, hexagon has flat top/bottom if angle=0
    # Vertices at angles 0, 60, 120, 180, 240, 300
    # Edges: 6 edges, each at distance s*sqrt(3)/2 from center
    inradius = s * np.sqrt(3.0) / 2.0
    min_d = np.inf
    for k in range(6):
        theta = (k + 0.5) * np.pi / 3.0
        nx, ny = np.cos(theta), np.sin(theta)
        d = inradius - (lx * nx + ly * ny)
        min_d = min(min_d, d)
    return min_d  # positive = inside, negative = outside

def _hex_hex_min_dist(v1, v2):
    """Min distance between two hexagons (approximate via vertices)."""
    min_d = np.inf
    for p in v1:
        d = np.min(np.sqrt(np.sum((v2 - p)**2, axis=1)))
        min_d = min(min_d, d)
    return min_d

def _is_valid_packing(inner_params, outer_s, outer_cx, outer_cy, outer_angle_rad):
    """Check if packing is valid: no overlaps, all inside outer hex."""
    xs = inner_params[0::3]
    ys = inner_params[1::3]
    angles = inner_params[2::3]
    
    verts_list = []
    for i in range(N):
        v = _hex_vertices(xs[i], ys[i], 1.0, angles[i])
        verts_list.append(v)
        # Check all vertices inside outer hex
        for p in v:
            d = _point_in_hex_dist(p, outer_cx, outer_cy, outer_s, outer_angle_rad)
            if d < -1e-6:
                return False
    
    # Check no overlaps
    for i in range(N):
        for j in range(i+1, N):
            # Check if any vertex of i is inside j or vice versa
            for p in verts_list[i]:
                d = _point_in_hex_dist(p, xs[j], ys[j], 1.0, angles[j])
                if d > 1e-6:  # inside
                    return False
            for p in verts_list[j]:
                d = _point_in_hex_dist(p, xs[i], ys[i], 1.0, angles[i])
                if d > 1e-6:
                    return False
    return True

def _packing_score(inner_params, outer_s, outer_cx, outer_cy, outer_angle_rad):
    """Score: 0 if valid, positive penalty if invalid."""
    if _is_valid_packing(inner_params, outer_s, outer_cx, outer_cy, outer_angle_rad):
        return 0.0
    # Compute violation magnitude
    xs = inner_params[0::3]
    ys = inner_params[1::3]
    angles = inner_params[2::3]
    penalty = 0.0
    for i in range(N):
        v = _hex_vertices(xs[i], ys[i], 1.0, angles[i])
        for p in v:
            d = _point_in_hex_dist(p, outer_cx, outer_cy, outer_s, outer_angle_rad)
            if d < 0:
                penalty += -d
    for i in range(N):
        for j in range(i+1, N):
            for p in _hex_vertices(xs[i], ys[i], 1.0, angles[i]):
                d = _point_in_hex_dist(p, xs[j], ys[j], 1.0, angles[j])
                if d > 0:
                    penalty += d
            for p in _hex_vertices(xs[j], ys[j], 1.0, angles[j]):
                d = _point_in_hex_dist(p, xs[i], ys[i], 1.0, angles[i])
                if d > 0:
                    penalty += d
    return penalty

def construct():
    t_start = time.time()
    
    # Generate several initial configurations
    configs = []
    
    # Config 1: Honeycomb cluster (1+6+4)
    d = np.sqrt(3.0)  # edge-to-edge center distance
    pos1 = [(0.0, 0.0, 0.0)]
    for k in range(6):
        theta = k * np.pi / 3.0
        pos1.append((d * np.cos(theta), d * np.sin(theta), 0.0))
    # 4 more at outer ring positions
    for k in range(4):
        theta = (2*k + 1) * np.pi / 6.0
        pos1.append((2*d * np.cos(theta), 2*d * np.sin(theta), 0.0))
    configs.append(pos1[:N])
    
    # Config 2: Linear arrangement with offsets
    pos2 = []
    for i in range(N):
        pos2.append((i * d, 0.0, 0.0))
    configs.append(pos2)
    
    # Config 3: Two rows
    pos3 = []
    for i in range(6):
        pos3.append((i * d, 0.0, 0.0))
    for i in range(5):
        pos3.append((i * d + d/2, d * 0.866, 0.0))  # sqrt(3)/2 * d
    configs.append(pos3[:N])
    
    # Config 4: Hexagonal with rotations
    pos4 = [(0.0, 0.0, 0.0)]
    for k in range(6):
        theta = k * np.pi / 3.0
        pos4.append((d * np.cos(theta), d * np.sin(theta), np.pi/6))
    for k in range(4):
        theta = (2*k + 1) * np.pi / 6.0
        pos4.append((2*d * np.cos(theta), 2*d * np.sin(theta), np.pi/6))
    configs.append(pos4[:N])
    
    best_score = np.inf
    best_result = None
    
    for config in configs:
        if time.time() - t_start > DEADLINE:
            break
        
        # Initial outer hex size
        max_r = max(np.sqrt(x**2 + y**2) for x, y, a in config)
        outer_s_init = max_r / (np.sqrt(3.0)/2.0) + 2.0
        
        # Try to shrink outer_s
        outer_s = outer_s_init
        inner_params = np.array([p for x, y, a in config for p in [x, y, a]])
        
        # Binary search on outer_s
        lo, hi = outer_s_init * 0.3, outer_s_init
        for _ in range(8):
            if time.time() - t_start > DEADLINE - 5:
                break
            mid = (lo + hi) / 2.0
            
            # Try to optimize inner positions for this outer_s
            def obj(params):
                return _packing_score(params, mid, 0.0, 0.0, 0.0)
            
            try:
                res = minimize(obj, inner_params, method='L-BFGS-B',
                              options={'maxiter': 40, 'ftol': 1e-8})
                if res.fun < obj(inner_params):
                    inner_params = res.x
            except:
                pass
            
            if _is_valid_packing(inner_params, mid, 0.0, 0.0, 0.0):
                hi = mid
                best_score = min(best_score, mid)
            else:
                lo = mid
        
        if best_score < outer_s_init:
            # Store best
            xs = inner_params[0::3]
            ys = inner_params[1::3]
            angles = inner_params[2::3]
            inner = [[float(xs[i]), float(ys[i]), float(np.degrees(angles[i]))] for i in range(N)]
            best_result = (inner, (0.0, 0.0), float(best_score), 0.0)
    
    if best_result is None:
        # Fallback: use first config with large outer hex
        config = configs[0]
        inner = [[float(x), float(y), float(np.degrees(a))] for x, y, a in config]
        max_r = max(np.sqrt(x**2 + y**2) for x, y, a in config)
        outer_s = max_r / (np.sqrt(3.0)/2.0) + 2.0
        return inner, (0.0, 0.0), float(outer_s), 0.0
    
    return best_result

