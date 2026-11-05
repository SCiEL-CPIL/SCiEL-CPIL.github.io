"""Packing 12 unit hexagons in minimal regular hexagon."""
import numpy as np
from scipy.optimize import minimize
import time

N = 12
DEADLINE = 115.0

def _hex_vertices(cx, cy, angle_rad, scale=1.0):
    angles = angle_rad + np.arange(6) * np.pi / 3
    return np.column_stack([cx + scale * np.cos(angles), cy + scale * np.sin(angles)])

def _point_to_segment_dist(p, a, b):
    ab = b - a
    denom = np.dot(ab, ab)
    if denom < 1e-15:
        return np.linalg.norm(p - a)
    t = np.dot(p - a, ab) / denom
    t = max(0.0, min(1.0, t))
    closest = a + t * ab
    return np.linalg.norm(p - closest)

def _polygon_min_dist(v1, v2):
    """Min distance between two convex polygons. Negative if overlapping."""
    min_d = float('inf')
    n1, n2 = len(v1), len(v2)
    for i in range(n1):
        for j in range(n2):
            min_d = min(min_d, _point_to_segment_dist(v1[i], v2[j], v2[(j+1)%n2]))
            min_d = min(min_d, _point_to_segment_dist(v2[j], v1[i], v1[(i+1)%n1]))
    return min_d

def _point_outside_hex_dist(p, hex_v):
    """Distance from point p to outside of convex hexagon. 0 if inside."""
    n = len(hex_v)
    center = np.mean(hex_v, axis=0)
    min_dist = float('inf')
    inside = True
    for i in range(n):
        edge = hex_v[(i+1)%n] - hex_v[i]
        normal = np.array([-edge[1], edge[0]])
        norm = np.linalg.norm(normal)
        if norm < 1e-12:
            continue
        normal /= norm
        mid = (hex_v[i] + hex_v[(i+1)%n]) / 2
        if np.dot(normal, center - mid) < 0:
            normal = -normal
        d = np.dot(p - hex_v[i], normal)
        if d < -1e-10:
            inside = False
            min_dist = min(min_dist, -d)
    if inside:
        return 0.0
    for i in range(n):
        min_dist = min(min_dist, _point_to_segment_dist(p, hex_v[i], hex_v[(i+1)%n]))
    return min_dist

def _compute_score(pos, angles, outer_c, outer_side, outer_angle):
    """Compute objective: outer_side if valid, else penalty."""
    # Non-overlap check
    min_dist = float('inf')
    for i in range(N):
        vi = _hex_vertices(pos[i][0], pos[i][1], angles[i])
        for j in range(i+1, N):
            vj = _hex_vertices(pos[j][0], pos[j][1], angles[j])
            d = _polygon_min_dist(vi, vj)
            if d < min_dist:
                min_dist = d
    if min_dist < -0.005:
        return 1e6 + abs(min_dist) * 5000
    
    # Containment check
    outer_v = _hex_vertices(outer_c[0], outer_c[1], outer_angle, outer_side)
    max_viol = 0.0
    for i in range(N):
        vi = _hex_vertices(pos[i][0], pos[i][1], angles[i])
        for v in vi:
            viol = _point_outside_hex_dist(v, outer_v)
            if viol > max_viol:
                max_viol = viol
    if max_viol > 0.005:
        return outer_side + max_viol * 5000
    
    return outer_side

def construct():
    start_time = time.time()
    
    # Generate initial configurations
    configs = []
    
    # Hexagonal lattice patterns with varying spacing
    for d in np.arange(1.7, 2.3, 0.1):
        # 3x4 pattern
        pos = []
        for r in range(3):
            for c in range(4):
                x = c * d * np.sqrt(3) + (r % 2) * d * np.sqrt(3) / 2
                y = r * d * 1.5
                pos.append([x, y])
        configs.append(np.array(pos))
        
        # 2x6 pattern
        pos = []
        for r in range(2):
            for c in range(6):
                x = c * d * np.sqrt(3) + (r % 2) * d * np.sqrt(3) / 2
                y = r * d * 1.5
                pos.append([x, y])
        configs.append(np.array(pos))
        
        # 4x3 pattern
        pos = []
        for r in range(4):
            for c in range(3):
                x = c * d * np.sqrt(3) + (r % 2) * d * np.sqrt(3) / 2
                y = r * d * 1.5
                pos.append([x, y])
        configs.append(np.array(pos))
    
    # Hexagonal ring pattern
    for d in np.arange(1.7, 2.3, 0.1):
        pos = [[0, 0]]
        for k in range(6):
            a = k * np.pi / 3
            pos.append([d * np.cos(a), d * np.sin(a)])
        for k in range(6):
            a = k * np.pi / 3 + np.pi / 6
            pos.append([2*d * np.cos(a), 2*d * np.sin(a)])
        configs.append(np.array(pos[:12]))
    
    best_side = float('inf')
    best_params = None
    
    for idx, config in enumerate(configs):
        if time.time() - start_time > DEADLINE:
            break
        
        # Center the config
        centroid = np.mean(config, axis=0)
        pos0 = config - centroid
        angles0 = np.zeros(N)
        
        # Estimate outer side
        farthest = np.max(np.linalg.norm(pos0, axis=1))
        est_side = farthest + 1.2
        
        # Parameters: positions (2N), angles (N), outer center (2), outer side (1), outer angle (1)
        x0 = np.concatenate([pos0.flatten(), angles0, [0.0, 0.0, est_side, 0.0]])
        
        def obj(params):
            if time.time() - start_time > DEADLINE:
                return 1e10
            pos = params[:2*N].reshape(N, 2)
            ang = params[2*N:3*N]
            oc = params[3*N:3*N+2]
            os_ = params[3*N+2]
            oa = params[3*N+3]
            return _compute_score(pos.tolist(), ang.tolist(), oc, os_, oa)
        
        try:
            res = minimize(obj, x0, method='Nelder-Mead',
                          options={'maxiter': 150, 'xatol': 1e-5, 'fatol': 1e-5, 'adaptive': True})
            if res.fun < best_side and res.fun < 50:
                best_side = res.fun
                best_params = res.x.copy()
        except:
            pass
    
    if best_params is None:
        # Fallback
        config = configs[0]
        centroid = np.mean(config, axis=0)
        pos0 = config - centroid
        farthest = np.max(np.linalg.norm(pos0, axis=1))
        side = farthest + 1.5
        return [[pos0[i][0], pos0[i][1], 0.0] for i in range(N)], (0.0, 0.0), side, 0.0
    
    pos = best_params[:2*N].reshape(N, 2)
    ang = best_params[2*N:3*N]
    oc = best_params[3*N:3*N+2]
    os_ = best_params[3*N+2]
    oa = best_params[3*N+3]
    
    # Final validation
    score = _compute_score(pos.tolist(), ang.tolist(), oc, os_, oa)
    if score > 50:
        config = configs[0]
        centroid = np.mean(config, axis=0)
        pos0 = config - centroid
        farthest = np.max(np.linalg.norm(pos0, axis=1))
        side = farthest + 1.5
        return [[pos0[i][0], pos0[i][1], 0.0] for i in range(N)], (0.0, 0.0), side, 0.0
    
    return [[pos[i][0], pos[i][1], np.degrees(ang[i])] for i in range(N)], (oc[0], oc[1]), os_, np.degrees(oa)

# EVOLVE-BLOCK-START
# (construct defined above)
# EVOLVE-BLOCK-END
