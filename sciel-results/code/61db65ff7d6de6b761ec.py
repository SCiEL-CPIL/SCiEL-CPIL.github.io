import numpy as np
from scipy.optimize import differential_evolution, minimize
import time

N = 12
INRADIUS = np.sqrt(3.0) / 2.0  # inradius of unit hexagon

def _hex_vertices(cx, cy, side, angle_deg):
    angle_rad = np.radians(angle_deg)
    verts = []
    for i in range(6):
        a = angle_rad + i * np.pi / 3.0
        verts.append((cx + side * np.cos(a), cy + side * np.sin(a)))
    return np.array(verts)

def _hex_overlap(v1, v2):
    """SAT test for two convex hexagons. Returns True if they overlap."""
    for verts in (v1, v2):
        for i in range(6):
            p1 = verts[i]
            p2 = verts[(i + 1) % 6]
            edge = p2 - p1
            normal = np.array([-edge[1], edge[0]])
            proj1 = v1 @ normal
            proj2 = v2 @ normal
            if proj1.max() < proj2.min() or proj2.max() < proj1.min():
                return False
    return True

def _min_enclosing_hex_side(positions, center, outer_angle_deg):
    """Minimal side length of regular hexagon enclosing all positions."""
    outer_angle_rad = np.radians(outer_angle_deg)
    normals = np.array([[np.cos(outer_angle_rad + i * np.pi / 3.0),
                         np.sin(outer_angle_rad + i * np.pi / 3.0)] for i in range(6)])
    cx, cy = center
    max_support = 0.0
    for pos in positions:
        rel = np.array([pos[0] - cx, pos[1] - cy])
        supports = np.abs(rel @ normals.T)
        max_support = max(max_support, supports.max())
    return max_support / INRADIUS

def _score_config(positions, shared_angle):
    """Compute outer side length with overlap penalty."""
    verts = [_hex_vertices(positions[i, 0], positions[i, 1], 1.0, shared_angle) 
             for i in range(N)]
    
    overlap_count = 0
    for i in range(N):
        for j in range(i + 1, N):
            if _hex_overlap(verts[i], verts[j]):
                overlap_count += 1
    
    if overlap_count > 0:
        return 100.0 + overlap_count * 10.0  # Heavy penalty
    
    center = (positions[:, 0].mean(), positions[:, 1].mean())
    best_side = float('inf')
    for oa in [0, 30, 60, 90, 120, 150]:
        s = _min_enclosing_hex_side(positions, center, oa)
        best_side = min(best_side, s)
    return best_side

def construct():
    start_time = time.time()
    deadline = start_time + 130.0
    
    # Initial: center + 6 ring + 5 outer
    pos0 = [[0.0, 0.0]]
    for i in range(6):
        a = i * np.pi / 3.0
        pos0.append([2 * INRADIUS * np.cos(a), 2 * INRADIUS * np.sin(a)])
    for i in range(5):
        a = i * 2 * np.pi / 5.0 + np.pi / 5
        pos0.append([4 * INRADIUS * np.cos(a), 4 * INRADIUS * np.sin(a)])
    pos0 = np.array(pos0)
    init_params = np.concatenate([pos0.flatten(), [0.0]])
    
    best_params = init_params.copy()
    best_score = _score_config(pos0, 0.0)
    
    # Nelder-Mead
    def obj(p):
        if time.time() > deadline:
            return 1e6
        pos = p[:2*N].reshape(N, 2)
        ang = p[2*N]
        return _score_config(pos, ang)
    
    try:
        res = minimize(obj, init_params, method='Nelder-Mead',
                      options={'maxiter': 300, 'xatol': 1e-4, 'fatol': 1e-4})
        if res.fun < best_score:
            best_score = res.fun
            best_params = res.x
    except Exception:
        pass
    
    # Differential evolution if time permits
    if time.time() < deadline - 30:
        bounds = [(-8, 8)] * (2*N) + [(-90, 90)]
        try:
            de_res = differential_evolution(obj, bounds, maxiter=30, popsize=12,
                                            seed=42, tol=1e-3, polish=False)
            if de_res.fun < best_score:
                best_score = de_res.fun
                best_params = de_res.x
        except Exception:
            pass
    
    # Final polish
    if time.time() < deadline - 15:
        try:
            res2 = minimize(obj, best_params, method='Nelder-Mead',
                           options={'maxiter': 200, 'xatol': 1e-5, 'fatol': 1e-5})
            if res2.fun < best_score:
                best_score = res2.fun
                best_params = res2.x
        except Exception:
            pass
    
    positions = best_params[:2*N].reshape(N, 2)
    shared_angle = best_params[2*N]
    
    # Optimize center and outer angle
    center = (positions[:, 0].mean(), positions[:, 1].mean())
    best_side = float('inf')
    best_oa = 0.0
    for oa in np.arange(0, 180, 2):
        s = _min_enclosing_hex_side(positions, center, oa)
        if s < best_side:
            best_side = s
            best_oa = oa
    for oa in np.arange(best_oa - 2, best_oa + 2, 0.1):
        s = _min_enclosing_hex_side(positions, center, oa)
        if s < best_side:
            best_side = s
            best_oa = oa
    
    # Fine-tune center
    for dx in np.arange(-0.3, 0.3, 0.05):
        for dy in np.arange(-0.3, 0.3, 0.05):
            c = (center[0] + dx, center[1] + dy)
            s = _min_enclosing_hex_side(positions, c, best_oa)
            if s < best_side:
                best_side = s
                center = c
    
    inner = [[float(positions[i, 0]), float(positions[i, 1]), float(shared_angle)] 
             for i in range(N)]
    return inner, (float(center[0]), float(center[1])), float(best_side), float(best_oa)

