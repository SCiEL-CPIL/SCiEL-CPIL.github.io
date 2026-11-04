import numpy as np
import time
from scipy.optimize import minimize

N = 12
DEADLINE = 145.0

def _hex_vertices(cx, cy, side=1.0, rot_deg=0.0):
    angles = np.radians(np.array([0, 60, 120, 180, 240, 300])) + np.radians(rot_deg)
    x = cx + side * np.cos(angles)
    y = cy + side * np.sin(angles)
    return np.column_stack((x, y))

def _hex_edge_normals(rot_deg):
    base = np.radians(np.array([30, 90, 150, 210, 270, 330])) + np.radians(rot_deg)
    return np.column_stack((np.cos(base), np.sin(base)))

def _sat_overlap_fast(v1, v2, n1, n2):
    min_overlap = float('inf')
    for k in range(6):
        n = n1[k]
        proj1 = v1 @ n
        proj2 = v2 @ n
        min1, max1 = np.min(proj1), np.max(proj1)
        min2, max2 = np.min(proj2), np.max(proj2)
        if max1 < min2 or max2 < min1:
            gap = min2 - max1 if min2 > max1 else min1 - max2
            return gap
        overlap = min(max1, max2) - max(min1, min2)
        if overlap < min_overlap:
            min_overlap = overlap
    for k in range(6):
        n = n2[k]
        proj1 = v1 @ n
        proj2 = v2 @ n
        min1, max1 = np.min(proj1), np.max(proj1)
        min2, max2 = np.min(proj2), np.max(proj2)
        if max1 < min2 or max2 < min1:
            gap = min2 - max1 if min2 > max1 else min1 - max2
            return gap
        overlap = min(max1, max2) - max(min1, min2)
        if overlap < min_overlap:
            min_overlap = overlap
    return -min_overlap

def _outer_hex_side_fast(points, outer_rot_deg):
    angles = np.radians(np.array([30, 90, 150, 210, 270, 330]) + outer_rot_deg)
    normals = np.column_stack((np.cos(angles), np.sin(angles)))
    proj = points @ normals.T
    max_proj = np.max(np.abs(proj))
    return (2.0 / np.sqrt(3.0)) * max_proj

def _check_disjoint(inner, eps=1e-9):
    n = len(inner)
    verts = []
    norms = []
    for i in range(n):
        x, y, rot = inner[i]
        v = _hex_vertices(x, y, 1.0, rot)
        nm = _hex_edge_normals(rot)
        verts.append(v)
        norms.append(nm)
    for i in range(n):
        for j in range(i + 1, n):
            d = _sat_overlap_fast(verts[i], verts[j], norms[i], norms[j])
            if d < -eps:
                return False
    return True

def _make_hex_grid(rows, cols, rot_offset=0.0, gap=0.0):
    pts = []
    sqrt3 = np.sqrt(3)
    spacing_x = 2.0 + gap
    spacing_y = sqrt3 + gap * 0.5
    for r in range(rows):
        y = r * spacing_y
        for c in range(cols):
            x = c * spacing_x
            if r % 2 == 1:
                x += spacing_x / 2.0
            pts.append([x, y, rot_offset])
    pts = np.array(pts)
    if len(pts) > 12:
        pts = pts[:12]
    elif len(pts) < 12:
        while len(pts) < 12:
            pts.append([0, 0, rot_offset])
    pts = np.array(pts)
    pts[:, 0] -= np.mean(pts[:, 0])
    pts[:, 1] -= np.mean(pts[:, 1])
    return pts

def _make_ring(gap=0.0):
    pts = []
    d1 = np.sqrt(3) + gap
    d2 = 2 * np.sqrt(3) + gap
    pts.append([0, 0, 0.0])
    for k in range(6):
        ang = np.radians(60 * k)
        pts.append([d1 * np.cos(ang), d1 * np.sin(ang), 0.0])
    for k in range(5):
        ang = np.radians(60 * k + 30)
        pts.append([d2 * np.cos(ang), d2 * np.sin(ang), 0.0])
    pts = np.array(pts)
    pts[:, 0] -= np.mean(pts[:, 0])
    pts[:, 1] -= np.mean(pts[:, 1])
    return pts

def _make_packed_cluster(gap=0.0):
    pts = []
    sqrt3 = np.sqrt(3)
    pts.append([0, 0, 0.0])
    for k in range(6):
        ang = np.radians(60 * k)
        pts.append([sqrt3 * np.cos(ang), sqrt3 * np.sin(ang), 0.0])
    for k in range(5):
        ang = np.radians(60 * k + 30)
        pts.append([2 * sqrt3 * np.cos(ang), 2 * sqrt3 * np.sin(ang), 0.0])
    pts = np.array(pts)
    pts[:, 0] -= np.mean(pts[:, 0])
    pts[:, 1] -= np.mean(pts[:, 1])
    return pts

def _make_strip(gap=0.0):
    pts = []
    sqrt3 = np.sqrt(3)
    for i in range(12):
        x = i * (2.0 + gap)
        y = 0 if i % 2 == 0 else sqrt3 * 0.5
        pts.append([x, y, 0.0])
    pts = np.array(pts)
    pts[:, 0] -= np.mean(pts[:, 0])
    pts[:, 1] -= np.mean(pts[:, 1])
    return pts

def _evaluate_config(inner, outer_rot_deg):
    all_pts = np.vstack([_hex_vertices(inner[i,0], inner[i,1], 1.0, inner[i,2]) for i in range(12)])
    ox = np.mean(all_pts[:, 0])
    oy = np.mean(all_pts[:, 1])
    
    if not _check_disjoint(inner):
        return float('inf'), False
    
    all_pts_centered = all_pts.copy()
    all_pts_centered[:, 0] -= ox
    all_pts_centered[:, 1] -= oy
    L = _outer_hex_side_fast(all_pts_centered, outer_rot_deg)
    return L, True

def _objective(p, t0, deadline):
    if time.time() - t0 > deadline:
        return 1e6
    
    inner = p[:36].reshape(12, 3)
    ox = p[36]
    oy = p[37]
    oang = p[38]
    
    verts_list = []
    norms_list = []
    for i in range(12):
        cx, cy, rot = inner[i]
        v = _hex_vertices(cx, cy, 1.0, rot)
        nm = _hex_edge_normals(rot)
        verts_list.append(v)
        norms_list.append(nm)
    
    penalty = 0.0
    for i in range(12):
        for j in range(i + 1, 12):
            d = _sat_overlap_fast(verts_list[i], verts_list[j], norms_list[i], norms_list[j])
            if d < 0:
                penalty += (-d) ** 2 * 100.0
    
    all_pts = np.vstack(verts_list)
    all_pts[:, 0] -= ox
    all_pts[:, 1] -= oy
    
    L = _outer_hex_side_fast(all_pts, oang)
    
    return L + penalty

def _polish(inner, t0, deadline, oang_init=0.0):
    if time.time() - t0 > deadline:
        return None
    
    all_pts = np.vstack([_hex_vertices(inner[i,0], inner[i,1], 1.0, inner[i,2]) for i in range(12)])
    ox = np.mean(all_pts[:, 0])
    oy = np.mean(all_pts[:, 1])
    
    init_params = np.concatenate((inner.flatten(), [ox, oy, oang_init]))
    
    try:
        res = minimize(
            _objective,
            init_params,
            args=(t0, deadline),
            method='Nelder-Mead',
            options={
                'maxiter': 300,
                'xatol': 1e-8,
                'fatol': 1e-8,
                'adaptive': True
            }
        )
        
        if res.fun < 1e5:
            inner_opt = res.x[:36].reshape(12, 3)
            if _check_disjoint(inner_opt):
                ox_opt = res.x[36]
                oy_opt = res.x[37]
                oang_opt = res.x[38]
                all_pts_opt = np.vstack([_hex_vertices(inner_opt[i,0], inner_opt[i,1], 1.0, inner_opt[i,2]) for i in range(12)])
                all_pts_opt[:, 0] -= ox_opt
                all_pts_opt[:, 1] -= oy_opt
                L_opt = _outer_hex_side_fast(all_pts_opt, oang_opt)
                return (L_opt, inner_opt, (ox_opt, oy_opt), oang_opt)
    except Exception:
        pass
    
    return None

def _simulated_annealing(inner, t0, deadline, t_start=5.0, t_end=0.01, max_iter=3000):
    """Simulated annealing for packing optimization."""
    if time.time() - t0 > deadline:
        return inner
    
    n = len(inner)
    best_inner = inner.copy()
    best_L = float('inf')
    
    for oang in [0.0, 30.0]:
        L, valid = _evaluate_config(inner, oang)
        if valid and L < best_L:
            best_L = L
            best_inner = inner.copy()
    
    current_L = best_L
    current_inner = best_inner.copy()
    current_oang = 0.0 if best_L < float('inf') else 0.0
    
    # Recompute current L properly
    L, valid = _evaluate_config(current_inner, current_oang)
    if not valid:
        return best_inner
    current_L = L
    
    for it in range(max_iter):
        if time.time() - t0 > deadline:
            break
        
        t = t_start * (t_end / t_start) ** (it / max_iter)
        
        # Generate perturbation
        perturbation = np.random.randn(n, 3) * 0.1
        perturbation[:, 2] *= 0.5  # Smaller rotation perturbations
        
        new_inner = current_inner + perturbation
        new_oang = current_oang + np.random.randn() * 2.0
        
        new_L, new_valid = _evaluate_config(new_inner, new_oang)
        
        if not new_valid:
            continue
        
        delta_L = new_L - current_L
        
        if delta_L < 0 or np.random.random() < np.exp(-delta_L / t):
            current_L = new_L
            current_inner = new_inner
            current_oang = new_oang
            
            if new_L < best_L:
                best_L = new_L
                best_inner = new_inner.copy()
    
    return best_inner

def construct():
    t0 = time.time()
    
    def time_left():
        return DEADLINE - (time.time() - t0)
    
    # Multiple initial configurations
    configs = [
        _make_hex_grid(4, 3, gap=0.1),
        _make_hex_grid(3, 4, gap=0.1),
        _make_hex_grid(2, 6, gap=0.1),
        _make_hex_grid(6, 2, gap=0.1),
        _make_ring(gap=0.1),
        _make_packed_cluster(gap=0.1),
        _make_strip(gap=0.1),
        _make_hex_grid(4, 3, rot_offset=30, gap=0.1),
        _make_hex_grid(3, 4, rot_offset=30, gap=0.1),
        _make_hex_grid(2, 6, rot_offset=30, gap=0.1),
    ]
    
    # Add random configurations
    np.random.seed(42)
    for _ in range(10):
        pts = np.random.randn(12, 3) * 3.0
        rot_pattern = np.random.choice(['uniform', 'alternating', 'radial'])
        if rot_pattern == 'uniform':
            pts[:, 2] = np.random.uniform(0, 60, 12)
        elif rot_pattern == 'alternating':
            pts[:, 2] = np.where(np.arange(12) % 2 == 0, 0.0, 30.0) + np.random.uniform(-5, 5, 12)
        else:
            cx = np.mean(pts[:, 0])
            cy = np.mean(pts[:, 1])
            angles = np.arctan2(pts[:, 1] - cy, pts[:, 0] - cx)
            pts[:, 2] = np.degrees(angles) % 60
        pts[:, 0] -= np.mean(pts[:, 0])
        pts[:, 1] -= np.mean(pts[:, 1])
        configs.append(pts)
    
    best_L = float('inf')
    best_inner = None
    best_center = None
    best_angle = None
    
    # Phase 1: Evaluate initial configs and run SA on promising ones
    for idx, cfg in enumerate(configs):
        if time_left() < 30:
            break
        
        # Quick evaluation
        for oang in [0.0, 30.0]:
            L, valid = _evaluate_config(cfg, oang)
            if valid and L < best_L:
                best_L = L
                best_inner = cfg.copy()
                all_pts = np.vstack([_hex_vertices(cfg[i,0], cfg[i,1], 1.0, cfg[i,2]) for i in range(12)])
                best_center = (np.mean(all_pts[:, 0]), np.mean(all_pts[:, 1]))
                best_angle = oang
        
        # Run SA on this config if it looks promising
        if time_left() > 15:
            sa_result = _simulated_annealing(cfg.copy(), t0, time_left() - 5)
            for oang in [0.0, 30.0]:
                L, valid = _evaluate_config(sa_result, oang)
                if valid and L < best_L:
                    best_L = L
                    best_inner = sa_result.copy()
                    all_pts = np.vstack([_hex_vertices(sa_result[i,0], sa_result[i,1], 1.0, sa_result[i,2]) for i in range(12)])
                    best_center = (np.mean(all_pts[:, 0]), np.mean(all_pts[:, 1]))
                    best_angle = oang
    
    # Phase 2: Local optimization from best config
    if best_inner is not None and time_left() > 40:
        for oang in [0.0, 30.0]:
            if time_left() < 10:
                break
            
            result = _polish(best_inner.copy(), t0, time_left() - 5, oang)
            if result is not None:
                L_opt, inner_opt, center_opt, angle_opt = result
                if L_opt < best_L:
                    best_L = L_opt
                    best_inner = inner_opt
                    best_center = center_opt
                    best_angle = angle_opt
    
    # Phase 3: More SA runs with different seeds
    if time_left() > 25:
        for seed in range(5):
            if time_left() < 10:
                break
            
            np.random.seed(1000 + seed)
            pts = np.random.randn(12, 3) * 2.5
            pts[:, 2] = np.random.uniform(0, 60, 12)
            pts[:, 0] -= np.mean(pts[:, 0])
            pts[:, 1] -= np.mean(pts[:, 1])
            
            sa_result = _simulated_annealing(pts.copy(), t0, time_left() - 5)
            for oang in [0.0, 30.0]:
                L, valid = _evaluate_config(sa_result, oang)
                if valid and L < best_L:
                    best_L = L
                    best_inner = sa_result.copy()
                    all_pts = np.vstack([_hex_vertices(sa_result[i,0], sa_result[i,1], 1.0, sa_result[i,2]) for i in range(12)])
                    best_center = (np.mean(all_pts[:, 0]), np.mean(all_pts[:, 1]))
                    best_angle = oang
    
    # Final polish
    if best_inner is not None and time_left() > 15:
        result = _polish(best_inner.copy(), t0, time_left() - 3, best_angle)
        if result is not None:
            L_opt, inner_opt, center_opt, angle_opt = result
            if L_opt < best_L:
                best_L = L_opt
                best_inner = inner_opt
                best_center = center_opt
                best_angle = angle_opt
    
    if best_inner is None:
        cfg = _make_hex_grid(4, 3, gap=0.5)
        inner = [[float(p[0]), float(p[1]), float(p[2])] for p in cfg]
        outer_center = (0.0, 0.0)
        outer_angle = 0.0
        all_pts = np.vstack([_hex_vertices(p[0], p[1], 1.0, p[2]) for p in inner])
        outer_side = _outer_hex_side_fast(all_pts, outer_angle)
        return inner, outer_center, outer_side, outer_angle
    
    inner = []
    for i in range(12):
        x = best_inner[i, 0]
        y = best_inner[i, 1]
        r = best_inner[i, 2]
        inner.append([float(x), float(y), float(r)])
    
    outer_center = (float(best_center[0]), float(best_center[1]))
    outer_angle = float(best_angle)
    
    if not _check_disjoint(inner):
        cfg = _make_hex_grid(4, 3, gap=0.5)
        inner = [[float(p[0]), float(p[1]), float(p[2])] for p in cfg]
        outer_center = (0.0, 0.0)
        outer_angle = 0.0
        all_pts = np.vstack([_hex_vertices(p[0], p[1], 1.0, p[2]) for p in inner])
        outer_side = _outer_hex_side_fast(all_pts, outer_angle)
        return inner, outer_center, outer_side, outer_angle
    
    all_pts = np.vstack([_hex_vertices(p[0], p[1], 1.0, p[2]) for p in inner])
    all_pts[:, 0] -= outer_center[0]
    all_pts[:, 1] -= outer_center[1]
    outer_side = _outer_hex_side_fast(all_pts, outer_angle)
    outer_side *= 1.0001
    
    return inner, outer_center, outer_side, outer_angle
