import numpy as np
import time
import math
import random
from scipy.optimize import minimize, minimize_scalar

N = 12
DEADLINE = 145.0  # Safety margin below 150s kill

# Precompute base hexagon vertices (side 1, center 0,0, rot 0)
_BASE_HEX_VERTS = np.column_stack((
    np.cos(np.radians([0, 60, 120, 180, 240, 300])),
    np.sin(np.radians([0, 60, 120, 180, 240, 300]))
))

def get_hex_vertices_fast(cx, cy, rot_deg):
    """Fast hexagon vertex computation."""
    rad = np.radians(rot_deg)
    c, s = np.cos(rad), np.sin(rad)
    R = np.array([[c, -s], [s, c]])
    return _BASE_HEX_VERTS @ R.T + np.array([cx, cy])

def sat_overlap(v1, v2):
    """
    Returns separation distance (positive) if not overlapping, or negative penetration if overlapping.
    """
    axes = []
    for v in (v1, v2):
        for i in range(len(v)):
            p1 = v[i]
            p2 = v[(i+1) % len(v)]
            e = p2 - p1
            n = np.array([-e[1], e[0]])
            norm = np.linalg.norm(n)
            if norm > 1e-12:
                n = n / norm
                axes.append(n)
    
    min_overlap = float('inf')
    for n in axes:
        proj1 = v1 @ n
        min1, max1 = np.min(proj1), np.max(proj1)
        proj2 = v2 @ n
        min2, max2 = np.min(proj2), np.max(proj2)
        
        if max1 < min2:
            gap = min2 - max1
            return gap
        if max2 < min1:
            gap = min1 - max2
            return gap
        
        overlap = min(max1, max2) - max(min1, min2)
        min_overlap = min(min_overlap, overlap)
    
    return -min_overlap

def outer_hex_side(points, outer_rot_deg):
    """
    Minimum side length of regular hexagon centered at origin with given rotation
    that encloses all points.
    """
    angles = np.radians(np.array([30, 90, 150, 210, 270, 330]) + outer_rot_deg)
    normals = np.column_stack((np.cos(angles), np.sin(angles)))
    proj = points @ normals.T
    max_proj = np.max(np.abs(proj))
    return (2.0 / np.sqrt(3.0)) * max_proj

def optimize_outer_rotation(points, init_rot=0.0):
    """
    Find the outer rotation that minimizes the side length for a fixed set of points.
    """
    def obj(theta):
        return outer_hex_side(points, theta)
    
    # The function is periodic with period 60 degrees.
    # We can search in [init_rot - 30, init_rot + 30] or just [0, 60].
    # Let's do a coarse grid first to find a good basin.
    best_L = float('inf')
    best_rot = init_rot
    # Coarse search
    for rot in np.linspace(0, 60, 360, endpoint=False):
        L = outer_hex_side(points, rot)
        if L < best_L:
            best_L = L
            best_rot = rot
            
    # Fine search around best_rot
    res = minimize_scalar(obj, bounds=(best_rot - 5, best_rot + 5), method='bounded', options={'xatol': 1e-6})
    if res.success:
        return res.fun, res.x
        
    return best_L, best_rot

def smooth_overlap_penalty(xys_rots, beta=30.0):
    """
    Compute smooth overlap penalty using log-sum-exp.
    """
    centers = xys_rots[:, :2]
    vertices_list = []
    for i in range(N):
        cx, cy, rot = xys_rots[i]
        v = get_hex_vertices_fast(cx, cy, rot)
        vertices_list.append(v)
    
    penetrations = []
    for i in range(N):
        for j in range(i+1, N):
            d = np.linalg.norm(centers[i] - centers[j])
            if d < 2.2:
                d_sat = sat_overlap(vertices_list[i], vertices_list[j])
                if d_sat < 0:
                    penetrations.append(-d_sat)
    
    if len(penetrations) == 0:
        return 0.0
    
    p_arr = np.array(penetrations)
    # Log-sum-exp stability
    max_p = np.max(p_arr)
    if max_p < 1e-12:
        return 0.0
    shifted = beta * (p_arr - max_p)
    shifted = np.clip(shifted, -50, 50)
    lse = max_p + (1.0/beta) * np.log(np.sum(np.exp(shifted)))
    return lse

def construct():
    start_time = time.time()
    random.seed(42)
    np.random.seed(42)
    
    def time_left():
        return DEADLINE - (time.time() - start_time)
    
    def objective(params):
        if time.time() - start_time > DEADLINE - 5:
            return 1e6
        
        xys_rots = params[:3*N].reshape(N, 3)
        outer_rot = params[3*N]
        
        penalty = smooth_overlap_penalty(xys_rots, beta=30.0)
        
        all_points = np.vstack([get_hex_vertices_fast(x, y, r) for x, y, r in xys_rots])
        L = outer_hex_side(all_points, outer_rot)
        
        return L + 100.0 * penalty
    
    def make_hex_grid(rows, cols, rot_offset=0.0, gap=0.0):
        pts = []
        sqrt3 = np.sqrt(3)
        spacing_x = sqrt3 + gap
        spacing_y = 1.5 + gap * 0.5
        for r in range(rows):
            y = r * spacing_y
            for c in range(cols):
                x = c * spacing_x
                if r % 2 == 1:
                    x += spacing_x / 2.0
                pts.append([x, y, rot_offset])
        pts = np.array(pts)
        if len(pts) > N:
            pts = pts[:N]
        pts[:, :2] -= np.mean(pts[:, :2], axis=0)
        return pts
    
    def make_ring(gap=0.0):
        pts = []
        sqrt3 = np.sqrt(3)
        d1 = sqrt3 + gap
        d2 = 2 * sqrt3 + gap
        pts.append([0, 0, 0.0])
        for k in range(6):
            ang = np.radians(60 * k)
            pts.append([d1 * np.cos(ang), d1 * np.sin(ang), 0.0])
        for k in range(5):
            ang = np.radians(60 * k + 30)
            pts.append([d2 * np.cos(ang), d2 * np.sin(ang), 0.0])
        pts = np.array(pts)
        pts[:, :2] -= np.mean(pts[:, :2], axis=0)
        return pts
    
    def make_ring12(gap=0.0):
        pts = []
        R = 2.5 + gap
        for i in range(N):
            theta = 2 * np.pi * i / N
            x = R * np.cos(theta)
            y = R * np.sin(theta)
            angle_deg = math.degrees(theta)
            pts.append([x, y, angle_deg])
        pts = np.array(pts)
        pts[:, :2] -= np.mean(pts[:, :2], axis=0)
        return pts

    def make_diamond(gap=0.0):
        pts = []
        sqrt3 = np.sqrt(3)
        dx = sqrt3 + gap
        dy = 1.5 + gap * 0.5
        # Pattern for 12 hexagons in a diamond shape
        # Row 0: 2
        # Row 1: 3
        # Row 2: 4
        # Row 3: 3
        # Total: 12
        for c in range(2):
            pts.append([c * dx + dx/2, 0, 0.0])
        for c in range(3):
            pts.append([c * dx, dy, 0.0])
        for c in range(4):
            pts.append([c * dx + dx/2, 2*dy, 0.0])
        for c in range(3):
            pts.append([c * dx, 3*dy, 0.0])
        pts = np.array(pts)
        pts[:, :2] -= np.mean(pts[:, :2], axis=0)
        return pts

    def is_valid(params, tol=1e-6):
        xys_rots = params[:3*N].reshape(N, 3)
        vertices_list = [get_hex_vertices_fast(x, y, r) for x, y, r in xys_rots]
        for i in range(N):
            for j in range(i+1, N):
                d = sat_overlap(vertices_list[i], vertices_list[j])
                if d < -tol:
                    return False
        return True

    def get_L(params):
        xys_rots = params[:3*N].reshape(N, 3)
        outer_rot = params[3*N]
        all_points = np.vstack([get_hex_vertices_fast(x, y, r) for x, y, r in xys_rots])
        return outer_hex_side(all_points, outer_rot)
    
    def full_score(cfg, angle):
        params = np.concatenate((cfg.flatten(), [angle]))
        if not is_valid(params):
            return None
        return get_L(params)

    best_L = float('inf')
    best_params = None
    
    # Generate diverse initial configurations
    configs = []
    gap = 0.05
    configs.append(make_hex_grid(4, 3, gap=gap))
    configs.append(make_hex_grid(3, 4, gap=gap))
    configs.append(make_ring(gap=gap))
    configs.append(make_ring12(gap=gap))
    configs.append(make_diamond(gap=gap))
    
    # Add rotated variants
    for base_cfg in configs[:3]:
        rotated = base_cfg.copy()
        rotated[:, 2] = 30.0
        configs.append(rotated)

    # Phase 1: Evaluate initial configurations with optimized outer rotation
    for cfg in configs:
        if time.time() - start_time > DEADLINE - 40:
            break
        
        # Find best outer rotation for this configuration
        all_points = np.vstack([get_hex_vertices_fast(x, y, r) for x, y, r in cfg])
        L_val, best_rot = optimize_outer_rotation(all_points)
        
        init_params = np.concatenate((cfg.flatten(), [best_rot]))
        
        if is_valid(init_params):
            # Re-calculate L with the specific params to be sure
            L_check = get_L(init_params)
            if L_check < best_L:
                best_L = L_check
                best_params = init_params.copy()

    # Phase 2: Simulated Annealing
    if time.time() - start_time < DEADLINE - 50 and best_params is not None:
        cfg = best_params[:3*N].reshape(N, 3).copy()
        angle = best_params[3*N]
        cur_score = best_L
        
        T = 0.8
        T_end = 0.003
        alpha = 0.9997
        
        while time.time() - start_time < DEADLINE - 50:
            r = random.random()
            
            if r < 0.50:
                idx = random.randrange(N)
                old_x, old_y = cfg[idx][0], cfg[idx][1]
                step = 0.2 * (1.0 + 0.5 * T)
                cfg[idx][0] += random.gauss(0, step)
                cfg[idx][1] += random.gauss(0, step)
            elif r < 0.70:
                idx = random.randrange(N)
                old_ang = cfg[idx][2]
                cfg[idx][2] += random.gauss(0, 5.0 + 3.0 * T)
            elif r < 0.80:
                old_angle = angle
                angle += random.gauss(0, 2.0 + 2.0 * T)
            else:
                idx1, idx2 = random.sample(range(N), 2)
                old_x1, old_y1 = cfg[idx1][0], cfg[idx1][1]
                old_x2, old_y2 = cfg[idx2][0], cfg[idx2][1]
                step = 0.15 * (1.0 + 0.5 * T)
                cfg[idx1][0] += random.gauss(0, step)
                cfg[idx1][1] += random.gauss(0, step)
                cfg[idx2][0] += random.gauss(0, step)
                cfg[idx2][1] += random.gauss(0, step)
            
            new_score = full_score(cfg, angle)
            
            if new_score is not None:
                if new_score < cur_score or (T > T_end and random.random() < math.exp(-(new_score - cur_score) / T)):
                    cur_score = new_score
                    if new_score < best_L:
                        best_L = new_score
                        best_params = np.concatenate((cfg.flatten(), [angle])).copy()
                else:
                    if r < 0.50:
                        cfg[idx][0], cfg[idx][1] = old_x, old_y
                    elif r < 0.70:
                        cfg[idx][2] = old_ang
                    elif r < 0.80:
                        angle = old_angle
                    else:
                        cfg[idx1][0], cfg[idx1][1] = old_x1, old_y1
                        cfg[idx2][0], cfg[idx2][1] = old_x2, old_y2
            else:
                if r < 0.50:
                    cfg[idx][0], cfg[idx][1] = old_x, old_y
                elif r < 0.70:
                    cfg[idx][2] = old_ang
                elif r < 0.80:
                    angle = old_angle
                else:
                    cfg[idx1][0], cfg[idx1][1] = old_x1, old_y1
                    cfg[idx2][0], cfg[idx2][1] = old_x2, old_y2
            
            T *= alpha
            if T < T_end:
                break

    # Phase 3: Gradient-based refinement (L-BFGS-B)
    if best_params is not None and time.time() - start_time < DEADLINE - 25:
        try:
            res_lbgfs = minimize(
                objective,
                best_params,
                method='L-BFGS-B',
                options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-10, 'maxfun': 1500}
            )
            if is_valid(res_lbgfs.x):
                L_val = get_L(res_lbgfs.x)
                if L_val < best_L:
                    best_L = L_val
                    best_params = res_lbgfs.x.copy()
        except Exception:
            pass

    # Phase 4: Optimize outer rotation specifically for the best inner config
    if best_params is not None:
        cfg = best_params[:3*N].reshape(N, 3)
        all_points = np.vstack([get_hex_vertices_fast(x, y, r) for x, y, r in cfg])
        L_val, best_rot = optimize_outer_rotation(all_points)
        
        new_params = np.concatenate((cfg.flatten(), [best_rot]))
        if is_valid(new_params):
            L_check = get_L(new_params)
            if L_check < best_L:
                best_L = L_check
                best_params = new_params

    # Phase 5: Final high-precision polish (Nelder-Mead)
    if best_params is not None and time.time() - start_time < DEADLINE - 10:
        try:
            res_nm = minimize(
                objective,
                best_params,
                method='Nelder-Mead',
                options={'maxiter': 2000, 'xatol': 1e-8, 'fatol': 1e-8, 'adaptive': True}
            )
            if is_valid(res_nm.x):
                L_val = get_L(res_nm.x)
                if L_val < best_L:
                    best_L = L_val
                    best_params = res_nm.x.copy()
        except Exception:
            pass

    # Final outer rotation optimization
    if best_params is not None:
        cfg = best_params[:3*N].reshape(N, 3)
        all_points = np.vstack([get_hex_vertices_fast(x, y, r) for x, y, r in cfg])
        L_val, best_rot = optimize_outer_rotation(all_points)
        new_params = np.concatenate((cfg.flatten(), [best_rot]))
        if is_valid(new_params):
            L_check = get_L(new_params)
            if L_check < best_L:
                best_L = L_check
                best_params = new_params

    if best_params is None:
        inner = [[4.0 * k, 0.0, 0.0] for k in range(-(N // 2), N - N // 2)]
        return inner, (0.0, 0.0), 40.0, 0.0

    xys_rots = best_params[:3*N].reshape(N, 3)
    outer_rot = best_params[3*N]
    
    inner = []
    for x, y, rot in xys_rots:
        inner.append([float(x), float(y), float(rot)])
    
    return inner, (0.0, 0.0), float(best_L), float(outer_rot)
