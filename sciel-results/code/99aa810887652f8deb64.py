import numpy as np
import time
import math
import random
from scipy.optimize import minimize

N = 12
DEADLINE = 140.0

def get_hex_vertices(cx, cy, side=1.0, rot_deg=0.0):
    angles = np.radians(np.array([0, 60, 120, 180, 240, 300])) + np.radians(rot_deg)
    x = cx + side * np.cos(angles)
    y = cy + side * np.sin(angles)
    return np.column_stack((x, y))

def outer_hex_side(points, outer_rot_deg):
    angles = np.radians(np.array([30, 90, 150, 210, 270, 330]) + outer_rot_deg)
    normals = np.column_stack((np.cos(angles), np.sin(angles)))
    proj = points @ normals.T
    max_proj = np.max(np.abs(proj))
    return (2.0 / np.sqrt(3.0)) * max_proj

def optimize_outer_rotation(points, init_rot=0.0, n_samples=720):
    best_L = float('inf')
    best_rot = init_rot
    
    rots = np.linspace(0, 60, n_samples, endpoint=False)
    for rot in rots:
        L = outer_hex_side(points, rot)
        if L < best_L:
            best_L = L
            best_rot = rot
    
    local_rots = np.linspace(best_rot - 2.0, best_rot + 2.0, n_samples)
    for rot in local_rots:
        L = outer_hex_side(points, rot)
        if L < best_L:
            best_L = L
            best_rot = rot
    
    return best_L, best_rot

def sat_overlap(v1, v2):
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
        
        if max1 < min2 or max2 < min1:
            gap = min2 - max1 if min2 > max1 else min1 - max2
            return gap
        
        overlap = min(max1, max2) - max(min1, min2)
        min_overlap = min(min_overlap, overlap)
    
    return -min_overlap

def compute_L_and_penalties(params):
    xys_rots = params[:3*N].reshape(N, 3)
    outer_rot = params[3*N]
    
    vertices_list = [get_hex_vertices(x, y, 1.0, r) for x, y, r in xys_rots]
    
    penalty = 0.0
    for i in range(N):
        for j in range(i+1, N):
            d = sat_overlap(vertices_list[i], vertices_list[j])
            if d < 0:
                penalty += d * d
    
    all_points = np.vstack(vertices_list)
    angles = np.radians(np.array([30, 90, 150, 210, 270, 330]) + outer_rot)
    normals = np.column_stack((np.cos(angles), np.sin(angles)))
    proj = all_points @ normals.T
    max_proj = np.max(np.abs(proj))
    L = (2.0 / np.sqrt(3.0)) * max_proj
    
    return L, penalty

def smooth_max(values, beta=50.0):
    if len(values) == 0:
        return 0.0
    m = np.max(values)
    if m < -50.0 / beta:
        return m
    return m + np.log(np.sum(np.exp(beta * (values - m)))) / beta

def smooth_overlap_penalty(params, beta=50.0):
    xys_rots = params[:3*N].reshape(N, 3)
    outer_rot = params[3*N]
    
    vertices_list = [get_hex_vertices(x, y, 1.0, r) for x, y, r in xys_rots]
    
    penalty = 0.0
    for i in range(N):
        for j in range(i+1, N):
            d = sat_overlap(vertices_list[i], vertices_list[j])
            if d < 0:
                penalty += d * d
    
    all_points = np.vstack(vertices_list)
    angles = np.radians(np.array([30, 90, 150, 210, 270, 330]) + outer_rot)
    normals = np.column_stack((np.cos(angles), np.sin(angles)))
    proj = all_points @ normals.T
    abs_proj = np.abs(proj)
    max_proj_smooth = smooth_max(abs_proj.flatten(), beta=beta)
    L_smooth = (2.0 / np.sqrt(3.0)) * max_proj_smooth
    
    return L_smooth + 100.0 * penalty

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

def is_valid(params, tol=1e-6):
    xys_rots = params[:3*N].reshape(N, 3)
    vertices_list = [get_hex_vertices(x, y, 1.0, r) for x, y, r in xys_rots]
    
    for i in range(N):
        for j in range(i+1, N):
            d = sat_overlap(vertices_list[i], vertices_list[j])
            if d < -tol:
                return False
    return True

def simulated_annealing(params, start_temp=1.0, end_temp=0.01, n_iter=2000, time_limit=30.0, start_time=None):
    if start_time is None:
        start_time = time.time()
    
    current_params = params.copy()
    current_L, current_penalty = compute_L_and_penalties(current_params)
    current_obj = current_L + 100.0 * current_penalty
    
    best_params = current_params.copy()
    best_L = current_L
    best_obj = current_obj
    
    temp = start_temp
    cooling_rate = (end_temp / start_temp) ** (1.0 / max(n_iter, 1))
    
    for it in range(n_iter):
        if time.time() - start_time > time_limit:
            break
        
        new_params = current_params.copy()
        n_perturb = random.choice([1, 1, 2])
        for _ in range(n_perturb):
            idx = random.randint(0, N-1)
            new_params[idx*3] += random.gauss(0, 0.3)
            new_params[idx*3+1] += random.gauss(0, 0.3)
            if random.random() < 0.3:
                new_params[idx*3+2] += random.gauss(0, 10.0)
        if random.random() < 0.1:
            new_params[3*N] += random.gauss(0, 5.0)
        
        new_L, new_penalty = compute_L_and_penalties(new_params)
        new_obj = new_L + 100.0 * new_penalty
        
        if new_obj < current_obj or random.random() < math.exp(-(new_obj - current_obj) / max(temp, 1e-10)):
            current_params = new_params
            current_L = new_L
            current_penalty = new_penalty
            current_obj = new_obj
            
            if current_obj < best_obj:
                best_params = current_params.copy()
                best_L = current_L
                best_obj = current_obj
        
        temp *= cooling_rate
    
    return best_params, best_L

def construct():
    start_time = time.time()
    random.seed(42)
    np.random.seed(42)
    
    def time_left():
        return DEADLINE - (time.time() - start_time)
    
    best_L = float('inf')
    best_params = None
    
    configs = []
    gap = 0.05
    configs.append(make_hex_grid(4, 3, gap=gap))
    configs.append(make_hex_grid(3, 4, gap=gap))
    configs.append(make_hex_grid(2, 6, gap=gap))
    configs.append(make_hex_grid(6, 2, gap=gap))
    configs.append(make_ring(gap=gap))
    
    for base_cfg in configs[:3]:
        rotated = base_cfg.copy()
        rotated[:, 2] = 30.0
        configs.append(rotated)
    
    outer_rots = [0.0, 30.0, 15.0]
    
    # Phase 1: Evaluate initial configs and run simulated annealing
    for cfg in configs:
        if time_left() < 20:
            break
        
        for orot in outer_rots:
            if time_left() < 20:
                break
            
            init_params = np.concatenate((cfg.flatten(), [orot]))
            
            if is_valid(init_params):
                xys_rots = init_params[:3*N].reshape(N, 3)
                all_points = np.vstack([get_hex_vertices(x, y, 1.0, r) for x, y, r in xys_rots])
                L_val = outer_hex_side(all_points, orot)
                if L_val < best_L:
                    best_L = L_val
                    best_params = init_params.copy()
            
            if time_left() < 25:
                continue
            
            try:
                sa_params, sa_L = simulated_annealing(
                    init_params, 
                    start_temp=1.5, 
                    end_temp=0.05,
                    n_iter=1500,
                    time_limit=8.0,
                    start_time=start_time
                )
                
                if is_valid(sa_params):
                    xys_rots = sa_params[:3*N].reshape(N, 3)
                    all_points = np.vstack([get_hex_vertices(x, y, 1.0, r) for x, y, r in xys_rots])
                    L_val = outer_hex_side(all_points, sa_params[3*N])
                    if L_val < best_L:
                        best_L = L_val
                        best_params = sa_params.copy()
                        
            except Exception:
                pass
    
    # Phase 2: Nelder-Mead polish
    if best_params is not None and time_left() > 30:
        try:
            res_nm = minimize(
                lambda p: smooth_overlap_penalty(p, beta=30.0),
                best_params,
                method='Nelder-Mead',
                options={
                    'maxiter': 150,
                    'xatol': 1e-6,
                    'fatol': 1e-6,
                    'adaptive': True
                }
            )
            
            if is_valid(res_nm.x):
                xys_rots = res_nm.x[:3*N].reshape(N, 3)
                all_points = np.vstack([get_hex_vertices(x, y, 1.0, r) for x, y, r in xys_rots])
                L_val = outer_hex_side(all_points, res_nm.x[3*N])
                if L_val < best_L:
                    best_L = L_val
                    best_params = res_nm.x.copy()
                    
        except Exception:
            pass
    
    # Phase 3: Coordinate descent refinement
    if best_params is not None and time_left() > 20:
        try:
            current = best_params.copy()
            for iteration in range(50):
                if time_left() < 15:
                    break
                
                improved = False
                for i in range(N):
                    for axis in [0, 1, 2]:
                        for delta in [0.05, -0.05, 0.02, -0.02]:
                            trial = current.copy()
                            trial[i*3 + axis] += delta
                            if is_valid(trial):
                                L_trial, _ = compute_L_and_penalties(trial)
                                if L_trial < best_L - 1e-8:
                                    best_L = L_trial
                                    best_params = trial.copy()
                                    current = trial.copy()
                                    improved = True
                
                if not improved:
                    break
                    
        except Exception:
            pass
    
    # Phase 4: Optimize outer rotation
    if best_params is not None and time_left() > 15:
        try:
            xys_rots = best_params[:3*N].reshape(N, 3)
            all_points = np.vstack([get_hex_vertices(x, y, 1.0, r) for x, y, r in xys_rots])
            L_opt, rot_opt = optimize_outer_rotation(all_points, best_params[3*N], n_samples=1440)
            
            if L_opt < best_L:
                best_L = L_opt
                new_params = best_params.copy()
                new_params[3*N] = rot_opt
                best_params = new_params
        except Exception:
            pass
    
    # Phase 5: Final high-precision polish
    if best_params is not None and time_left() > 10:
        try:
            res_final = minimize(
                lambda p: smooth_overlap_penalty(p, beta=50.0),
                best_params,
                method='Nelder-Mead',
                options={
                    'maxiter': 200,
                    'xatol': 1e-8,
                    'fatol': 1e-8,
                    'adaptive': True
                }
            )
            
            if is_valid(res_final.x):
                xys_rots = res_final.x[:3*N].reshape(N, 3)
                all_points = np.vstack([get_hex_vertices(x, y, 1.0, r) for x, y, r in xys_rots])
                L_val = outer_hex_side(all_points, res_final.x[3*N])
                if L_val < best_L:
                    best_L = L_val
                    best_params = res_final.x.copy()
                    
        except Exception:
            pass
    
    # Phase 6: Fine outer rotation search
    if best_params is not None and time_left() > 5:
        try:
            xys_rots = best_params[:3*N].reshape(N, 3)
            all_points = np.vstack([get_hex_vertices(x, y, 1.0, r) for x, y, r in xys_rots])
            L_opt, rot_opt = optimize_outer_rotation(all_points, best_params[3*N], n_samples=2880)
            
            if L_opt < best_L:
                best_L = L_opt
                new_params = best_params.copy()
                new_params[3*N] = rot_opt
                best_params = new_params
        except Exception:
            pass
    
    # Phase 7: Micro-polish outer rotation
    if best_params is not None and time_left() > 2:
        try:
            xys_rots = best_params[:3*N].reshape(N, 3)
            all_points = np.vstack([get_hex_vertices(x, y, 1.0, r) for x, y, r in xys_rots])
            cur_rot = best_params[3*N]
            fine_rots = np.linspace(cur_rot - 0.1, cur_rot + 0.1, 200)
            best_local_L = best_L
            best_local_rot = cur_rot
            for rot in fine_rots:
                L = outer_hex_side(all_points, rot)
                if L < best_local_L:
                    best_local_L = L
                    best_local_rot = rot
            
            if best_local_L < best_L:
                best_L = best_local_L
                best_params[3*N] = best_local_rot
        except Exception:
            pass
    
    # Fallback
    if best_params is None:
        inner = []
        for k in range(N):
            inner.append([4.0 * k, 0.0, 0.0])
        pts = np.array([[x, y] for x, y, r in inner])
        L_val = outer_hex_side(pts, 0.0)
        return inner, (0.0, 0.0), L_val, 0.0
    
    xys_rots = best_params[:3*N].reshape(N, 3)
    outer_rot = best_params[3*N]
    
    inner = [[float(x), float(y), float(r)] for x, y, r in xys_rots]
    center = [0.0, 0.0]
    side = float(best_L)
    angle = float(outer_rot)
    
    return inner, center, side, angle
