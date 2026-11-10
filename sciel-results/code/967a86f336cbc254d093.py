# EVOLVE-BLOCK-START
import numpy as np
import time
import random
from scipy.optimize import minimize

def _hex_vertices(cx, cy, side, angle_deg):
    a = np.radians(angle_deg)
    angles = a + np.arange(6) * np.pi / 3.0
    return np.column_stack([
        cx + side * np.cos(angles),
        cy + side * np.sin(angles)
    ])

def _compute_all_vertices(centers, angles_deg, side=1.0):
    n = len(centers)
    verts = np.empty((n * 6, 2))
    for i in range(n):
        verts[i*6:(i+1)*6] = _hex_vertices(centers[i,0], centers[i,1], side, angles_deg[i])
    return verts

def _min_outer_side(vertices_all, outer_angle_deg=0.0):
    """Minimum side of outer hexagon (centered at origin) with given angle that contains all vertices."""
    k = np.arange(6)
    base_angle = np.radians(outer_angle_deg)
    angles = base_angle + (k + 0.5) * np.pi / 3
    m = np.column_stack([np.cos(angles), np.sin(angles)])
    dots = vertices_all @ m.T
    max_dot = np.max(dots, axis=1)
    global_max = np.max(max_dot)
    if global_max < 0:
        global_max = 0.0
    s_min = (2.0 / np.sqrt(3.0)) * global_max
    return s_min

def _min_outer_side_smooth(vertices_all, outer_angle_deg=0.0, beta=80.0):
    """Smooth approximation using log-sum-exp over all vertex-projection pairs."""
    k = np.arange(6)
    base_angle = np.radians(outer_angle_deg)
    angles = base_angle + (k + 0.5) * np.pi / 3
    m = np.column_stack([np.cos(angles), np.sin(angles)])
    dots = vertices_all @ m.T  # shape (72, 6)
    flat_dots = dots.ravel()  # shape (432,)
    
    max_val = np.max(flat_dots)
    if max_val < 0:
        max_val = 0.0
    
    shifted = flat_dots - max_val
    log_sum_exp = max_val + np.log(np.sum(np.exp(beta * shifted)) / len(flat_dots)) / beta
    
    s_min = (2.0 / np.sqrt(3.0)) * log_sum_exp
    return s_min

def _check_nonoverlap(centers, angles_deg, side=1.0, tol=1e-6):
    n = len(centers)
    vertices_list = []
    normals_list = []
    for i in range(n):
        verts = _hex_vertices(centers[i,0], centers[i,1], side, angles_deg[i])
        vertices_list.append(verts)
        normals = []
        for e in range(6):
            p1 = verts[e]
            p2 = verts[(e+1) % 6]
            edge = p2 - p1
            nx, ny = edge[1], -edge[0]
            norm = np.sqrt(nx*nx + ny*ny)
            if norm > 1e-12:
                normals.append(np.array([nx/norm, ny/norm]))
        normals_list.append(normals)
    
    for i in range(n):
        for j in range(i+1, n):
            v1 = vertices_list[i]
            v2 = vertices_list[j]
            overlap = True
            for hex_idx in range(2):
                verts = [v1, v2][hex_idx]
                normals = normals_list[hex_idx]
                for normal in normals:
                    proj1 = v1 @ normal
                    proj2 = v2 @ normal
                    min1, max1 = np.min(proj1), np.max(proj1)
                    min2, max2 = np.min(proj2), np.max(proj2)
                    if max1 + tol < min2 or max2 + tol < min1:
                        overlap = False
                        break
                if not overlap:
                    break
            if overlap:
                return False
    return True

def _initial_honeycomb():
    d = np.sqrt(3.0)
    positions = []
    angles = []
    
    positions.append([0.0, 0.0])
    angles.append(0.0)
    
    for k in range(6):
        angle = k * np.pi / 3
        x = d * np.cos(angle)
        y = d * np.sin(angle)
        positions.append([x, y])
        angles.append(0.0)
    
    d2 = 2.0 * d
    for k in range(5):
        angle = (k * 72 + 30) * np.pi / 180
        x = d2 * np.cos(angle)
        y = d2 * np.sin(angle)
        positions.append([x, y])
        angles.append(0.0)
    
    positions = np.array(positions)
    angles = np.array(angles)
    return positions, angles

def _objective_smooth(params, beta=80.0):
    """params: (37,) = [x1,y1,theta1, ..., x12,y12,theta12, outer_angle_deg]"""
    params_2d = params[:36].reshape(12, 3)
    centers = params_2d[:, :2]
    angles_deg = params_2d[:, 2]
    outer_angle = params[36]
    
    all_verts = _compute_all_vertices(centers, angles_deg)
    s = _min_outer_side_smooth(all_verts, outer_angle, beta)
    
    if not _check_nonoverlap(centers, angles_deg):
        return s + 50.0
    
    return s

def _objective_exact(params):
    """params: (37,) = [x1,y1,theta1, ..., x12,y12,theta12, outer_angle_deg]"""
    params_2d = params[:36].reshape(12, 3)
    centers = params_2d[:, :2]
    angles_deg = params_2d[:, 2]
    outer_angle = params[36]
    
    all_verts = _compute_all_vertices(centers, angles_deg)
    s = _min_outer_side(all_verts, outer_angle)
    
    if not _check_nonoverlap(centers, angles_deg):
        return s + 50.0
    
    return s

def hexagon_packing_12():
    start_time = time.time()
    deadline = start_time + 600.0
    
    centers, angles_deg = _initial_honeycomb()
    # params: 36 values for inner + 1 for outer angle = 37
    params = np.column_stack([centers, angles_deg]).flatten()
    params = np.concatenate([params, [0.0]])  # outer angle = 0 initially
    
    all_verts = _compute_all_vertices(centers, angles_deg)
    best_s = _min_outer_side(all_verts, 0.0)
    best_params = params.copy()
    
    if not _check_nonoverlap(centers, angles_deg):
        scale = 1.1
        while not _check_nonoverlap(centers * scale, angles_deg) and scale < 3.0:
            scale += 0.05
        centers = centers * scale
        params = np.column_stack([centers, angles_deg]).flatten()
        params = np.concatenate([params, [0.0]])
        all_verts = _compute_all_vertices(centers, angles_deg)
        best_s = _min_outer_side(all_verts, 0.0)
        best_params = params.copy()
    
    np.random.seed(42)
    random.seed(42)
    
    # Phase 1: Directed Simulated Annealing
    T = 0.5
    cooling = 0.99985
    step_size = 0.3
    angle_step = 5.0
    outer_angle_step = 3.0
    accept_count = 0
    total_count = 0
    
    current_params = best_params.copy()
    current_s = best_s
    
    max_iters = 500000
    for it in range(max_iters):
        if time.time() > deadline - 200.0:
            break
        
        # Pick random hexagon to move
        i = random.randint(0, 11)
        
        old_cx = current_params[i*3]
        old_cy = current_params[i*3+1]
        old_ang = current_params[i*3+2]
        
        new_cx = old_cx + np.random.randn() * step_size
        new_cy = old_cy + np.random.randn() * step_size
        new_ang = old_ang + np.random.randn() * angle_step
        
        current_params[i*3] = new_cx
        current_params[i*3+1] = new_cy
        current_params[i*3+2] = new_ang
        
        # Occasionally perturb outer angle
        if random.random() < 0.1:
            current_params[36] += np.random.randn() * outer_angle_step
        
        centers_new = current_params[:36].reshape(12, 3)[:, :2]
        angles_new = current_params[:36].reshape(12, 3)[:, 2]
        outer_angle_new = current_params[36]
        
        all_verts_new = _compute_all_vertices(centers_new, angles_new)
        s_new = _min_outer_side(all_verts_new, outer_angle_new)
        
        if not _check_nonoverlap(centers_new, angles_new):
            s_new += 50.0
        
        delta = s_new - current_s
        total_count += 1
        
        if delta < 0 or np.random.random() < np.exp(-delta / max(T, 1e-10)):
            current_s = s_new
            accept_count += 1
            if s_new < best_s:
                best_s = s_new
                best_params = current_params.copy()
        else:
            current_params[i*3] = old_cx
            current_params[i*3+1] = old_cy
            current_params[i*3+2] = old_ang
        
        T *= cooling
        
        if it % 500 == 0 and it > 0:
            acc_rate = accept_count / max(total_count, 1)
            if acc_rate < 0.15:
                step_size *= 0.85
                angle_step *= 0.85
                outer_angle_step *= 0.85
            elif acc_rate > 0.75:
                step_size *= 1.15
                angle_step *= 1.15
                outer_angle_step *= 1.15
            step_size = np.clip(step_size, 0.01, 1.5)
            angle_step = np.clip(angle_step, 0.5, 15.0)
            outer_angle_step = np.clip(outer_angle_step, 0.5, 10.0)
            accept_count = 0
            total_count = 0
    
    # Phase 2: Scipy optimization with smooth objective
    n_starts = 15
    for si in range(n_starts):
        if time.time() > deadline - 80.0:
            break
        
        if si == 0:
            x0 = best_params.copy()
        elif si == 1:
            x0 = best_params + np.random.randn(37) * 0.05
        elif si == 2:
            x0 = best_params + np.random.randn(37) * 0.1
        elif si == 3:
            x0 = best_params + np.random.randn(37) * 0.15
        elif si == 4:
            x0 = best_params + np.random.randn(37) * 0.2
        else:
            x0 = best_params + np.random.randn(37) * (0.08 + 0.03 * si)
        
        try:
            result = minimize(
                _objective_smooth,
                x0,
                method='L-BFGS-B',
                options={'maxiter': 1500, 'maxls': 150, 'ftol': 1e-14, 'gtol': 1e-12},
                tol=1e-14
            )
            
            c = result.x[:36].reshape(12, 3)[:, :2]
            a = result.x[:36].reshape(12, 3)[:, 2]
            if _check_nonoverlap(c, a):
                s_exact = _min_outer_side(_compute_all_vertices(c, a), result.x[36])
                if s_exact < best_s:
                    best_s = s_exact
                    best_params = result.x.copy()
        except Exception:
            pass
    
    # Phase 3: Fine polish with small random steps
    for _ in range(50000):
        if time.time() > deadline - 30.0:
            break
        trial = best_params + np.random.randn(37) * 0.015
        c = trial[:36].reshape(12,3)[:, :2]
        a = trial[:36].reshape(12,3)[:, 2]
        oa = trial[36]
        if not _check_nonoverlap(c, a):
            continue
        s_trial = _min_outer_side(_compute_all_vertices(c, a), oa)
        if s_trial < best_s:
            best_s = s_trial
            best_params = trial
    
    # Phase 4: Ultra-fine polish
    for _ in range(30000):
        if time.time() > deadline - 10.0:
            break
        trial = best_params + np.random.randn(37) * 0.004
        c = trial[:36].reshape(12,3)[:, :2]
        a = trial[:36].reshape(12,3)[:, 2]
        oa = trial[36]
        if not _check_nonoverlap(c, a):
            continue
        s_trial = _min_outer_side(_compute_all_vertices(c, a), oa)
        if s_trial < best_s:
            best_s = s_trial
            best_params = trial
    
    # Phase 5: Very fine polish
    for _ in range(20000):
        if time.time() > deadline - 3.0:
            break
        trial = best_params + np.random.randn(37) * 0.001
        c = trial[:36].reshape(12,3)[:, :2]
        a = trial[:36].reshape(12,3)[:, 2]
        oa = trial[36]
        if not _check_nonoverlap(c, a):
            continue
        s_trial = _min_outer_side(_compute_all_vertices(c, a), oa)
        if s_trial < best_s:
            best_s = s_trial
            best_params = trial
    
    centers_final = best_params[:36].reshape(12, 3)[:, :2]
    angles_final = best_params[:36].reshape(12, 3)[:, 2]
    outer_angle_final = best_params[36]
    
    centroid = np.mean(centers_final, axis=0)
    centers_final = centers_final - centroid
    
    all_verts_final = _compute_all_vertices(centers_final, angles_final)
    s_final = _min_outer_side(all_verts_final, outer_angle_final)
    
    if not _check_nonoverlap(centers_final, angles_final):
        centers_final, angles_final = _initial_honeycomb()
        outer_angle_final = 0.0
        all_verts_final = _compute_all_vertices(centers_final, angles_final)
        s_final = _min_outer_side(all_verts_final, outer_angle_final)
    
    inner_hex_data = np.column_stack([centers_final, angles_final])
    outer_hex_data = np.array([0.0, 0.0, outer_angle_final])
    
    return inner_hex_data, outer_hex_data, s_final

# EVOLVE-BLOCK-END
