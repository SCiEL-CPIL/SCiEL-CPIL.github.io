# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize, Bounds

def _hex_vertices(cx, cy, side, angle_deg):
    """Compute 6 vertices of a regular hexagon centered at (cx,cy) with given side and angle."""
    a = np.radians(angle_deg)
    angles = a + np.arange(6) * np.pi / 3.0
    return np.column_stack([
        cx + side * np.cos(angles),
        cy + side * np.sin(angles)
    ])

def _min_outer_side(vertices_all, outer_angle_deg=0.0):
    """
    Given array of all inner hexagon vertices (shape (N,2)), compute minimum outer hex side s
    (centered at origin, with given angle) that contains all vertices.
    """
    k = np.arange(6)
    base_angles = (k + 0.5) * np.pi / 3  # 30, 90, 150, 210, 270, 330 degrees
    angles = base_angles + np.radians(outer_angle_deg)
    m = np.column_stack([np.cos(angles), np.sin(angles)])  # shape (6,2)
    
    dots = vertices_all @ m.T  # shape (N, 6)
    max_dot = np.max(dots, axis=1)  # shape (N,)
    global_max = np.max(max_dot)
    
    if global_max < 0:
        global_max = 0.0
    
    s_min = (2.0 / np.sqrt(3.0)) * global_max
    return s_min

def _min_outer_side_smooth(vertices_all, outer_angle_deg=0.0, beta=50.0):
    """
    Smooth approximation of _min_outer_side using log-sum-exp.
    """
    k = np.arange(6)
    base_angles = (k + 0.5) * np.pi / 3
    angles = base_angles + np.radians(outer_angle_deg)
    m = np.column_stack([np.cos(angles), np.sin(angles)])
    
    dots = vertices_all @ m.T  # shape (N, 6)
    # For each vertex, max over 6 directions; then max over vertices
    # Use log-sum-exp for smooth max
    scaled_dots = dots * beta
    # Smooth max over the 6 directions for each vertex
    max_scaled = np.max(scaled_dots, axis=1)
    smooth_max_vertex = max_scaled / beta + np.log(np.sum(np.exp(scaled_dots - max_scaled[:, None]), axis=1)) / beta
    
    # Smooth max over all vertices
    global_max_scaled = np.max(smooth_max_vertex * beta)
    smooth_global = global_max_scaled / beta + np.log(np.sum(np.exp(smooth_max_vertex * beta - global_max_scaled))) / beta
    
    if smooth_global < 0:
        smooth_global = 0.0
    
    s_min = (2.0 / np.sqrt(3.0)) * smooth_global
    return s_min

def _check_nonoverlap(centers, angles_deg, side=1.0, tol=1e-6):
    """
    Check if all pairs of hexagons are non-overlapping using SAT.
    Returns True if no overlap, False otherwise.
    centers: (12,2), angles_deg: (12,)
    """
    n = len(centers)
    vertices_list = []
    for i in range(n):
        verts = _hex_vertices(centers[i,0], centers[i,1], side, angles_deg[i])
        vertices_list.append(verts)
    
    # Precompute edge normals for each hexagon
    normals_list = []
    for verts in vertices_list:
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
            # SAT: check all 12 edge normals (6 from each hexagon)
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

def _compute_all_vertices(centers, angles_deg, side=1.0):
    """Compute all 72 vertices of 12 hexagons."""
    verts = []
    for i in range(len(centers)):
        verts.append(_hex_vertices(centers[i,0], centers[i,1], side, angles_deg[i]))
    return np.vstack(verts)

def _initial_honeycomb():
    """
    Create initial honeycomb-like arrangement of 12 hexagons.
    """
    d = np.sqrt(3.0)
    positions = []
    angles = []
    
    # Center
    positions.append([0.0, 0.0])
    angles.append(0.0)
    
    # Inner ring: 6 hexagons
    for k in range(6):
        angle = k * np.pi / 3
        x = d * np.cos(angle)
        y = d * np.sin(angle)
        positions.append([x, y])
        angles.append(0.0)
    
    # Outer ring: 5 more hexagons
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

def _objective_scipy(params):
    """
    Objective for scipy optimizer: minimize outer hex side s, with penalty for overlaps.
    params: (37,) = [x1,y1,theta1, ..., x12,y12,theta12, outer_angle]
    """
    params_2d = params.reshape(12, 3)
    centers = params_2d[:, :2]
    angles_deg = params_2d[:, 2]
    outer_angle = params[36]
    
    all_verts = _compute_all_vertices(centers, angles_deg)
    s = _min_outer_side(all_verts, outer_angle)
    
    if not _check_nonoverlap(centers, angles_deg):
        return s + 50.0
    
    return s

def _objective_smooth(params, beta=50.0):
    """
    Smooth objective for scipy optimizer using log-sum-exp surrogate.
    params: (37,) = [x1,y1,theta1, ..., x12,y12,theta12, outer_angle]
    """
    params_2d = params.reshape(12, 3)
    centers = params_2d[:, :2]
    angles_deg = params_2d[:, 2]
    outer_angle = params[36]
    
    all_verts = _compute_all_vertices(centers, angles_deg)
    s = _min_outer_side_smooth(all_verts, outer_angle, beta)
    
    # Overlap penalty using smooth distance
    if not _check_nonoverlap(centers, angles_deg):
        s += 50.0
    
    return s

def hexagon_packing_12():
    start_time = time.time()
    deadline = start_time + 610.0
    
    # Start with honeycomb arrangement
    centers, angles_deg = _initial_honeycomb()
    # params: 36 for inner hexagons + 1 for outer angle = 37 total
    params = np.column_stack([centers, angles_deg]).flatten()
    params = np.append(params, 0.0)  # outer angle
    
    all_verts = _compute_all_vertices(centers, angles_deg)
    best_s = _min_outer_side(all_verts, 0.0)
    best_params = params.copy()
    
    if not _check_nonoverlap(centers, angles_deg):
        scale = 1.1
        while not _check_nonoverlap(centers * scale, angles_deg) and scale < 3.0:
            scale += 0.05
        centers = centers * scale
        params = np.column_stack([centers, angles_deg]).flatten()
        params = np.append(params, 0.0)
        all_verts = _compute_all_vertices(centers, angles_deg)
        best_s = _min_outer_side(all_verts, 0.0)
        best_params = params.copy()
    
    # Simulated annealing phase 1: coarse
    np.random.seed(42)
    T = 0.5
    cooling = 0.9998
    step_size = 0.3
    accept_count = 0
    total_count = 0
    
    current_params = best_params.copy()
    current_s = best_s
    
    max_iters = 300000
    for it in range(max_iters):
        if time.time() > deadline - 200.0:
            break
        
        # Perturb: 36 params for inner + 1 for outer angle (smaller step for angle)
        noise = np.random.randn(37) * step_size
        noise[36] *= 0.5  # smaller step for outer angle
        
        new_params = current_params + noise
        
        centers_new = new_params[:36].reshape(12, 3)[:, :2]
        angles_new = new_params[:36].reshape(12, 3)[:, 2]
        outer_angle_new = new_params[36]
        
        all_verts_new = _compute_all_vertices(centers_new, angles_new)
        s_new = _min_outer_side(all_verts_new, outer_angle_new)
        
        if not _check_nonoverlap(centers_new, angles_new):
            s_new += 50.0
        
        delta = s_new - current_s
        total_count += 1
        
        if delta < 0 or np.random.random() < np.exp(-delta / max(T, 1e-10)):
            current_params = new_params
            current_s = s_new
            accept_count += 1
            if s_new < best_s:
                best_s = s_new
                best_params = new_params.copy()
        
        T *= cooling
        
        # Adaptive step size
        if it % 500 == 0 and it > 0:
            acc_rate = accept_count / total_count
            if acc_rate < 0.2:
                step_size *= 0.8
            elif acc_rate > 0.8:
                step_size *= 1.2
            step_size = np.clip(step_size, 0.005, 1.0)
            accept_count = 0
            total_count = 0
    
    # Scipy optimization phase: use multiple starting points
    n_starts = 5
    for si in range(n_starts):
        if time.time() > deadline - 60.0:
            break
        
        if si == 0:
            x0 = best_params.copy()
        else:
            # Perturbed starting points
            x0 = best_params + np.random.randn(37) * 0.1
            x0[36] *= 0.5
        
        try:
            result = minimize(
                _objective_scipy,
                x0,
                method='L-BFGS-B',
                options={'maxiter': 500, 'maxls': 50},
                tol=1e-10
            )
            if result.fun < best_s and not np.any(np.isnan(result.fun)):
                # Verify non-overlap
                c = result.x[:36].reshape(12, 3)[:, :2]
                a = result.x[:36].reshape(12, 3)[:, 2]
                if _check_nonoverlap(c, a):
                    best_s = result.fun
                    best_params = result.x.copy()
        except Exception:
            pass
    
    # Smooth optimization phase for finer convergence
    for si in range(3):
        if time.time() > deadline - 30.0:
            break
        
        if si == 0:
            x0 = best_params.copy()
        else:
            x0 = best_params + np.random.randn(37) * 0.05
            x0[36] *= 0.5
        
        try:
            result = minimize(
                lambda p: _objective_smooth(p, beta=100.0),
                x0,
                method='L-BFGS-B',
                options={'maxiter': 300, 'maxls': 30},
                tol=1e-12
            )
            if result.fun < best_s and not np.any(np.isnan(result.fun)):
                c = result.x[:36].reshape(12, 3)[:, :2]
                a = result.x[:36].reshape(12, 3)[:, 2]
                oa = result.x[36]
                # Verify with exact objective
                all_v = _compute_all_vertices(c, a)
                s_exact = _min_outer_side(all_v, oa)
                if _check_nonoverlap(c, a) and s_exact < best_s:
                    best_s = s_exact
                    best_params = result.x.copy()
        except Exception:
            pass
    
    # Final refinement with smaller steps
    for _ in range(20000):
        if time.time() > deadline - 20.0:
            break
        noise = np.random.randn(37) * 0.03
        noise[36] *= 0.5
        trial = best_params + noise
        c = trial[:36].reshape(12,3)[:, :2]
        a = trial[:36].reshape(12,3)[:, 2]
        oa = trial[36]
        if not _check_nonoverlap(c, a):
            continue
        s_trial = _min_outer_side(_compute_all_vertices(c, a), oa)
        if s_trial < best_s:
            best_s = s_trial
            best_params = trial
    
    # Very fine polish
    for _ in range(5000):
        if time.time() > deadline - 5.0:
            break
        noise = np.random.randn(37) * 0.01
        noise[36] *= 0.5
        trial = best_params + noise
        c = trial[:36].reshape(12,3)[:, :2]
        a = trial[:36].reshape(12,3)[:, 2]
        oa = trial[36]
        if not _check_nonoverlap(c, a):
            continue
        s_trial = _min_outer_side(_compute_all_vertices(c, a), oa)
        if s_trial < best_s:
            best_s = s_trial
            best_params = trial
    
    # Dedicated outer angle optimization: fix inner positions, sweep outer angle
    c_final = best_params[:36].reshape(12, 3)[:, :2]
    a_final = best_params[:36].reshape(12, 3)[:, 2]
    all_v_final = _compute_all_vertices(c_final, a_final)
    
    best_oe = best_params[36]
    best_s_oe = best_s
    for oa in np.linspace(0, 60, 181):
        s_oe = _min_outer_side(all_v_final, oa)
        if s_oe < best_s_oe:
            best_s_oe = s_oe
            best_oe = oa
    
    # Local refinement of outer angle
    for _ in range(1000):
        oa_trial = best_oe + np.random.randn() * 0.5
        s_oe = _min_outer_side(all_v_final, oa_trial)
        if s_oe < best_s_oe:
            best_s_oe = s_oe
            best_oe = oa_trial
    
    best_params[36] = best_oe
    best_s = best_s_oe
    
    centers_final = best_params[:36].reshape(12, 3)[:, :2]
    angles_final = best_params[:36].reshape(12, 3)[:, 2]
    outer_angle_final = best_params[36]
    
    centroid = np.mean(centers_final, axis=0)
    centers_final = centers_final - centroid
    
    all_verts_final = _compute_all_vertices(centers_final, angles_final)
    s_final = _min_outer_side(all_verts_final, outer_angle_final)
    
    if not _check_nonoverlap(centers_final, angles_final):
        centers_final, angles_final = _initial_honeycomb()
        all_verts_final = _compute_all_vertices(centers_final, angles_final)
        s_final = _min_outer_side(all_verts_final, 0.0)
        outer_angle_final = 0.0
    
    inner_hex_data = np.column_stack([centers_final, angles_final])
    outer_hex_data = np.array([0.0, 0.0, outer_angle_final])
    
    return inner_hex_data, outer_hex_data, s_final

# EVOLVE-BLOCK-END
