# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def _hex_vertices(cx, cy, side, angle_deg):
    """Compute 6 vertices of a regular hexagon centered at (cx,cy) with given side and angle."""
    a = np.radians(angle_deg)
    angles = a + np.arange(6) * np.pi / 3.0
    return np.column_stack([
        cx + side * np.cos(angles),
        cy + side * np.sin(angles)
    ])

def _min_outer_side(vertices_all):
    """
    Given array of all inner hexagon vertices (shape (N,2)), compute minimum outer hex side s
    (centered at origin, angle 0) that contains all vertices.
    """
    k = np.arange(6)
    angles = (k + 0.5) * np.pi / 3  # 30, 90, 150, 210, 270, 330 degrees
    m = np.column_stack([np.cos(angles), np.sin(angles)])  # shape (6,2)
    
    dots = vertices_all @ m.T  # shape (N, 6)
    max_dot = np.max(dots, axis=1)  # shape (N,)
    global_max = np.max(max_dot)
    
    if global_max < 0:
        global_max = 0.0
    
    s_min = (2.0 / np.sqrt(3.0)) * global_max
    return s_min

def _min_outer_side_smooth(vertices_all, beta=100.0):
    """
    Smooth approximation of _min_outer_side using log-sum-exp for gradient-based optimization.
    """
    k = np.arange(6)
    angles = (k + 0.5) * np.pi / 3
    m = np.column_stack([np.cos(angles), np.sin(angles)])
    
    dots = vertices_all @ m.T  # shape (N, 6)
    max_dot = np.max(dots, axis=1)  # shape (N,)
    
    # Log-sum-exp for smooth max
    shifted = max_dot - np.max(max_dot)
    log_sum_exp = np.max(max_dot) + np.log(np.sum(np.exp(shifted)))
    smooth_max = log_sum_exp / beta
    
    if smooth_max < 0:
        smooth_max = 0.0
    
    s_smooth = (2.0 / np.sqrt(3.0)) * smooth_max
    return s_smooth

def _check_nonoverlap(centers, angles_deg, side=1.0, tol=1e-6):
    """
    Check if all pairs of hexagons are non-overlapping using SAT (vectorized).
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
        normals_list.append(np.array(normals))
    
    for i in range(n):
        for j in range(i+1, n):
            v1 = vertices_list[i]
            v2 = vertices_list[j]
            # SAT: check all 12 edge normals (6 from each hexagon)
            overlap = True
            for hex_idx in range(2):
                verts = [v1, v2][hex_idx]
                normals = normals_list[hex_idx]
                # Vectorized projection check
                proj1 = verts @ normals.T  # (6, 6)
                proj2 = v2 @ normals.T  # (6, 6)
                min1 = np.min(proj1, axis=0)
                max1 = np.max(proj1, axis=0)
                min2 = np.min(proj2, axis=0)
                max2 = np.max(proj2, axis=0)
                # Check if any axis separates
                separated = np.any(max1 + tol < min2) or np.any(max2 + tol < min1)
                if separated:
                    overlap = False
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

def _objective_smooth(params, beta=100.0):
    """
    Smooth objective for scipy optimizer: minimize smooth outer hex side s, with penalty for overlaps.
    params: (36,) = [x1,y1,theta1, x2,y2,theta2, ..., x12,y12,theta12]
    """
    params_2d = params.reshape(12, 3)
    centers = params_2d[:, :2]
    angles_deg = params_2d[:, 2]
    
    all_verts = _compute_all_vertices(centers, angles_deg)
    s = _min_outer_side_smooth(all_verts, beta)
    
    if not _check_nonoverlap(centers, angles_deg):
        return s + 50.0
    
    return s

def _objective_exact(params):
    """
    Exact objective for scipy optimizer: minimize outer hex side s, with penalty for overlaps.
    params: (36,) = [x1,y1,theta1, x2,y2,theta2, ..., x12,y12,theta12]
    """
    params_2d = params.reshape(12, 3)
    centers = params_2d[:, :2]
    angles_deg = params_2d[:, 2]
    
    all_verts = _compute_all_vertices(centers, angles_deg)
    s = _min_outer_side(all_verts)
    
    if not _check_nonoverlap(centers, angles_deg):
        return s + 50.0
    
    return s

def hexagon_packing_12():
    start_time = time.time()
    deadline = start_time + 610.0
    
    # Start with honeycomb arrangement
    centers, angles_deg = _initial_honeycomb()
    params = np.column_stack([centers, angles_deg]).flatten()
    
    all_verts = _compute_all_vertices(centers, angles_deg)
    best_s = _min_outer_side(all_verts)
    best_params = params.copy()
    
    if not _check_nonoverlap(centers, angles_deg):
        scale = 1.1
        while not _check_nonoverlap(centers * scale, angles_deg) and scale < 3.0:
            scale += 0.05
        centers = centers * scale
        params = np.column_stack([centers, angles_deg]).flatten()
        all_verts = _compute_all_vertices(centers, angles_deg)
        best_s = _min_outer_side(all_verts)
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
        
        new_params = current_params + np.random.randn(36) * step_size
        
        centers_new = new_params.reshape(12, 3)[:, :2]
        angles_new = new_params.reshape(12, 3)[:, 2]
        
        all_verts_new = _compute_all_vertices(centers_new, angles_new)
        s_new = _min_outer_side(all_verts_new)
        
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
            x0 = best_params + np.random.randn(36) * 0.1
        
        try:
            # First use smooth objective for convergence
            result_smooth = minimize(
                lambda p: _objective_smooth(p, beta=50.0),
                x0,
                method='L-BFGS-B',
                options={'maxiter': 300, 'maxls': 50},
                tol=1e-8
            )
            
            # Then refine with exact objective
            if result_smooth.fun < best_s + 1.0:
                result = minimize(
                    _objective_exact,
                    result_smooth.x,
                    method='L-BFGS-B',
                    options={'maxiter': 300, 'maxls': 50},
                    tol=1e-10
                )
                if result.fun < best_s and not np.any(np.isnan(result.fun)):
                    # Verify non-overlap
                    c = result.x.reshape(12, 3)[:, :2]
                    a = result.x.reshape(12, 3)[:, 2]
                    if _check_nonoverlap(c, a):
                        best_s = result.fun
                        best_params = result.x.copy()
        except Exception:
            pass
    
    # Final refinement with smaller steps
    for _ in range(20000):
        if time.time() > deadline - 20.0:
            break
        trial = best_params + np.random.randn(36) * 0.03
        c = trial.reshape(12,3)[:, :2]
        a = trial.reshape(12,3)[:, 2]
        if not _check_nonoverlap(c, a):
            continue
        s_trial = _min_outer_side(_compute_all_vertices(c, a))
        if s_trial < best_s:
            best_s = s_trial
            best_params = trial
    
    # Very fine polish
    for _ in range(5000):
        if time.time() > deadline - 5.0:
            break
        trial = best_params + np.random.randn(36) * 0.01
        c = trial.reshape(12,3)[:, :2]
        a = trial.reshape(12,3)[:, 2]
        if not _check_nonoverlap(c, a):
            continue
        s_trial = _min_outer_side(_compute_all_vertices(c, a))
        if s_trial < best_s:
            best_s = s_trial
            best_params = trial
    
    # High-precision local polish using scipy with tight tolerance
    if time.time() < deadline - 10.0:
        try:
            result_final = minimize(
                _objective_exact,
                best_params,
                method='L-BFGS-B',
                options={'maxiter': 1000, 'maxls': 100},
                tol=1e-12
            )
            if result_final.fun < best_s and not np.any(np.isnan(result_final.fun)):
                c = result_final.x.reshape(12, 3)[:, :2]
                a = result_final.x.reshape(12, 3)[:, 2]
                if _check_nonoverlap(c, a):
                    best_s = result_final.fun
                    best_params = result_final.x.copy()
        except Exception:
            pass
    
    centers_final = best_params.reshape(12, 3)[:, :2]
    angles_final = best_params.reshape(12, 3)[:, 2]
    
    centroid = np.mean(centers_final, axis=0)
    centers_final = centers_final - centroid
    
    all_verts_final = _compute_all_vertices(centers_final, angles_final)
    s_final = _min_outer_side(all_verts_final)
    
    if not _check_nonoverlap(centers_final, angles_final):
        centers_final, angles_final = _initial_honeycomb()
        all_verts_final = _compute_all_vertices(centers_final, angles_final)
        s_final = _min_outer_side(all_verts_final)
    
    inner_hex_data = np.column_stack([centers_final, angles_final])
    outer_hex_data = np.array([0.0, 0.0, 0.0])
    
    return inner_hex_data, outer_hex_data, s_final

# EVOLVE-BLOCK-END
