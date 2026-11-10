# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def _hex_vertices(cx, cy, side, angle_deg):
    """Compute 6 vertices of a regular hexagon."""
    a = np.radians(angle_deg)
    angles = a + np.arange(6) * np.pi / 3
    return np.column_stack([
        cx + side * np.cos(angles),
        cy + side * np.sin(angles)
    ])

def _outer_side_length_from_positions(positions, angles_deg):
    """
    Given inner hexagon centers (positions, shape (12,2)) and angles (shape (12,)),
    compute the minimum outer hexagon side length (centered at origin, angle 0)
    that contains all inner hexagons.
    """
    n = len(positions)
    normal_angles = np.radians([30, 90, 150, 210, 270, 330])
    normals = np.column_stack([np.cos(normal_angles), np.sin(normal_angles)])
    apothem_factor = np.sqrt(3) / 2
    
    max_proj = 0
    for i in range(n):
        verts = _hex_vertices(positions[i, 0], positions[i, 1], 1.0, angles_deg[i])
        projs = verts @ normals.T
        max_proj = max(max_proj, np.max(projs))
    
    s = max_proj / apothem_factor
    return s

def _check_validity(inner_hex_data, outer_hex_data, s, tol=1e-6):
    """Check if the packing is valid."""
    if not np.all(np.isfinite(inner_hex_data)) or not np.all(np.isfinite(outer_hex_data)):
        return False
    if s < np.sqrt(12) - 1e-9:
        return False
    
    outer_verts = _hex_vertices(outer_hex_data[0], outer_hex_data[1], s, outer_hex_data[2])
    for i in range(6):
        for j in range(i+1, 6):
            if np.linalg.norm(outer_verts[i] - outer_verts[j]) < 1e-10:
                return False
    
    for i in range(12):
        verts = _hex_vertices(inner_hex_data[i, 0], inner_hex_data[i, 1], 1.0, inner_hex_data[i, 2])
        for k in range(6):
            edge_len = np.linalg.norm(verts[(k+1)%6] - verts[k])
            if abs(edge_len - 1.0) > 1e-8:
                return False
    
    # SAT check for separation
    for i in range(12):
        for j in range(i+1, 12):
            verts_i = _hex_vertices(inner_hex_data[i, 0], inner_hex_data[i, 1], 1.0, inner_hex_data[i, 2])
            verts_j = _hex_vertices(inner_hex_data[j, 0], inner_hex_data[j, 1], 1.0, inner_hex_data[j, 2])
            
            separated = False
            # Check normals of hex i
            for k in range(6):
                edge = verts_i[(k+1)%6] - verts_i[k]
                normal = np.array([-edge[1], edge[0]])
                norm = np.linalg.norm(normal)
                if norm > 1e-12:
                    normal = normal / norm
                    proj_i = verts_i @ normal
                    proj_j = verts_j @ normal
                    if np.max(proj_i) + tol < np.min(proj_j) or np.max(proj_j) + tol < np.min(proj_i):
                        separated = True
                        break
            
            if not separated:
                # Check normals of hex j
                for k in range(6):
                    edge = verts_j[(k+1)%6] - verts_j[k]
                    normal = np.array([-edge[1], edge[0]])
                    norm = np.linalg.norm(normal)
                    if norm > 1e-12:
                        normal = normal / norm
                        proj_i = verts_i @ normal
                        proj_j = verts_j @ normal
                        if np.max(proj_i) + tol < np.min(proj_j) or np.max(proj_j) + tol < np.min(proj_i):
                            separated = True
                            break
            
            if not separated:
                return False
    
    # Containment check
    for i in range(12):
        verts = _hex_vertices(inner_hex_data[i, 0], inner_hex_data[i, 1], 1.0, inner_hex_data[i, 2])
        for v in verts:
            for k in range(6):
                p1 = outer_verts[k]
                p2 = outer_verts[(k+1)%6]
                cross = (p2[0]-p1[0])*(v[1]-p1[1]) - (p2[1]-p1[1])*(v[0]-p1[0])
                if cross < -tol:
                    return False
    
    return True

def _smooth_objective(params, beta=50.0):
    """
    Smooth surrogate objective using log-sum-exp over all vertex-normal projections.
    params: (36,) array with [x0, y0, a0, x1, y1, a1, ...]
    Returns objective value.
    """
    positions = np.column_stack([params[0::3], params[1::3]])
    angles_deg = params[2::3]
    
    normal_angles = np.radians([30, 90, 150, 210, 270, 330])
    normals = np.column_stack([np.cos(normal_angles), np.sin(normal_angles)])
    apothem_factor = np.sqrt(3) / 2
    
    # Compute all projections: shape (12, 6, 6) = (hex, vertex, normal)
    all_projs = np.zeros((12, 6, 6))
    for i in range(12):
        verts = _hex_vertices(positions[i, 0], positions[i, 1], 1.0, angles_deg[i])
        all_projs[i] = verts @ normals.T
    
    # Flatten all projections
    flat = all_projs.flatten()
    
    # Log-sum-exp for numerical stability
    max_val = np.max(flat)
    if max_val > 1e15:
        return max_val / apothem_factor  # Fallback for overflow
    
    log_sum_exp = max_val + np.log(np.sum(np.exp(flat - max_val)))
    
    # Smooth approximation to max: (1/beta) * log(sum(exp(beta * x)))
    beta_max = beta * max_val + np.log(np.sum(np.exp(beta * (flat - max_val))))
    s = beta_max / beta / apothem_factor
    return s

def _smooth_objective_and_gradient(params, beta=50.0):
    """
    Smooth surrogate objective with analytical gradient.
    params: (36,) array with [x0, y0, a0, x1, y1, a1, ...]
    Returns (objective_value, gradient)
    """
    positions = np.column_stack([params[0::3], params[1::3]])
    angles_deg = params[2::3]
    
    normal_angles = np.radians([30, 90, 150, 210, 270, 330])
    normals = np.column_stack([np.cos(normal_angles), np.sin(normal_angles)])
    apothem_factor = np.sqrt(3) / 2
    
    # Compute all projections: shape (12, 6, 6) = (hex, vertex, normal)
    all_projs = np.zeros((12, 6, 6))
    for i in range(12):
        verts = _hex_vertices(positions[i, 0], positions[i, 1], 1.0, angles_deg[i])
        all_projs[i] = verts @ normals.T
    
    flat = all_projs.flatten()
    
    # Numerically stable log-sum-exp
    max_val = np.max(flat)
    shifted = flat - max_val
    exp_shifted = np.exp(shifted)
    sum_exp = np.sum(exp_shifted)
    
    if sum_exp < 1e-300:
        return max_val / apothem_factor, np.zeros(36)
    
    # s = (1/beta) * (beta*max_val + log(sum(exp(beta*(x - max_val))))) / apothem_factor
    #    = (max_val + (1/beta)*log(sum(exp(beta*(x - max_val))))) / apothem_factor
    # Simplified: s = (max_val + log(sum(exp(beta*(x-max_val))))/beta) / apothem_factor
    
    log_sum = np.log(sum_exp)
    s = (max_val + log_sum / beta) / apothem_factor
    
    # Gradient: ds/dparam = (1/(beta * apothem_factor)) * sum_j exp(beta*(x_j - max_val)) * dx_j/dparam / sum_j exp(...)
    # = (1/(apothem_factor)) * sum_j w_j * dx_j/dparam where w_j = exp(beta*(x_j-max_val)) / sum_exp
    
    weights = exp_shifted / sum_exp  # shape (432,)
    
    grad = np.zeros(36)
    
    for i in range(12):
        a = np.radians(angles_deg[i])
        for j in range(6):  # vertex index
            angle_j = a + j * np.pi / 3
            cos_j = np.cos(angle_j)
            sin_j = np.sin(angle_j)
            
            for k in range(6):  # normal index
                w = weights[i*36 + j*6 + k]
                if w < 1e-15:
                    continue
                # dx/d(cx) = normals[k,0]
                # dx/d(cy) = normals[k,1]
                # dx/d(angle) = -sin(angle_j)*normals[k,0] + cos(angle_j)*normals[k,1]
                grad[3*i] += w * normals[k, 0]
                grad[3*i+1] += w * normals[k, 1]
                grad[3*i+2] += w * (-sin_j * normals[k, 0] + cos_j * normals[k, 1])
    
    grad /= apothem_factor
    
    return s, grad

def _objective_and_gradient(params):
    """
    Objective function and analytical gradient for minimizing outer hexagon side length.
    params: (36,) array with [x0, y0, a0, x1, y1, a1, ...]
    Returns (objective_value, gradient)
    """
    positions = np.column_stack([params[0::3], params[1::3]])
    angles_deg = params[2::3]
    
    normal_angles = np.radians([30, 90, 150, 210, 270, 330])
    normals = np.column_stack([np.cos(normal_angles), np.sin(normal_angles)])
    apothem_factor = np.sqrt(3) / 2
    
    # Compute vertices for each hexagon
    all_verts = np.zeros((12, 6, 2))
    for i in range(12):
        all_verts[i] = _hex_vertices(positions[i, 0], positions[i, 1], 1.0, angles_deg[i])
    
    # Compute projections onto each normal
    max_proj = 0
    max_idx = None  # (hex_idx, vert_idx, normal_idx)
    
    for i in range(12):
        for j in range(6):
            v = all_verts[i, j]
            for k in range(6):
                proj = v[0] * normals[k, 0] + v[1] * normals[k, 1]
                if proj > max_proj:
                    max_proj = proj
                    max_idx = (i, j, k)
    
    s = max_proj / apothem_factor
    
    # Compute gradient
    grad = np.zeros(36)
    if max_idx is not None:
        i, j, k = max_idx
        a = np.radians(angles_deg[i])
        angle_j = a + j * np.pi / 3
        cos_j = np.cos(angle_j)
        sin_j = np.sin(angle_j)
        
        grad[3*i] = normals[k, 0] / apothem_factor
        grad[3*i+1] = normals[k, 1] / apothem_factor
        grad[3*i+2] = (-sin_j * normals[k, 0] + cos_j * normals[k, 1]) / apothem_factor
    
    return s, grad

def hexagon_packing_12():
    start_time = time.time()
    deadline = 600.0
    
    # Tighter hexagonal packing seed
    sqrt3 = np.sqrt(3)
    hex_positions = np.array([
        [0, 0],
        [3, 0],
        [1.5, sqrt3],
        [-1.5, sqrt3],
        [-3, 0],
        [-1.5, -sqrt3],
        [1.5, -sqrt3],
        [4.5, sqrt3],
        [3, 2*sqrt3],
        [0, 2*sqrt3],
        [-3, 2*sqrt3],
        [-4.5, sqrt3]
    ])
    hex_angles = np.zeros(12)
    
    # Generate multiple starting configurations
    configs = []
    
    # Config 1: Hexagonal packing
    configs.append((hex_positions, hex_angles))
    
    # Config 2: Hexagonal packing with various rotations
    for rot in [15, 30, 45, 60]:
        angles = np.full(12, rot)
        configs.append((hex_positions.copy(), angles))
    
    # Config 3: Perturbed hexagonal packing
    for trial in range(30):
        np.random.seed(trial * 100 + 42)
        perturbed_pos = hex_positions + np.random.randn(12, 2) * 0.15
        angles = np.random.uniform(-30, 30, 12)
        configs.append((perturbed_pos, angles))
    
    # Config 4: Random positions
    for trial in range(30):
        if time.time() > deadline - 60:
            break
        np.random.seed(trial * 1000 + 7)
        pos = np.zeros((12, 2))
        for i in range(12):
            r = np.sqrt(np.random.uniform(0, 1)) * 4
            theta = np.random.uniform(0, 2*np.pi)
            pos[i] = [r * np.cos(theta), r * np.sin(theta)]
        angles = np.random.uniform(-30, 30, 12)
        configs.append((pos, angles))
    
    best_s = float('inf')
    best_params = None
    
    def objective_smooth(params):
        return _smooth_objective(params, beta=50.0)
    
    def objective_smooth_grad(params):
        return _smooth_objective_and_gradient(params, beta=50.0)
    
    def objective(params):
        val, _ = _objective_and_gradient(params)
        return val
    
    def objective_grad(params):
        val, grad = _objective_and_gradient(params)
        return val, grad
    
    for config_idx, (pos, ang) in enumerate(configs):
        if time.time() > deadline - 30:
            break
        
        # Pack into params vector
        p = np.zeros(36)
        for i in range(12):
            p[3*i] = pos[i, 0]
            p[3*i+1] = pos[i, 1]
            p[3*i+2] = ang[i]
        
        # Phase 1: Smooth optimization (better for finding good basins)
        try:
            result_smooth = minimize(
                objective_smooth_grad,
                p,
                method='L-BFGS-B',
                jac=True,
                options={
                    'maxiter': 3000,
                    'ftol': 1e-12,
                    'gtol': 1e-8,
                    'maxfun': 80000
                }
            )
            
            if result_smooth.success or np.isfinite(result_smooth.fun):
                p_smooth = result_smooth.x
                s_smooth = _outer_side_length_from_positions(
                    np.column_stack([p_smooth[0::3], p_smooth[1::3]]),
                    p_smooth[2::3]
                )
                
                if s_smooth < best_s:
                    best_s = s_smooth
                    best_params = p_smooth.copy()
        except Exception as e:
            pass
        
        # Phase 2: Sharp optimization from smooth result or original
        p_start = best_params.copy() if best_params is not None else p
        
        try:
            result = minimize(
                objective_grad,
                p_start,
                method='L-BFGS-B',
                jac=True,
                options={
                    'maxiter': 3000,
                    'ftol': 1e-13,
                    'gtol': 1e-9,
                    'maxfun': 80000
                }
            )
            
            if result.success or np.isfinite(result.fun):
                p_opt = result.x
                s_opt = _outer_side_length_from_positions(
                    np.column_stack([p_opt[0::3], p_opt[1::3]]),
                    p_opt[2::3]
                )
                
                if s_opt < best_s:
                    best_s = s_opt
                    best_params = p_opt.copy()
        except Exception as e:
            pass
        
        # Phase 3: Nelder-Mead polish
        if time.time() < deadline - 20:
            try:
                result_nm = minimize(
                    objective,
                    best_params if best_params is not None else p,
                    method='Nelder-Mead',
                    options={
                        'maxiter': 5000,
                        'xatol': 1e-10,
                        'fatol': 1e-12,
                        'adaptive': True
                    }
                )
                
                if np.isfinite(result_nm.fun):
                    p_nm = result_nm.x
                    s_nm = _outer_side_length_from_positions(
                        np.column_stack([p_nm[0::3], p_nm[1::3]]),
                        p_nm[2::3]
                    )
                    
                    if s_nm < best_s:
                        best_s = s_nm
                        best_params = p_nm.copy()
            except Exception as e:
                pass
    
    # If no optimization succeeded, use the hexagonal packing as fallback
    if best_params is None:
        best_params = np.zeros(36)
        for i in range(12):
            best_params[3*i] = hex_positions[i, 0]
            best_params[3*i+1] = hex_positions[i, 1]
            best_params[3*i+2] = hex_angles[i]
        best_s = _outer_side_length_from_positions(hex_positions, hex_angles)
    
    # Final refinement with gradient-based method
    if time.time() < deadline - 15:
        try:
            result_final = minimize(
                objective_grad,
                best_params,
                method='L-BFGS-B',
                jac=True,
                options={
                    'maxiter': 8000,
                    'ftol': 1e-14,
                    'gtol': 1e-10,
                    'maxfun': 200000
                }
            )
            
            if np.isfinite(result_final.fun):
                p_final = result_final.x
                s_final = _outer_side_length_from_positions(
                    np.column_stack([p_final[0::3], p_final[1::3]]),
                    p_final[2::3]
                )
                
                if s_final < best_s:
                    best_s = s_final
                    best_params = p_final.copy()
        except Exception as e:
            pass
    
    # Additional Nelder-Mead refinement
    if time.time() < deadline - 5:
        try:
            result_nm_final = minimize(
                objective,
                best_params,
                method='Nelder-Mead',
                options={
                    'maxiter': 15000,
                    'xatol': 1e-12,
                    'fatol': 1e-14,
                    'adaptive': True
                }
            )
            
            if np.isfinite(result_nm_final.fun):
                p_nm_final = result_nm_final.x
                s_nm_final = _outer_side_length_from_positions(
                    np.column_stack([p_nm_final[0::3], p_nm_final[1::3]]),
                    p_nm_final[2::3]
                )
                
                if s_nm_final < best_s:
                    best_s = s_nm_final
                    best_params = p_nm_final.copy()
        except Exception as e:
            pass
    
    # Construct output
    inner_hex_data = np.zeros((12, 3))
    for i in range(12):
        inner_hex_data[i, 0] = best_params[3*i]
        inner_hex_data[i, 1] = best_params[3*i+1]
        inner_hex_data[i, 2] = best_params[3*i+2]
    
    outer_hex_data = np.array([0.0, 0.0, 0.0])
    s = _outer_side_length_from_positions(
        np.column_stack([best_params[0::3], best_params[1::3]]),
        best_params[2::3]
    )
    
    # Verify validity; if invalid, fall back to seed
    if not _check_validity(inner_hex_data, outer_hex_data, s):
        inner_hex_data = np.array(
            [
                [0, 0, 0],
                [-2.5, 0, 0],
                [2.5, 0, 0],
                [-1.25, 2.17, 0],
                [1.25, 2.17, 0],
                [-1.25, -2.17, 0],
                [1.25, -2.17, 0],
                [-3.75, 2.17, 0],
                [3.75, 2.17, 0],
                [-3.75, -2.17, 0],
                [3.75, -2.17, 0],
                [0, -4, 0]
            ]
        )
        outer_hex_data = np.array([0, 0, 0])
        s = 8.0
    
    return inner_hex_data, outer_hex_data, s

# EVOLVE-BLOCK-END
