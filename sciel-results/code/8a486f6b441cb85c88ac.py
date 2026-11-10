# EVOLVE-BLOCK-START
import numpy as np
import time

def _hex_vertices(cx, cy, side, angle_deg):
    a = np.radians(angle_deg)
    angles = a + np.arange(6) * np.pi / 3
    return np.column_stack([
        cx + side * np.cos(angles),
        cy + side * np.sin(angles)
    ])

def _outer_side_length_from_positions(positions, angles_deg):
    n = len(positions)
    normal_angles = np.radians([30, 90, 150, 210, 270, 330])
    normals = np.column_stack([np.cos(normal_angles), np.sin(normal_angles)])
    apothem_factor = np.sqrt(3) / 2
    max_proj = 0.0
    for i in range(n):
        verts = _hex_vertices(positions[i, 0], positions[i, 1], 1.0, angles_deg[i])
        projs = verts @ normals.T
        max_proj = max(max_proj, np.max(projs))
    s = max_proj / apothem_factor
    return max(s, np.sqrt(12) - 1e-12)

def _check_validity(inner_hex_data, outer_hex_data, s, tol=1e-6):
    if not np.all(np.isfinite(inner_hex_data)) or not np.all(np.isfinite(outer_hex_data)):
        return False
    if s < np.sqrt(12) - 1e-9:
        return False
    outer_verts = _hex_vertices(outer_hex_data[0], outer_hex_data[1], s, outer_hex_data[2])
    for i in range(6):
        for j in range(i + 1, 6):
            if np.linalg.norm(outer_verts[i] - outer_verts[j]) < 1e-10:
                return False
    for i in range(12):
        verts = _hex_vertices(inner_hex_data[i, 0], inner_hex_data[i, 1], 1.0, inner_hex_data[i, 2])
        for k in range(6):
            edge_len = np.linalg.norm(verts[(k + 1) % 6] - verts[k])
            if abs(edge_len - 1.0) > 1e-8:
                return False
    # SAT check
    for i in range(12):
        for j in range(i + 1, 12):
            verts_i = _hex_vertices(inner_hex_data[i, 0], inner_hex_data[i, 1], 1.0, inner_hex_data[i, 2])
            verts_j = _hex_vertices(inner_hex_data[j, 0], inner_hex_data[j, 1], 1.0, inner_hex_data[j, 2])
            separated = False
            for k in range(6):
                edge = verts_i[(k + 1) % 6] - verts_i[k]
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
                for k in range(6):
                    edge = verts_j[(k + 1) % 6] - verts_j[k]
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
    # Containment
    for i in range(12):
        verts = _hex_vertices(inner_hex_data[i, 0], inner_hex_data[i, 1], 1.0, inner_hex_data[i, 2])
        for v in verts:
            for k in range(6):
                p1 = outer_verts[k]
                p2 = outer_verts[(k + 1) % 6]
                cross = (p2[0] - p1[0]) * (v[1] - p1[1]) - (p2[1] - p1[1]) * (v[0] - p1[0])
                if cross < -tol:
                    return False
    return True

def _smooth_objective(params, beta=50.0):
    """
    Smooth differentiable objective for pressure-based optimization.
    Uses log-sum-exp to approximate the max projection (outer side length),
    and quadratic penalties for overlap (SAT-based) and containment.
    """
    pos = np.column_stack([params[0::3], params[1::3]])
    ang = params[2::3]
    
    # Compute all vertex positions for all 12 hexagons
    all_verts = np.zeros((12, 6, 2))
    for i in range(12):
        all_verts[i] = _hex_vertices(pos[i, 0], pos[i, 1], 1.0, ang[i])
    
    # Outer hexagon normals (for a hexagon with angle 0, the outward normals are at 30, 90, 150, 210, 270, 330)
    normal_angles = np.radians([30, 90, 150, 210, 270, 330])
    normals = np.column_stack([np.cos(normal_angles), np.sin(normal_angles)])  # (6, 2)
    apothem_factor = np.sqrt(3) / 2
    
    # For each hexagon vertex, project onto each normal direction
    # proj[i, j, k] = dot(vertex_i_j, normal_k)
    # Shape: (12, 6, 6)
    proj = np.einsum('ijc,c->ij', all_verts.reshape(72, 2), normals)
    proj = proj.reshape(12, 6, 6)
    
    # Smooth max over all vertices and normal directions using log-sum-exp
    flat_proj = proj.reshape(-1)
    # Add a large negative bias so log-sum-exp approximates max
    m = np.max(flat_proj)
    if m > 1e10:
        m = 1e10
    lse = m + np.log(np.sum(np.exp((flat_proj - m) * beta)) / beta)
    s_smooth = lse / apothem_factor
    s_smooth = max(s_smooth, np.sqrt(12) - 1e-12)
    
    # Overlap penalty: sum of squared penetration depths over all pairs
    overlap_pen = 0.0
    for i in range(12):
        for j in range(i + 1, 12):
            verts_i = all_verts[i]
            verts_j = all_verts[j]
            # Check all 12 edge normals (6 from each)
            for k in range(6):
                edge_i = verts_i[(k + 1) % 6] - verts_i[k]
                normal_i = np.array([-edge_i[1], edge_i[0]])
                norm_i = np.linalg.norm(normal_i)
                if norm_i > 1e-12:
                    normal_i = normal_i / norm_i
                    proj_i = verts_i @ normal_i
                    proj_j = verts_j @ normal_i
                    pen = np.max(proj_i) - np.min(proj_j)
                    if pen > 0:
                        overlap_pen += pen * pen
            for k in range(6):
                edge_j = verts_j[(k + 1) % 6] - verts_j[k]
                normal_j = np.array([-edge_j[1], edge_j[0]])
                norm_j = np.linalg.norm(normal_j)
                if norm_j > 1e-12:
                    normal_j = normal_j / norm_j
                    proj_i = verts_i @ normal_j
                    proj_j = verts_j @ normal_j
                    pen = np.max(proj_i) - np.min(proj_j)
                    if pen > 0:
                        overlap_pen += pen * pen
    
    # Containment penalty: vertices outside the outer hexagon
    # Outer hexagon centered at origin with side s_smooth and angle 0
    s_for_contain = max(s_smooth, 2.0)
    outer_verts = _hex_vertices(0.0, 0.0, s_for_contain, 0.0)
    contain_pen = 0.0
    for i in range(12):
        for v in all_verts[i]:
            for k in range(6):
                p1 = outer_verts[k]
                p2 = outer_verts[(k + 1) % 6]
                cross = (p2[0] - p1[0]) * (v[1] - p1[1]) - (p2[1] - p1[1]) * (v[0] - p1[0])
                if cross < 0:
                    contain_pen += cross * cross
    
    return s_smooth + 50.0 * overlap_pen + 50.0 * contain_pen

def hexagon_packing_12():
    start_time = time.time()
    deadline = 600.0
    sqrt3 = np.sqrt(3)
    
    # Fallback valid configuration
    seed_positions = np.array([
        [0, 0], [-2.5, 0], [2.5, 0],
        [-1.25, 2.17], [1.25, 2.17], [-1.25, -2.17],
        [1.25, -2.17], [-3.75, 2.17], [3.75, 2.17],
        [-3.75, -2.17], [3.75, -2.17], [0, -4]
    ])
    seed_angles = np.zeros(12)
    
    # Honeycomb configuration (center + ring of 6 + 5 more)
    honeycomb_positions = np.array([
        [0, 0],
        [3, 0], [1.5, sqrt3], [-1.5, sqrt3],
        [-3, 0], [-1.5, -sqrt3], [1.5, -sqrt3],
        [4.5, sqrt3], [3, 2*sqrt3], [0, 2*sqrt3],
        [-3, 2*sqrt3], [-4.5, sqrt3]
    ])
    honeycomb_angles = np.zeros(12)
    
    # Generate diverse starting configurations
    starting_configs = []
    
    # 1. Honeycomb with 0 angle
    starting_configs.append((honeycomb_positions.copy(), honeycomb_angles.copy()))
    
    # 2. Seed arrangement
    starting_configs.append((seed_positions.copy(), seed_angles.copy()))
    
    # 3. Various rotated honeycombs
    for rot in [10, 20, 30, 40, 50]:
        ang = np.full(12, rot)
        starting_configs.append((honeycomb_positions.copy(), ang))
    
    # 4. Perturbed honeycombs with random angles
    np.random.seed(42)
    for t in range(15):
        pos = honeycomb_positions + np.random.randn(12, 2) * 0.3
        ang = np.random.uniform(-30, 30, 12)
        starting_configs.append((pos, ang))
    
    # 5. Ring configuration: 6 in inner ring, 6 in outer ring
    inner_ring = np.array([
        [3, 0], [1.5, sqrt3], [-1.5, sqrt3],
        [-3, 0], [-1.5, -sqrt3], [1.5, -sqrt3]
    ])
    outer_ring = np.array([
        [4.5, 1.5*sqrt3], [0, 3*sqrt3], [-4.5, 1.5*sqrt3],
        [-4.5, -1.5*sqrt3], [0, -3*sqrt3], [4.5, -1.5*sqrt3]
    ])
    ring_pos = np.vstack([inner_ring, outer_ring])
    starting_configs.append((ring_pos, np.zeros(12)))
    
    # 6. Dense cluster
    cluster_pos = np.array([
        [0, 0], [2.8, 0], [1.4, 2.42],
        [-1.4, 2.42], [-2.8, 0], [-1.4, -2.42],
        [1.4, -2.42], [4.2, 2.42], [2.8, 4.85],
        [0, 4.85], [-2.8, 4.85], [-4.2, 2.42]
    ])
    starting_configs.append((cluster_pos, np.zeros(12)))
    
    best_params = None
    best_s = float('inf')
    
    from scipy.optimize import minimize
    
    for idx, (pos, ang) in enumerate(starting_configs):
        if time.time() > deadline - 20:
            break
        
        p = np.zeros(36)
        for i in range(12):
            p[3*i] = pos[i, 0]
            p[3*i+1] = pos[i, 1]
            p[3*i+2] = ang[i]
        
        # Compute initial smooth objective
        try:
            init_obj = _smooth_objective(p, beta=30.0)
        except Exception:
            continue
        
        # Run L-BFGS-B optimization
        if time.time() < deadline - 10:
            try:
                def obj_func(params):
                    return _smooth_objective(params, beta=50.0)
                
                res = minimize(obj_func, p, method='L-BFGS-B',
                               options={'maxiter': 500, 'ftol': 1e-10, 'gtol': 1e-8, 'maxls': 50})
                
                if res.success or res.fun < init_obj:
                    # Compute actual s from the optimized positions
                    pos_opt = np.column_stack([res.x[0::3], res.x[1::3]])
                    ang_opt = res.x[2::3]
                    s_opt = _outer_side_length_from_positions(pos_opt, ang_opt)
                    
                    # Verify validity
                    inner_hex_data = np.zeros((12, 3))
                    for i in range(12):
                        inner_hex_data[i, 0] = res.x[3*i]
                        inner_hex_data[i, 1] = res.x[3*i+1]
                        inner_hex_data[i, 2] = res.x[3*i+2]
                    outer_hex_data = np.array([0.0, 0.0, 0.0])
                    
                    if _check_validity(inner_hex_data, outer_hex_data, s_opt):
                        if s_opt < best_s:
                            best_s = s_opt
                            best_params = res.x.copy()
            except Exception:
                pass
        
        # If we found a good result, do a finer polish with higher beta
        if best_params is not None and time.time() < deadline - 5:
            try:
                def obj_func_fine(params):
                    return _smooth_objective(params, beta=200.0)
                
                res2 = minimize(obj_func_fine, best_params, method='L-BFGS-B',
                                options={'maxiter': 300, 'ftol': 1e-12, 'gtol': 1e-10, 'maxls': 30})
                
                pos_fine = np.column_stack([res2.x[0::3], res2.x[1::3]])
                ang_fine = res2.x[2::3]
                s_fine = _outer_side_length_from_positions(pos_fine, ang_fine)
                
                inner_hex_data = np.zeros((12, 3))
                for i in range(12):
                    inner_hex_data[i, 0] = res2.x[3*i]
                    inner_hex_data[i, 1] = res2.x[3*i+1]
                    inner_hex_data[i, 2] = res2.x[3*i+2]
                outer_hex_data = np.array([0.0, 0.0, 0.0])
                
                if _check_validity(inner_hex_data, outer_hex_data, s_fine):
                    if s_fine < best_s:
                        best_s = s_fine
                        best_params = res2.x.copy()
            except Exception:
                pass
    
    # Final result
    if best_params is None:
        best_params = np.zeros(36)
        for i in range(12):
            best_params[3*i] = seed_positions[i, 0]
            best_params[3*i+1] = seed_positions[i, 1]
            best_params[3*i+2] = seed_angles[i]
        best_s = _outer_side_length_from_positions(seed_positions, seed_angles)
    
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
    
    if not _check_validity(inner_hex_data, outer_hex_data, s):
        inner_hex_data = np.array([
            [0, 0, 0], [-2.5, 0, 0], [2.5, 0, 0],
            [-1.25, 2.17, 0], [1.25, 2.17, 0], [-1.25, -2.17, 0],
            [1.25, -2.17, 0], [-3.75, 2.17, 0], [3.75, 2.17, 0],
            [-3.75, -2.17, 0], [3.75, -2.17, 0], [0, -4, 0]
        ])
        outer_hex_data = np.array([0, 0, 0])
        s = 8.0
    
    return inner_hex_data, outer_hex_data, s

# EVOLVE-BLOCK-END
