# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def _hex_vertices(cx, cy, side, angle_deg):
    """Compute 6 vertices of a regular hexagon with center (cx,cy), side length, angle in degrees."""
    a = np.radians(angle_deg)
    angles = a + np.arange(6) * np.pi / 3
    return np.column_stack([
        cx + side * np.cos(angles),
        cy + side * np.sin(angles)
    ])

def _outer_side_length(positions, angles_deg):
    """
    Compute the minimum side length s of an outer hexagon (centered at origin, angle 0)
    that contains all inner hexagons with given centers and angles.
    The outer hexagon has apothem = s * sqrt(3)/2.
    For each inner hexagon vertex, compute the max projection onto the 6 outer normals.
    s = max_proj / (sqrt(3)/2)
    """
    normal_angles = np.radians([30, 90, 150, 210, 270, 330])
    normals = np.column_stack([np.cos(normal_angles), np.sin(normal_angles)])
    apothem_factor = np.sqrt(3) / 2
    
    max_proj = 0.0
    for i in range(len(positions)):
        verts = _hex_vertices(positions[i, 0], positions[i, 1], 1.0, angles_deg[i])
        projs = verts @ normals.T  # shape (6, 6)
        max_proj = max(max_proj, np.max(projs))
    
    return max_proj / apothem_factor

def _smooth_outer_side(positions, angles_deg, beta=20.0):
    """Smooth approximation of outer side length using log-sum-exp over all vertex projections."""
    normal_angles = np.radians([30, 90, 150, 210, 270, 330])
    normals = np.column_stack([np.cos(normal_angles), np.sin(normal_angles)])
    apothem_factor = np.sqrt(3) / 2
    
    all_projs = []
    for i in range(len(positions)):
        verts = _hex_vertices(positions[i, 0], positions[i, 1], 1.0, angles_deg[i])
        projs = verts @ normals.T
        all_projs.append(projs.flatten())
    
    all_projs = np.concatenate(all_projs)
    m = np.max(all_projs)
    lse = m + np.log(np.sum(np.exp(all_projs - m))) / beta
    return lse / apothem_factor

def _separation_penalty(positions, angles_deg, strength=1000.0):
    """Penalty for overlapping hexagons using SAT-based penetration depth."""
    n = len(positions)
    penalty = 0.0
    for i in range(n):
        verts_i = _hex_vertices(positions[i, 0], positions[i, 1], 1.0, angles_deg[i])
        for j in range(i + 1, n):
            verts_j = _hex_vertices(positions[j, 0], positions[j, 1], 1.0, angles_deg[j])
            
            min_gap = float('inf')
            # Check all 12 edge normals (6 from each hexagon)
            for k in range(6):
                edge = verts_i[(k + 1) % 6] - verts_i[k]
                normal = np.array([-edge[1], edge[0]])
                norm = np.linalg.norm(normal)
                if norm > 1e-12:
                    normal = normal / norm
                    proj_i = verts_i @ normal
                    proj_j = verts_j @ normal
                    gap = min(np.min(proj_j) - np.max(proj_i), np.min(proj_i) - np.max(proj_j))
                    min_gap = min(min_gap, -gap)
            
            for k in range(6):
                edge = verts_j[(k + 1) % 6] - verts_j[k]
                normal = np.array([-edge[1], edge[0]])
                norm = np.linalg.norm(normal)
                if norm > 1e-12:
                    normal = normal / norm
                    proj_i = verts_i @ normal
                    proj_j = verts_j @ normal
                    gap = min(np.min(proj_j) - np.max(proj_i), np.min(proj_i) - np.max(proj_j))
                    min_gap = min(min_gap, -gap)
            
            if min_gap < 0:
                penalty += strength * min_gap * min_gap
    
    return penalty

def _check_validity(inner_hex_data, outer_hex_data, s, tol=1e-6):
    """Check if the packing is valid."""
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
    
    # SAT check for separation
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
    
    # Containment check
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

def _objective(params):
    """Combined objective: smooth outer side + separation penalty."""
    positions = np.column_stack([params[0::3], params[1::3]])
    angles = params[2::3]
    base = _smooth_outer_side(positions, angles, beta=30.0)
    pen = _separation_penalty(positions, angles, strength=500.0)
    return base + pen

def _objective_no_penalty(params):
    """Objective without penalty, for final polish."""
    positions = np.column_stack([params[0::3], params[1::3]])
    angles = params[2::3]
    return _smooth_outer_side(positions, angles, beta=50.0)

def hexagon_packing_12():
    start_time = time.time()
    deadline = 600.0  # safety margin before 630s kill
    
    sqrt3 = np.sqrt(3)
    
    # Best known hexagonal packing: 1 center + 6 ring + 5 more
    # Standard hexagonal tiling: center at origin, 6 neighbors at distance 2, etc.
    # For 12 hexagons in a hexagonal arrangement:
    # Row 0: 3 hexagons
    # Row 1: 4 hexagons  
    # Row 2: 3 hexagons
    # Row 3: 2 hexagons
    # Total: 12
    
    # Hexagonal grid positions (side=1 hexagons, touching means center distance = 2)
    # Using axial coordinates converted to Cartesian
    hex_positions = np.array([
        [0, 0],
        [2, 0],
        [4, 0],
        [1, sqrt3],
        [3, sqrt3],
        [5, sqrt3],
        [-1, sqrt3],
        [-3, sqrt3],
        [-1, -sqrt3],
        [1, -sqrt3],
        [3, -sqrt3],
        [-2, 0]
    ])
    hex_angles = np.zeros(12)
    
    # Center the configuration
    hex_positions = hex_positions - np.mean(hex_positions, axis=0)
    
    best_s = float('inf')
    best_params = None
    
    # Generate multiple initial configurations
    configs = []
    
    # Config 1: Hexagonal arrangement
    configs.append((hex_positions.copy(), hex_angles.copy()))
    
    # Config 2: 4x3 rectangular grid
    grid_pos = np.array([
        [-3, -sqrt3], [0, -sqrt3], [3, -sqrt3], [6, -sqrt3],
        [-1.5, 0], [1.5, 0], [4.5, 0], [7.5, 0],
        [-3, sqrt3], [0, sqrt3], [3, sqrt3], [6, sqrt3]
    ])
    grid_pos = grid_pos - np.mean(grid_pos, axis=0)
    configs.append((grid_pos, np.zeros(12)))
    
    # Config 3: 3x4 rectangular grid
    grid_pos2 = np.array([
        [-4.5, -sqrt3], [-1.5, -sqrt3], [1.5, -sqrt3], [4.5, -sqrt3],
        [-3, 0], [0, 0], [3, 0], [6, 0],
        [-4.5, sqrt3], [-1.5, sqrt3], [1.5, sqrt3], [4.5, sqrt3]
    ])
    grid_pos2 = grid_pos2 - np.mean(grid_pos2, axis=0)
    configs.append((grid_pos2, np.zeros(12)))
    
    # Config 4-13: Perturbed hexagonal
    np.random.seed(42)
    for trial in range(10):
        perturbed_pos = hex_positions + np.random.randn(12, 2) * 0.5
        angles = np.random.uniform(-15, 15, 12)
        configs.append((perturbed_pos, angles))
    
    # Phase 1: L-BFGS-B optimization from each config
    for config_idx, (pos, ang) in enumerate(configs):
        if time.time() > deadline - 60:
            break
        
        params = np.zeros(36)
        for i in range(12):
            params[3 * i] = pos[i, 0]
            params[3 * i + 1] = pos[i, 1]
            params[3 * i + 2] = ang[i]
        
        try:
            result = minimize(
                _objective,
                params,
                method='L-BFGS-B',
                options={
                    'maxiter': 3000,
                    'ftol': 1e-12,
                    'gtol': 1e-8,
                    'maxfun': 10000
                }
            )
            
            p_opt = result.x
            positions = np.column_stack([p_opt[0::3], p_opt[1::3]])
            angles = p_opt[2::3]
            s_opt = _outer_side_length(positions, angles)
            
            # Check validity
            inner_test = np.zeros((12, 3))
            for i in range(12):
                inner_test[i, 0] = positions[i, 0]
                inner_test[i, 1] = positions[i, 1]
                inner_test[i, 2] = angles[i]
            
            if _check_validity(inner_test, np.array([0.0, 0.0, 0.0]), s_opt):
                if s_opt < best_s:
                    best_s = s_opt
                    best_params = p_opt.copy()
        except Exception:
            pass
    
    # Phase 2: Nelder-Mead polish from best
    if best_params is not None and time.time() < deadline - 30:
        try:
            result_final = minimize(
                _objective_no_penalty,
                best_params,
                method='Nelder-Mead',
                options={
                    'maxiter': 5000,
                    'xatol': 1e-10,
                    'fatol': 1e-12,
                    'adaptive': True
                }
            )
            
            p_opt = result_final.x
            positions = np.column_stack([p_opt[0::3], p_opt[1::3]])
            angles = p_opt[2::3]
            s_opt = _outer_side_length(positions, angles)
            
            inner_test = np.zeros((12, 3))
            for i in range(12):
                inner_test[i, 0] = positions[i, 0]
                inner_test[i, 1] = positions[i, 1]
                inner_test[i, 2] = angles[i]
            
            if _check_validity(inner_test, np.array([0.0, 0.0, 0.0]), s_opt):
                if s_opt < best_s:
                    best_s = s_opt
                    best_params = p_opt.copy()
        except Exception:
            pass
    
    # Phase 3: Additional L-BFGS-B with different penalty strength
    if best_params is not None and time.time() < deadline - 20:
        def _objective_strong_penalty(params):
            positions = np.column_stack([params[0::3], params[1::3]])
            angles = params[2::3]
            base = _smooth_outer_side(positions, angles, beta=40.0)
            pen = _separation_penalty(positions, angles, strength=2000.0)
            return base + pen
        
        try:
            result_sp = minimize(
                _objective_strong_penalty,
                best_params,
                method='L-BFGS-B',
                options={
                    'maxiter': 2000,
                    'ftol': 1e-12,
                    'gtol': 1e-8,
                    'maxfun': 8000
                }
            )
            
            p_opt = result_sp.x
            positions = np.column_stack([p_opt[0::3], p_opt[1::3]])
            angles = p_opt[2::3]
            s_opt = _outer_side_length(positions, angles)
            
            inner_test = np.zeros((12, 3))
            for i in range(12):
                inner_test[i, 0] = positions[i, 0]
                inner_test[i, 1] = positions[i, 1]
                inner_test[i, 2] = angles[i]
            
            if _check_validity(inner_test, np.array([0.0, 0.0, 0.0]), s_opt):
                if s_opt < best_s:
                    best_s = s_opt
                    best_params = p_opt.copy()
        except Exception:
            pass
    
    # Fallback if no valid solution found
    if best_params is None:
        best_params = np.zeros(36)
        for i in range(12):
            best_params[3 * i] = hex_positions[i, 0]
            best_params[3 * i + 1] = hex_positions[i, 1]
            best_params[3 * i + 2] = hex_angles[i]
        best_s = _outer_side_length(hex_positions, hex_angles)
    
    # Construct output
    inner_hex_data = np.zeros((12, 3))
    for i in range(12):
        inner_hex_data[i, 0] = best_params[3 * i]
        inner_hex_data[i, 1] = best_params[3 * i + 1]
        inner_hex_data[i, 2] = best_params[3 * i + 2]
    
    outer_hex_data = np.array([0.0, 0.0, 0.0])
    s = _outer_side_length(
        np.column_stack([best_params[0::3], best_params[1::3]]),
        best_params[2::3]
    )
    
    # Final validity check; if invalid, use simple valid fallback
    if not _check_validity(inner_hex_data, outer_hex_data, s):
        # Simple valid fallback: spread out hexagons in a larger hexagon
        inner_hex_data = np.array([
            [0, 0, 0],
            [-3, 0, 0],
            [3, 0, 0],
            [-1.5, 2.6, 0],
            [1.5, 2.6, 0],
            [-1.5, -2.6, 0],
            [1.5, -2.6, 0],
            [-4.5, 2.6, 0],
            [4.5, 2.6, 0],
            [-4.5, -2.6, 0],
            [4.5, -2.6, 0],
            [0, -5.2, 0]
        ])
        outer_hex_data = np.array([0, 0, 0])
        s = 9.0
    
    return inner_hex_data, outer_hex_data, s

# EVOLVE-BLOCK-END
