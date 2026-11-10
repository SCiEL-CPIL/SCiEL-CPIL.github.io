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
        cy + side * np.sin(a + np.arange(6) * np.pi / 3)
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

def _smooth_outer_side(positions, angles_deg, beta=50.0):
    """Smooth approximation of outer side length using log-sum-exp."""
    n = len(positions)
    normal_angles = np.radians([30, 90, 150, 210, 270, 330])
    normals = np.column_stack([np.cos(normal_angles), np.sin(normal_angles)])
    apothem_factor = np.sqrt(3) / 2
    
    all_projs = []
    for i in range(n):
        verts = _hex_vertices(positions[i, 0], positions[i, 1], 1.0, angles_deg[i])
        projs = verts @ normals.T
        all_projs.append(projs)
    
    all_projs = np.concatenate(all_projs)
    # log-sum-exp for smooth max
    m = np.max(all_projs)
    lse = m + np.log(np.sum(np.exp(all_projs - m)))
    return lse / apothem_factor

def _separation_penalty(positions, angles_deg, strength=100.0, tol=0.01):
    """Penalty for overlapping hexagons using SAT-based distance."""
    n = len(positions)
    penalty = 0.0
    for i in range(n):
        verts_i = _hex_vertices(positions[i, 0], positions[i, 1], 1.0, angles_deg[i])
        for j in range(i+1, n):
            verts_j = _hex_vertices(positions[j, 0], positions[j, 1], 1.0, angles_deg[j])
            
            min_gap = float('inf')
            # Check all 12 edge normals (6 from each)
            for k in range(6):
                edge = verts_i[(k+1)%6] - verts_i[k]
                normal = np.array([-edge[1], edge[0]])
                norm = np.linalg.norm(normal)
                if norm > 1e-12:
                    normal = normal / norm
                    proj_i = verts_i @ normal
                    proj_j = verts_j @ normal
                    gap = min(np.min(proj_j) - np.max(proj_i), np.min(proj_i) - np.max(proj_j))
                    min_gap = min(min_gap, -gap)  # negative if separated
            
            for k in range(6):
                edge = verts_j[(k+1)%6] - verts_j[k]
                normal = np.array([-edge[1], edge[0]])
                norm = np.linalg.norm(normal)
                if norm > 1e-12:
                    normal = normal / normal
                    normal = normal / norm
                    proj_i = verts_i @ normal
                    proj_j = verts_j @ normal
                    gap = min(np.min(proj_j) - np.max(proj_i), np.min(proj_i) - np.max(proj_j))
                    min_gap = min(min_gap, -gap)
            
            if min_gap < 0:  # overlapping
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

def _physics_simulate(positions, angles_deg, iterations=5000, dt=0.01):
    """
    Physics-based simulation: repulsive forces between overlapping hexagons,
    attractive force toward center to minimize enclosing hexagon.
    """
    n = len(positions)
    pos = positions.copy().astype(float)
    ang = angles_deg.copy().astype(float)
    
    # Repulsion strength and attraction to center
    repulse_strength = 50.0
    attract_strength = 0.5
    damping = 0.95
    
    forces = np.zeros_like(pos)
    
    for it in range(iterations):
        forces = np.zeros((n, 2))
        
        # Repulsive forces between pairs
        for i in range(n):
            verts_i = _hex_vertices(pos[i, 0], pos[i, 1], 1.0, ang[i])
            for j in range(i+1, n):
                verts_j = _hex_vertices(pos[j, 0], pos[j, 1], 1.0, ang[j])
                
                # Find minimum penetration depth and direction
                min_overlap = float('inf')
                push_dir = np.zeros(2)
                
                # Check all 12 normals
                for k in range(6):
                    edge = verts_i[(k+1)%6] - verts_i[k]
                    normal = np.array([-edge[1], edge[0]])
                    norm = np.linalg.norm(normal)
                    if norm > 1e-12:
                        normal = normal / norm
                        proj_i = verts_i @ normal
                        proj_j = verts_j @ normal
                        # If overlapping along this normal
                        if np.max(proj_i) > np.min(proj_j) and np.max(proj_j) > np.min(proj_i):
                            overlap = min(np.max(proj_i) - np.min(proj_j), np.max(proj_j) - np.min(proj_i))
                            if overlap < min_overlap:
                                min_overlap = overlap
                                # Push direction: away from each other
                                center_diff = pos[i] - pos[j]
                                proj_center = center_diff @ normal
                                if proj_center > 0:
                                    push_dir = normal * overlap
                                else:
                                    push_dir = -normal * overlap
                
                for k in range(6):
                    edge = verts_j[(k+1)%6] - verts_j[k]
                    normal = np.array([-edge[1], edge[0]])
                    norm = np.linalg.norm(normal)
                    if norm > 1e-12:
                        normal = normal / norm
                        proj_i = verts_i @ normal
                        proj_j = verts_j @ normal
                        if np.max(proj_i) > np.min(proj_j) and np.max(proj_j) > np.min(proj_i):
                            overlap = min(np.max(proj_i) - np.min(proj_j), np.max(proj_j) - np.min(proj_i))
                            if overlap < min_overlap:
                                min_overlap = overlap
                                center_diff = pos[i] - pos[j]
                                proj_center = center_diff @ normal
                                if proj_center > 0:
                                    push_dir = normal * overlap
                                else:
                                    push_dir = -normal * overlap
                
                if min_overlap < float('inf') and min_overlap > 0:
                    forces[i] += repulse_strength * push_dir
                    forces[j] -= repulse_strength * push_dir
        
        # Attractive force toward center (to minimize enclosing hexagon)
        for i in range(n):
            forces[i] += -attract_strength * pos[i]
        
        # Update positions
        pos += dt * forces
        forces *= damping
        
        # Anneal: reduce repulsion over time
        if it % 1000 == 0 and it > 0:
            repulse_strength *= 0.8
            attract_strength *= 1.2
    
    return pos, ang

def hexagon_packing_12():
    start_time = time.time()
    deadline = 630.0
    
    # Initial hexagonal grid arrangement
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
    
    best_s = float('inf')
    best_params = None
    
    # Phase 1: Physics-based simulation from multiple starting configs
    configs = []
    configs.append((hex_positions, hex_angles))
    
    # Perturbed starts
    for trial in range(10):
        np.random.seed(trial * 100 + 42)
        perturbed_pos = hex_positions + np.random.randn(12, 2) * 0.3
        angles = np.random.uniform(-30, 30, 12)
        configs.append((perturbed_pos, angles))
    
    # Different initial arrangements
    # 3x4 grid
    grid_pos = np.array([
        [-3, 0], [0, 0], [3, 0], [6, 0],
        [-1.5, sqrt3], [1.5, sqrt3], [4.5, sqrt3], [7.5, sqrt3],
        [-3, 2*sqrt3], [0, 2*sqrt3], [3, 2*sqrt3], [6, 2*sqrt3]
    ])
    grid_pos = grid_pos - np.mean(grid_pos, axis=0)  # Center it
    configs.append((grid_pos, np.zeros(12)))
    
    for config_idx, (pos, ang) in enumerate(configs):
        if time.time() > deadline - 120:
            break
        
        # Run physics simulation
        try:
            sim_pos, sim_ang = _physics_simulate(pos, ang, iterations=3000, dt=0.01)
            
            # Evaluate
            s_val = _outer_side_length_from_positions(sim_pos, sim_ang)
            
            if s_val < best_s:
                best_s = s_val
                best_params = np.zeros(36)
                for i in range(12):
                    best_params[3*i] = sim_pos[i, 0]
                    best_params[3*i+1] = sim_pos[i, 1]
                    best_params[3*i+2] = sim_ang[i]
        except Exception as e:
            pass
    
    # Phase 2: Gradient-based optimization with smooth objective
    if best_params is not None and time.time() < deadline - 60:
        def objective(params):
            positions = np.column_stack([params[0::3], params[1::3]])
            angles = params[2::3]
            return _smooth_outer_side(positions, angles, beta=30.0)
        
        def penalty_objective(params):
            positions = np.column_stack([params[0::3], params[1::3]])
            angles = params[2::3]
            base = _smooth_outer_side(positions, angles, beta=30.0)
            pen = _separation_penalty(positions, angles, strength=50.0)
            return base + pen
        
        # Try both with and without penalty
        for obj_func in [penalty_objective, objective]:
            try:
                result = minimize(
                    obj_func,
                    best_params,
                    method='L-BFGS-B',
                    options={
                        'maxiter': 2000,
                        'ftol': 1e-14,
                        'gtol': 1e-10,
                        'maxfun': 5000
                    }
                )
                
                p_opt = result.x
                positions = np.column_stack([p_opt[0::3], p_opt[1::3]])
                angles = p_opt[2::3]
                s_opt = _outer_side_length_from_positions(positions, angles)
                
                # Check validity before accepting
                inner_test = np.zeros((12, 3))
                for i in range(12):
                    inner_test[i, 0] = positions[i, 0]
                    inner_test[i, 1] = positions[i, 1]
                    inner_test[i, 2] = angles[i]
                
                if _check_validity(inner_test, np.array([0.0, 0.0, 0.0]), s_opt):
                    if s_opt < best_s:
                        best_s = s_opt
                        best_params = p_opt.copy()
            except Exception as e:
                pass
    
    # Phase 3: Final Nelder-Mead polish
    if best_params is not None and time.time() < deadline - 30:
        def objective(params):
            positions = np.column_stack([params[0::3], params[1::3]])
            angles = params[2::3]
            return _outer_side_length_from_positions(positions, angles)
        
        try:
            result_final = minimize(
                objective,
                best_params,
                method='Nelder-Mead',
                options={
                    'maxiter': 3000,
                    'xatol': 1e-10,
                    'fatol': 1e-12,
                    'adaptive': True
                }
            )
            
            p_opt = result_final.x
            positions = np.column_stack([p_opt[0::3], p_opt[1::3]])
            angles = p_opt[2::3]
            s_opt = _outer_side_length_from_positions(positions, angles)
            
            inner_test = np.zeros((12, 3))
            for i in range(12):
                inner_test[i, 0] = positions[i, 0]
                inner_test[i, 1] = positions[i, 1]
                inner_test[i, 2] = angles[i]
            
            if _check_validity(inner_test, np.array([0.0, 0.0, 0.0]), s_opt):
                if s_opt < best_s:
                    best_s = s_opt
                    best_params = p_opt.copy()
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
        # Try a simple valid fallback
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
