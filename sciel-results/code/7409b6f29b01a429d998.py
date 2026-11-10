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

def hexagon_packing_12():
    start_time = time.time()
    deadline = 600.0
    
    # Seed arrangement
    seed_positions = np.array([
        [0, 0],
        [-2.5, 0],
        [2.5, 0],
        [-1.25, 2.17],
        [1.25, 2.17],
        [-1.25, -2.17],
        [1.25, -2.17],
        [-3.75, 2.17],
        [3.75, 2.17],
        [-3.75, -2.17],
        [3.75, -2.17],
        [0, -4]
    ])
    seed_angles = np.zeros(12)
    
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
    
    # Config 0: Seed
    configs.append((seed_positions, seed_angles))
    
    # Config 1: Hexagonal packing
    configs.append((hex_positions, hex_angles))
    
    # Config 2: Hexagonal packing with rotations
    for rot in [15, 30, 45]:
        angles = np.zeros(12)
        for i in range(12):
            angles[i] = rot
        configs.append((hex_positions.copy(), angles))
    
    # Config 3: Perturbed hexagonal packing
    for trial in range(20):
        np.random.seed(trial * 100 + 42)
        perturbed_pos = hex_positions + np.random.randn(12, 2) * 0.1
        angles = np.random.uniform(-30, 30, 12)
        configs.append((perturbed_pos, angles))
    
    # Config 4: Different arrangement - 3 rows
    # Row 0: 4 hexagons
    # Row 1: 4 hexagons
    # Row 2: 4 hexagons
    # Actually, let's try a more compact arrangement
    
    # Try to find a better arrangement using random search
    for trial in range(50):
        if time.time() > deadline - 60:
            break
        np.random.seed(trial * 1000 + 7)
        
        # Random positions in a hexagonal region
        pos = np.zeros((12, 2))
        for i in range(12):
            # Place in a hexagonal grid pattern with some randomness
            r = np.sqrt(np.random.uniform(0, 1)) * 4
            theta = np.random.uniform(0, 2*np.pi)
            pos[i] = [r * np.cos(theta), r * np.sin(theta)]
        
        angles = np.random.uniform(-30, 30, 12)
        configs.append((pos, angles))
    
    best_s = float('inf')
    best_params = None
    
    for config_idx, (pos, ang) in enumerate(configs):
        if time.time() > deadline - 30:
            break
        
        # Pack into params vector
        p = np.zeros(36)
        for i in range(12):
            p[3*i] = pos[i, 0]
            p[3*i+1] = pos[i, 1]
            p[3*i+2] = ang[i]
        
        # Initial evaluation
        s_val = _outer_side_length_from_positions(pos, ang)
        
        # Use scipy minimize
        def objective(params):
            positions = np.column_stack([params[0::3], params[1::3]])
            angles = params[2::3]
            return _outer_side_length_from_positions(positions, angles)
        
        try:
            result = minimize(
                objective,
                p,
                method='Nelder-Mead',
                options={
                    'maxiter': 2000,
                    'xatol': 1e-8,
                    'fatol': 1e-10,
                    'adaptive': True
                }
            )
            
            if result.success or result.fun < s_val:
                p_opt = result.x
                s_opt = result.fun
                
                if s_opt < best_s:
                    best_s = s_opt
                    best_params = p_opt.copy()
        except Exception as e:
            pass
        
        # Also try L-BFGS-B
        try:
            result2 = minimize(
                objective,
                p,
                method='L-BFGS-B',
                options={
                    'maxiter': 1000,
                    'ftol': 1e-12,
                    'gtol': 1e-8
                }
            )
            
            if result2.success or result2.fun < s_val:
                p_opt2 = result2.x
                s_opt2 = result2.fun
                
                if s_opt2 < best_s:
                    best_s = s_opt2
                    best_params = p_opt2.copy()
        except Exception as e:
            pass
    
    # If no optimization succeeded, use the best initial config
    if best_params is None:
        # Use the hexagonal packing as fallback
        best_params = np.zeros(36)
        for i in range(12):
            best_params[3*i] = hex_positions[i, 0]
            best_params[3*i+1] = hex_positions[i, 1]
            best_params[3*i+2] = hex_angles[i]
        best_s = _outer_side_length_from_positions(hex_positions, hex_angles)
    
    # Final refinement with more iterations
    if time.time() < deadline - 10:
        try:
            def objective(params):
                positions = np.column_stack([params[0::3], params[1::3]])
                angles = params[2::3]
                return _outer_side_length_from_positions(positions, angles)
            
            result_final = minimize(
                objective,
                best_params,
                method='Nelder-Mead',
                options={
                    'maxiter': 5000,
                    'xatol': 1e-10,
                    'fatol': 1e-12,
                    'adaptive': True
                }
            )
            
            if result_final.fun < best_s:
                best_s = result_final.fun
                best_params = result_final.x.copy()
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
