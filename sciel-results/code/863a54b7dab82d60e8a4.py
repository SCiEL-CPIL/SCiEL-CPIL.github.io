# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def _hex_vertices(cx, cy, d, angle_deg):
    """Compute 6 vertices of a regular hexagon with center (cx,cy), side d, angle in degrees."""
    a = np.radians(angle_deg)
    angles = a + np.radians(60) * np.arange(6)
    return np.column_stack([cx + d * np.cos(angles), cy + d * np.sin(angles)])

def _compute_min_outer_side(inner, outer_center, outer_angle):
    """Compute the minimum side length of outer hexagon that contains all inner hexagons."""
    shifted_centers = inner[:, :2] - outer_center
    
    all_verts = []
    for i in range(12):
        verts = _hex_vertices(shifted_centers[i, 0], shifted_centers[i, 1], 1.0, inner[i, 2])
        all_verts.append(verts)
    all_verts = np.vstack(all_verts)
    
    a_rad = np.radians(outer_angle)
    unit_outer_verts = np.column_stack([
        np.cos(a_rad + np.radians(60) * np.arange(6)),
        np.sin(a_rad + np.radians(60) * np.arange(6))
    ])
    
    max_proj = 0
    for k in range(6):
        p1 = unit_outer_verts[k]
        p2 = unit_outer_verts[(k+1)%6]
        edge = p2 - p1
        normal = np.array([edge[1], -edge[0]])
        normal = normal / (np.linalg.norm(normal) + 1e-15)
        
        mid = (p1 + p2) / 2
        if np.dot(normal, mid) < 0:
            normal = -normal
            
        projs = all_verts @ normal
        max_k = np.max(projs)
        if max_k > max_proj:
            max_proj = max_k
    
    min_s = max_proj * 2.0 / np.sqrt(3)
    return max(min_s, np.sqrt(12))

def _is_valid_fast(inner, outer_center, outer_angle, s):
    """
    Fast validity check.
    """
    if not np.all(np.isfinite(inner)) or not np.all(np.isfinite(outer_center)) or not np.isfinite(s):
        return False
    if s < np.sqrt(12) - 1e-9:
        return False
    
    outer_verts = _hex_vertices(outer_center[0], outer_center[1], s, outer_angle)
    
    # Pre-compute outer edge normals and offsets for point-in-polygon test
    outer_norms = np.zeros((6, 2))
    outer_offsets = np.zeros(6)
    for k in range(6):
        p1 = outer_verts[k]
        p2 = outer_verts[(k+1)%6]
        edge = p2 - p1
        # Inward normal
        n = np.array([edge[1], -edge[0]])
        norm_n = np.linalg.norm(n)
        if norm_n < 1e-9: return False
        n = n / norm_n
        outer_norms[k] = n
        outer_offsets[k] = np.dot(n, p1)

    # Check each inner hexagon
    for i in range(12):
        cx, cy, ang = inner[i]
        verts = _hex_vertices(cx, cy, 1.0, ang)
        
        # Check if all vertices are inside outer hexagon
        for v in verts:
            for k in range(6):
                # cross(p2-p1, v-p1) >= -1e-6
                p1 = outer_verts[k]
                p2 = outer_verts[(k+1)%6]
                cross = (p2[0] - p1[0]) * (v[1] - p1[1]) - (p2[1] - p1[1]) * (v[0] - p1[0])
                if cross < -1e-6:
                    return False

        # Check overlap with other inner hexagons
        for j in range(i+1, 12):
            cx2, cy2, ang2 = inner[j]
            verts2 = _hex_vertices(cx2, cy2, 1.0, ang2)
            
            # Quick distance check: if centers are far apart, skip SAT
            dist = np.linalg.norm(inner[i, :2] - inner[j, :2])
            if dist > 3.5:
                continue

            # Separating Axis Test
            normals = []
            for v in [verts, verts2]:
                for k in range(6):
                    p1 = v[k]
                    p2 = v[(k+1)%6]
                    edge = p2 - p1
                    normal = np.array([-edge[1], edge[0]])
                    norm_n = np.linalg.norm(normal)
                    if norm_n < 1e-9: continue
                    normal = normal / norm_n
                    normals.append(normal)
            
            overlap = True
            for normal in normals:
                proj_i = np.dot(verts, normal)
                proj_j = np.dot(verts2, normal)
                min_i, max_i = np.min(proj_i), np.max(proj_i)
                min_j, max_j = np.min(proj_j), np.max(proj_j)
                
                if max_i < min_j - 1e-6 or max_j < min_i - 1e-6:
                    overlap = False
                    break
            
            if overlap:
                return False

    return True

def _generate_rigid_configurations():
    """
    Generate a list of rigid configurations (positions, angles) for 12 hexagons.
    """
    configs = []
    
    # Config 1: 3x4 Grid on Triangular Lattice
    L = np.sqrt(3)
    points = []
    for r in range(3):
        for q in range(4):
            x = (q + r/2.0) * L
            y = r * L * np.sqrt(3)/2
            points.append([x, y, 0.0])
    centers = np.array([p[:2] for p in points])
    center = np.mean(centers, axis=0)
    for p in points:
        p[0] -= center[0]
        p[1] -= center[1]
    configs.append(np.array(points))
    
    # Config 2: 4x3 Grid
    points = []
    for r in range(4):
        for q in range(3):
            x = (q + r/2.0) * L
            y = r * L * np.sqrt(3)/2
            points.append([x, y, 0.0])
    centers = np.array([p[:2] for p in points])
    center = np.mean(centers, axis=0)
    for p in points:
        p[0] -= center[0]
        p[1] -= center[1]
    configs.append(np.array(points))
    
    # Config 3: Hexagonal Cluster (7) + 5 more
    axial = [(0,0), (1,0), (1,-1), (0,-1), (-1,0), (-1,1), (0,1), 
             (2,0), (2,-1), (1,-2), (0,-2), (-1,-2)]
    points = []
    for q, r in axial:
        x = (q + r/2.0) * L
        y = r * L * np.sqrt(3)/2
        points.append([x, y, 0.0])
    centers = np.array([p[:2] for p in points])
    center = np.mean(centers, axis=0)
    for p in points:
        p[0] -= center[0]
        p[1] -= center[1]
    configs.append(np.array(points))
    
    # Config 4: Original seed from current program
    initial_inner = np.array(
        [
            [0, 0, 0],
            [-2.5, 0, 0],
            [2.5, 0, 0],
            [-1.25, 2.1650635094610965, 0],
            [1.25, 2.1650635094610965, 0],
            [-1.25, -2.1650635094610965, 0],
            [1.25, -2.1650635094610965, 0],
            [-3.75, 2.1650635094610965, 0],
            [3.75, 2.1650635094610965, 0],
            [-3.75, -2.1650635094610965, 0],
            [3.75, -2.1650635094610965, 0],
            [0, -4.330127018922193, 0],
        ]
    )
    # Center it
    c = np.mean(initial_inner[:, :2], axis=0)
    initial_inner[:, :2] -= c
    configs.append(initial_inner)
    
    return configs

def hexagon_packing_12():
    start_time = time.time()
    deadline = start_time + 600.0
    
    configs = _generate_rigid_configurations()
    
    best_inner = None
    best_s = float('inf')
    best_outer_angle = 0.0
    
    # Optimize each rigid configuration
    for config in configs:
        # Define objective for rigid body rotation
        def rigid_objective(params):
            alpha = params[0] # Cluster rotation
            beta = params[1]  # Outer hexagon rotation
            
            # Rotate cluster positions
            c, s_ = np.cos(alpha), np.sin(alpha)
            rot_mat = np.array([[c, -s_], [s_, c]])
            
            # Rotate angles
            new_angles = config[:, 2] + np.degrees(alpha)
            
            # Rotate positions
            new_positions = config[:, :2] @ rot_mat.T
            
            new_inner = np.hstack([new_positions, new_angles.reshape(-1, 1)])
            
            # Compute min s
            s = _compute_min_outer_side(new_inner, np.array([0.0, 0.0]), np.degrees(beta))
            return s

        # Try multiple starting points for alpha, beta
        for alpha0 in [0, np.pi/6, np.pi/3]:
            for beta0 in [0, np.pi/6, np.pi/3]:
                x0 = np.array([alpha0, beta0])
                try:
                    result = minimize(rigid_objective, x0, method='Nelder-Mead', 
                                      options={'maxiter': 200, 'xatol': 1e-6, 'fatol': 1e-8})
                    
                    if result.success or result.fun < best_s:
                        alpha_opt, beta_opt = result.x
                        c, s_ = np.cos(alpha_opt), np.sin(alpha_opt)
                        rot_mat = np.array([[c, -s_], [s_, c]])
                        new_positions = config[:, :2] @ rot_mat.T
                        new_angles = config[:, 2] + np.degrees(alpha_opt)
                        candidate_inner = np.hstack([new_positions, new_angles.reshape(-1, 1)])
                        
                        candidate_s = result.fun
                        candidate_outer_angle = np.degrees(beta_opt)
                        
                        # Validate
                        if _is_valid_fast(candidate_inner, np.array([0.0, 0.0]), candidate_outer_angle, candidate_s):
                            if candidate_s < best_s:
                                best_s = candidate_s
                                best_inner = candidate_inner
                                best_outer_angle = candidate_outer_angle
                except Exception:
                    pass
                    
        if time.time() > deadline - 30.0:
            break

    # If we found a good rigid solution, try a small flexible refinement
    if best_inner is not None and time.time() < deadline - 60.0:
        current_inner = best_inner.copy()
        current_s = best_s
        current_outer_angle = best_outer_angle
        
        def flexible_objective(params):
            inner = params[:36].reshape(12, 3)
            outer_angle = np.degrees(params[36])
            
            s = _compute_min_outer_side(inner, np.array([0.0, 0.0]), outer_angle)
            if not _is_valid_fast(inner, np.array([0.0, 0.0]), outer_angle, s):
                return s + 10.0
            return s

        x0 = np.concatenate([current_inner.flatten(), [np.radians(current_outer_angle)]])
        
        try:
            result = minimize(flexible_objective, x0, method='Nelder-Mead',
                              options={'maxiter': 100, 'xatol': 1e-5, 'fatol': 1e-7})
            if result.fun < best_s:
                candidate_inner = result.x[:36].reshape(12, 3)
                candidate_outer_angle = np.degrees(result.x[36])
                candidate_s = result.fun
                if _is_valid_fast(candidate_inner, np.array([0.0, 0.0]), candidate_outer_angle, candidate_s):
                    best_inner = candidate_inner
                    best_s = candidate_s
                    best_outer_angle = candidate_outer_angle
        except Exception:
            pass

    # Fallback
    if best_inner is None:
        best_inner = configs[0]
        best_s = _compute_min_outer_side(best_inner, np.array([0.0, 0.0]), 0)
        best_outer_angle = 0

    outer_hex_data = np.array([0.0, 0.0, best_outer_angle])
    return best_inner, outer_hex_data, float(best_s)

# EVOLVE-BLOCK-END
