import numpy as np
import time
from scipy.optimize import minimize

def _hex_vertices(cx, cy, d, a):
    a_rad = np.deg2rad(a)
    angles = a_rad + np.arange(6) * np.pi / 3
    return np.column_stack([cx + d * np.cos(angles), cy + d * np.sin(angles)])

def _min_outer_side(positions, angle_deg):
    """
    Compute the minimum side length of a regular hexagon with the given angle
    that encloses all unit hexagons at the given positions.
    """
    unit_hex_verts = _hex_vertices(0, 0, 1, 0)
    all_verts = positions[:, np.newaxis, :] + unit_hex_verts[np.newaxis, :, :]
    all_verts_flat = all_verts.reshape(-1, 2)
    
    a_rad = np.deg2rad(angle_deg)
    normals = []
    for i in range(6):
        n_angle = a_rad + i * np.pi/3 + np.pi/6
        n_unit = np.array([np.cos(n_angle), np.sin(n_angle)])
        normals.append(n_unit)
    normals = np.array(normals)
    
    projections = all_verts_flat @ normals.T
    max_projections = np.max(projections, axis=0)
    min_s = np.max(max_projections) * 2 / np.sqrt(3)
    return min_s

def _check_overlap_fast(pos):
    """
    Fast check for overlaps using center distance.
    Returns True if any pair is definitely overlapping (distance < sqrt(3) - eps).
    Returns False if no pair is definitely overlapping.
    Note: Some pairs might be in the "gray zone" (sqrt(3) - eps < d < sqrt(3) + eps) 
    where SAT is needed for precision, but for optimization penalties, 
    treating d < sqrt(3) as overlap is a safe conservative constraint 
    that keeps the search away from the boundary.
    """
    n = len(pos)
    min_dist_sq = 3.0 - 1e-6 # (sqrt(3) - eps)^2 approx
    for i in range(n):
        for j in range(i+1, n):
            d_sq = np.sum((pos[i] - pos[j])**2)
            if d_sq < min_dist_sq:
                return True
    return False

def _objective(x):
    """
    Objective function for Nelder-Mead.
    x: [x1, y1, ..., x12, y12, angle]
    """
    pos = x[:24].reshape(12, 2)
    angle = x[24]
    
    # Check overlaps
    if _check_overlap_fast(pos):
        # Penalize heavily
        # Calculate how much we are violating
        penalty = 0
        min_dist_sq = 3.0
        for i in range(12):
            for j in range(i+1, 12):
                d_sq = np.sum((pos[i] - pos[j])**2)
                if d_sq < min_dist_sq:
                    # Penalty proportional to violation
                    penalty += (min_dist_sq - d_sq)
        # Return large value
        return 1000.0 + penalty * 10
    
    # Compute s
    s = _min_outer_side(pos, angle)
    return s

def _refine_solution(pos, angle):
    """
    Run Nelder-Mead to minimize s starting from pos, angle.
    """
    x0 = np.concatenate([pos.flatten(), [angle]])
    
    # Bounds for angle
    # Nelder-Mead doesn't support bounds directly in older scipy, 
    # but we can wrap angle or just let it run and normalize.
    # Since hex symmetry is 60 deg, we can restrict angle to [0, 60).
    
    result = minimize(_objective, x0, method='Nelder-Mead', 
                      options={'maxiter': 2000, 'xatol': 1e-6, 'fatol': 1e-6})
    
    best_x = result.x
    best_pos = best_x[:24].reshape(12, 2)
    best_angle = best_x[24] % 60.0 # Normalize angle
    
    # Final check and repair if needed
    if _check_overlap_fast(best_pos):
        # If still overlapping, try to push apart
        for _ in range(100):
            changed = False
            for i in range(12):
                for j in range(i+1, 12):
                    d_vec = best_pos[j] - best_pos[i]
                    d = np.linalg.norm(d_vec)
                    if d < np.sqrt(3) - 1e-5:
                        if d < 1e-8:
                            d_vec = np.array([1.0, 0.0])
                        else:
                            d_vec = d_vec / d
                        # Push apart
                        target_d = np.sqrt(3)
                        push = (target_d - d) / 2.0
                        best_pos[i] -= d_vec * push
                        best_pos[j] += d_vec * push
                        changed = True
            if not changed:
                break
        
        # Re-optimize after push
        x0 = np.concatenate([best_pos.flatten(), [best_angle]])
        result = minimize(_objective, x0, method='Nelder-Mead', 
                          options={'maxiter': 1000, 'xatol': 1e-6, 'fatol': 1e-6})
        best_x = result.x
        best_pos = best_x[:24].reshape(12, 2)
        best_angle = best_x[24] % 60.0
        
    best_s = _min_outer_side(best_pos, best_angle)
    return best_pos, best_angle, best_s

def hexagon_packing_12():
    start_time = time.time()
    deadline = start_time + 590 # Safety margin
    
    best_score = 0.0
    best_inner = None
    best_outer = None
    best_s = None
    
    # Generate diverse initial configurations
    # 1. Standard hexagonal lattice cluster
    # 2. Random loose cluster
    
    h = np.sqrt(3)/2
    
    # Config 1: 1 center, 6 ring, 5 more
    # Centers:
    # (0,0)
    # (sqrt(3), 0), (sqrt(3)/2, 1.5), (-sqrt(3)/2, 1.5), (-sqrt(3), 0), (-sqrt(3)/2, -1.5), (sqrt(3)/2, -1.5)
    # Next ring would be distance 2*sqrt(3)
    # We have 12 hexes. 1+6=7. Need 5 more.
    # Place them at the "gaps" of the next ring?
    # Or just start with a 3x4 grid and let optimizer fix it.
    
    initial_configs = []
    
    # Grid 3x4
    grid_pos = []
    for i in range(3):
        for j in range(4):
            x = j * 2 * h
            y = i * 1.5
            if i % 2 == 1:
                x += h
            grid_pos.append([x, y])
    initial_configs.append(np.array(grid_pos))
    
    # Hexagonal cluster 1+6+5 (approx)
    hex_pos = [
        [0, 0],
        [2*h, 0],
        [h, 1.5],
        [-h, 1.5],
        [-2*h, 0],
        [-h, -1.5],
        [h, -1.5],
        # 5 more
        [3*h, 1.5],
        [0, 3],
        [-3*h, 1.5],
        [-3*h, -1.5],
        [3*h, -1.5]
    ]
    initial_configs.append(np.array(hex_pos))
    
    # Random loose
    random_pos = np.random.randn(12, 2) * 2.0
    initial_configs.append(random_pos)
    
    num_restarts = 10
    
    for r in range(num_restarts):
        if time.time() > deadline:
            break
            
        # Select initial config
        if r < len(initial_configs):
            pos = initial_configs[r].copy()
        else:
            pos = initial_configs[r % len(initial_configs)] + np.random.randn(12, 2) * 0.5
            
        # Add small noise to break symmetry
        pos += np.random.randn(12, 2) * 0.01
        
        angle = np.random.uniform(0, 60)
        
        # Run refinement
        try:
            best_pos, best_angle, best_s = _refine_solution(pos, angle)
            
            # Construct final arrays
            inner_candidate = np.zeros((12, 3))
            inner_candidate[:, :2] = best_pos
            inner_candidate[:, 2] = 0.0 # Inner hexes aligned with 0 angle for simplicity in SAT?
            # Wait, the task allows inner hexes to have any angle.
            # However, for optimal packing of regular hexagons, they should be aligned 
            # to the lattice or to each other. 
            # If we allow them to rotate, the search space is larger.
            # Given the time budget, fixing inner angles to 0 is a strong heuristic 
            # that likely holds for the global optimum (hexagons tile best when aligned).
            
            outer_candidate = np.array([0, 0, best_angle])
            
            # Final validity check
            if _check_validity(inner_candidate, outer_candidate, best_s):
                score_candidate = 1.0 / best_s
                if score_candidate > best_score:
                    best_score = score_candidate
                    best_inner = inner_candidate
                    best_outer = outer_candidate
                    best_s = best_s
        except Exception as e:
            continue
            
    if best_inner is not None:
        return best_inner, best_outer, best_s
    else:
        # Fallback
        inner_fallback = np.array([
            [0, 0, 0],
            [2.5, 0, 0],
            [-2.5, 0, 0],
            [1.25, 2.165, 0],
            [-1.25, 2.165, 0],
            [1.25, -2.165, 0],
            [-1.25, -2.165, 0],
            [3.75, 2.165, 0],
            [-3.75, 2.165, 0],
            [3.75, -2.165, 0],
            [-3.75, -2.165, 0],
            [0, 4.33, 0]
        ])
        outer_fallback = np.array([0, 0, 0])
        s_fallback = _min_outer_side(inner_fallback[:, :2], 0)
        if s_fallback < np.sqrt(12):
            s_fallback = np.sqrt(12)
        return inner_fallback, outer_fallback, s_fallback

def _check_validity(inner, outer, s):
    """
    Validates the solution according to the task contract.
    """
    if not np.all(np.isfinite(inner)) or not np.all(np.isfinite(outer)) or not np.isfinite(s):
        return False
    if s < np.sqrt(12) - 1e-9:
        return False
        
    outer_verts = _hex_vertices(outer[0], outer[1], s, outer[2])
    for i in range(6):
        for j in range(i+1, 6):
            if np.linalg.norm(outer_verts[i] - outer_verts[j]) < 1e-9:
                return False
                
    for i in range(12):
        cx, cy, ang = inner[i]
        verts = _hex_vertices(cx, cy, 1, ang)
        for k in range(6):
            v1 = verts[k]
            v2 = verts[(k+1)%6]
            if abs(np.linalg.norm(v2 - v1) - 1.0) > 1e-8:
                return False

    inner_verts_list = []
    for i in range(12):
        cx, cy, ang = inner[i]
        inner_verts_list.append(_hex_vertices(cx, cy, 1, ang))
        
    for i in range(12):
        for j in range(i+1, 12):
            va = inner_verts_list[i]
            vb = inner_verts_list[j]
            separated = False
            for k in range(6):
                v1 = va[k]
                v2 = va[(k+1)%6]
                e = v2 - v1
                n = np.array([-e[1], e[0]])
                norm = np.linalg.norm(n)
                if norm < 1e-12: continue
                n = n / norm
                proj_a = np.dot(va, n)
                proj_b = np.dot(vb, n)
                if np.max(proj_a) < np.min(proj_b) - 1e-6 or np.max(proj_b) < np.min(proj_a) - 1e-6:
                    separated = True
                    break
            if separated: continue
            for k in range(6):
                v1 = vb[k]
                v2 = vb[(k+1)%6]
                e = v2 - v1
                n = np.array([-e[1], e[0]])
                norm = np.linalg.norm(n)
                if norm < 1e-12: continue
                n = n / norm
                proj_a = np.dot(va, n)
                proj_b = np.dot(vb, n)
                if np.max(proj_a) < np.min(proj_b) - 1e-6 or np.max(proj_b) < np.min(proj_a) - 1e-6:
                    separated = True
                    break
            if not separated:
                return False
                
    for i in range(12):
        for v in inner_verts_list[i]:
            inside = True
            for k in range(6):
                p1 = outer_verts[k]
                p2 = outer_verts[(k+1)%6]
                e = p2 - p1
                cross_val = e[0] * (v[1] - p1[1]) - e[1] * (v[0] - p1[0])
                if cross_val < -1e-6:
                    inside = False
                    break
            if not inside:
                return False
                
    return True

# EVOLVE-BLOCK-END
