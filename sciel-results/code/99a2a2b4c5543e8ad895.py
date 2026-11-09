# EVOLVE-BLOCK-START
import numpy as np
import time


def _hex_vertices(cx, cy, d, angle_deg):
    """Compute 6 vertices of a regular hexagon with center (cx,cy), side d, angle in degrees."""
    a = np.radians(angle_deg)
    angles = a + np.radians(60) * np.arange(6)
    return np.column_stack([cx + d * np.cos(angles), cy + d * np.sin(angles)])


def _is_valid(inner, outer, s):
    """Check validity of the packing configuration."""
    inner = np.array(inner, dtype=float)
    outer = np.array(outer, dtype=float)
    s = float(s)
    
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
        verts = _hex_vertices(inner[i, 0], inner[i, 1], 1.0, inner[i, 2])
        for j in range(6):
            edge_len = np.linalg.norm(verts[j] - verts[(j+1)%6])
            if abs(edge_len - 1.0) > 1e-8:
                return False
    
    for i in range(12):
        for j in range(i+1, 12):
            dist = np.linalg.norm(inner[i, :2] - inner[j, :2])
            if dist > 2.0 + 1e-9:
                continue
            
            verts_i = _hex_vertices(inner[i, 0], inner[i, 1], 1.0, inner[i, 2])
            verts_j = _hex_vertices(inner[j, 0], inner[j, 1], 1.0, inner[j, 2])
            
            normals = []
            for v in [verts_i, verts_j]:
                for k in range(6):
                    p1 = v[k]
                    p2 = v[(k+1)%6]
                    edge = p2 - p1
                    normal = np.array([-edge[1], edge[0]])
                    norm = np.linalg.norm(normal)
                    if norm > 1e-15:
                        normal = normal / norm
                        normals.append(normal)
            
            overlap = True
            for normal in normals:
                proj_i = np.dot(verts_i, normal)
                proj_j = np.dot(verts_j, normal)
                min_i, max_i = np.min(proj_i), np.max(proj_i)
                min_j, max_j = np.min(proj_j), np.max(proj_j)
                
                if max_i < min_j - 1e-6 or max_j < min_i - 1e-6:
                    overlap = False
                    break
            
            if overlap:
                return False
    
    for i in range(12):
        verts_i = _hex_vertices(inner[i, 0], inner[i, 1], 1.0, inner[i, 2])
        for v in verts_i:
            inside = True
            for k in range(6):
                p1 = outer_verts[k]
                p2 = outer_verts[(k+1)%6]
                cross = (p2[0] - p1[0]) * (v[1] - p1[1]) - (p2[1] - p1[1]) * (v[0] - p1[0])
                if cross < -1e-6:
                    inside = False
                    break
            if not inside:
                return False
    
    return True


def _compute_min_outer_side(inner, outer_center, outer_angle):
    """Compute the minimum side length of outer hexagon that contains all inner hexagons."""
    inner = np.array(inner, dtype=float)
    outer_center = np.array(outer_center, dtype=float)
    
    shifted_centers = inner[:, :2] - outer_center
    
    angles_deg = inner[:, 2]
    a_rad = np.radians(angles_deg)
    dir_cos = np.cos(a_rad[:, None] + np.radians(60) * np.arange(6)[None, :])
    dir_sin = np.sin(a_rad[:, None] + np.radians(60) * np.arange(6)[None, :])
    
    all_verts = np.zeros((72, 2))
    for i in range(12):
        all_verts[6*i:6*i+6] = shifted_centers[i] + np.column_stack([dir_cos[i], dir_sin[i]])
    
    a_outer = np.radians(outer_angle)
    unit_outer_verts = np.column_stack([
        np.cos(a_outer + np.radians(60) * np.arange(6)),
        np.sin(a_outer + np.radians(60) * np.arange(6))
    ])
    
    max_proj = 0.0
    for k in range(6):
        p1 = unit_outer_verts[k]
        p2 = unit_outer_verts[(k+1)%6]
        edge = p2 - p1
        normal = np.array([edge[1], -edge[0]])
        norm = np.linalg.norm(normal)
        if norm > 1e-15:
            normal = normal / norm
            projs = all_verts @ normal
            max_k = np.max(projs)
            if max_k > max_proj:
                max_proj = max_k
    
    min_s = max_proj * 2.0 / np.sqrt(3)
    return max(min_s, np.sqrt(12))


def _fast_overlap_check(inner):
    """Fast check: returns True if any pair of inner hexagons likely overlap (distance < 2)."""
    for i in range(12):
        for j in range(i+1, 12):
            if np.linalg.norm(inner[i, :2] - inner[j, :2]) < 2.0 - 1e-9:
                return True
    return False


def _overlap_penalty(inner):
    """Compute penalty for overlapping hexagons."""
    penalty = 0.0
    for i in range(12):
        for j in range(i+1, 12):
            dist = np.linalg.norm(inner[i, :2] - inner[j, :2])
            if dist < 2.0:
                penalty += (2.0 - dist) ** 2
    return penalty


def _smooth_overlap_penalty(inner, sigma=0.5):
    """Smooth approximation of overlap penalty using soft-max of distance violations."""
    penalty = 0.0
    for i in range(12):
        for j in range(i+1, 12):
            dist = np.linalg.norm(inner[i, :2] - inner[j, :2])
            violation = 2.0 - dist
            if violation > 0:
                penalty += violation**2
    return penalty


def hexagon_packing_12():
    """
    Constructs a packing of 12 disjoint unit regular hexagons inside a larger regular hexagon,
    maximizing 1/outer_hex_side_length.
    
    Strategy:
    1. Start from a good honeycomb-like initial arrangement
    2. Use Simulated Annealing with fast overlap rejection
    3. Polish with Nelder-Mead using a smooth objective
    4. Final high-precision polish pass
    """
    start_time = time.time()
    deadline = start_time + 580.0  # safety margin before 630s kill
    
    sqrt3 = np.sqrt(3)
    
    # Initial honeycomb arrangement: 7 hexagons in a ring + center + 4 more
    # This is a 4-3-3-2 honeycomb slice pattern
    initial_inner = np.array([
        [0.0, 0.0, 0.0],
        [2.5, 0.0, 0.0],
        [-2.5, 0.0, 0.0],
        [1.25, 2.1650635094610965, 0.0],
        [-1.25, 2.1650635094610965, 0.0],
        [1.25, -2.1650635094610965, 0.0],
        [-1.25, -2.1650635094610965, 0.0],
        [3.75, 2.1650635094610965, 0.0],
        [-3.75, 2.1650635094610965, 0.0],
        [3.75, -2.1650635094610965, 0.0],
        [-3.75, -2.1650635094610965, 0.0],
        [0.0, -4.330127018922193, 0.0],
    ])
    
    best_inner = initial_inner.copy()
    best_outer_center = np.array([0.0, 0.0])
    best_outer_angle = 0.0
    best_s = _compute_min_outer_side(best_inner, best_outer_center, best_outer_angle)
    
    # Try different outer hexagon angles
    for angle_deg in [0, 30, 60, 90, 120, 150]:
        s = _compute_min_outer_side(initial_inner, best_outer_center, angle_deg)
        if s < best_s:
            best_s = s
            best_outer_angle = angle_deg
    
    # Simulated Annealing
    current_inner = initial_inner.copy()
    current_s = _compute_min_outer_side(current_inner, best_outer_center, best_outer_angle)
    
    T_initial = 0.25
    T_min = 0.0003
    n_iterations = 300000
    
    np.random.seed(42)
    
    for it in range(n_iterations):
        if time.time() > deadline - 90.0:
            break
        
        # Exponential cooling
        T = T_initial * np.exp(-it / 25000.0)
        T = max(T, T_min)
        
        # Scale perturbation with temperature
        scale = 0.04 + 0.12 * (T / T_initial)
        
        new_inner = current_inner.copy()
        n_perturb = 1 if np.random.rand() < 0.7 else 2
        indices = np.random.choice(12, size=n_perturb, replace=False)
        for idx in indices:
            new_inner[idx, 0] += np.random.randn() * scale
            new_inner[idx, 1] += np.random.randn() * scale
            new_inner[idx, 2] += np.random.randn() * 0.02
        
        # Fast overlap rejection
        if _fast_overlap_check(new_inner):
            continue
        
        new_s = _compute_min_outer_side(new_inner, best_outer_center, best_outer_angle)
        
        # Full validity check
        outer_arr = np.array([best_outer_center[0], best_outer_center[1], best_outer_angle])
        new_valid = _is_valid(new_inner, outer_arr, new_s)
        
        if new_valid:
            if new_s < current_s:
                current_inner = new_inner
                current_s = new_s
                if current_s < best_s:
                    best_inner = current_inner.copy()
                    best_s = current_s
            else:
                delta = new_s - current_s
                if delta < T * 10 and np.random.rand() < np.exp(-delta / max(T, 1e-10)):
                    current_inner = new_inner
                    current_s = new_s
        else:
            penalty = _overlap_penalty(new_inner)
            if penalty < 0.03 and np.random.rand() < 0.03:
                current_inner = new_inner
                current_s = new_s + penalty * 0.1
    
    # Final refinement with Nelder-Mead using smooth objective
    from scipy.optimize import minimize
    
    def objective(params):
        if time.time() > deadline - 15.0:
            return 1e10
        inner = params.reshape(12, 3).copy()
        s = _compute_min_outer_side(inner, best_outer_center, best_outer_angle)
        
        # Smooth penalty for overlaps
        pen = 0.0
        for i in range(12):
            for j in range(i+1, 12):
                dist = np.linalg.norm(inner[i, :2] - inner[j, :2])
                if dist < 2.0:
                    pen += (2.0 - dist) ** 2
        
        # Check if all vertices inside outer
        outer_arr = np.array([best_outer_center[0], best_outer_center[1], best_outer_angle])
        outer_verts = _hex_vertices(outer_arr[0], outer_arr[1], s, outer_arr[2])
        inside_penalty = 0.0
        for i in range(12):
            verts_i = _hex_vertices(inner[i, 0], inner[i, 1], 1.0, inner[i, 2])
            for v in verts_i:
                for k in range(6):
                    p1 = outer_verts[k]
                    p2 = outer_verts[(k+1)%6]
                    cross = (p2[0]-p1[0])*(v[1]-p1[1]) - (p2[1]-p1[1])*(v[0]-p1[0])
                    if cross < 0:
                        inside_penalty += cross**2
        
        return s + 10.0 * pen + 10.0 * inside_penalty
    
    x0 = best_inner.flatten()
    try:
        result = minimize(
            objective, x0,
            method='Nelder-Mead',
            options={'maxiter': 800, 'xatol': 1e-8, 'fatol': 1e-10, 'adaptive': True}
        )
        if result.fun < best_s + 0.01:
            candidate_inner = result.x.reshape(12, 3).copy()
            candidate_s = _compute_min_outer_side(candidate_inner, best_outer_center, best_outer_angle)
            outer_arr = np.array([best_outer_center[0], best_outer_center[1], best_outer_angle])
            if _is_valid(candidate_inner, outer_arr, candidate_s):
                best_inner = candidate_inner
                best_s = candidate_s
    except Exception:
        pass
    
    # Second polish pass with tighter tolerances
    if time.time() < deadline - 40.0:
        try:
            result2 = minimize(
                objective, best_inner.flatten(),
                method='Nelder-Mead',
                options={'maxiter': 500, 'xatol': 1e-9, 'fatol': 1e-11, 'adaptive': True}
            )
            if result2.fun < best_s + 0.005:
                candidate_inner = result2.x.reshape(12, 3).copy()
                candidate_s = _compute_min_outer_side(candidate_inner, best_outer_center, best_outer_angle)
                outer_arr = np.array([best_outer_center[0], best_outer_center[1], best_outer_angle])
                if _is_valid(candidate_inner, outer_arr, candidate_s):
                    best_inner = candidate_inner
                    best_s = candidate_s
        except Exception:
            pass
    
    # Third polish pass if time allows
    if time.time() < deadline - 20.0:
        try:
            result3 = minimize(
                objective, best_inner.flatten(),
                method='Nelder-Mead',
                options={'maxiter': 300, 'xatol': 1e-10, 'fatol': 1e-12, 'adaptive': True}
            )
            if result3.fun < best_s + 0.002:
                candidate_inner = result3.x.reshape(12, 3).copy()
                candidate_s = _compute_min_outer_side(candidate_inner, best_outer_center, best_outer_angle)
                outer_arr = np.array([best_outer_center[0], best_outer_center[1], best_outer_angle])
                if _is_valid(candidate_inner, outer_arr, candidate_s):
                    best_inner = candidate_inner
                    best_s = candidate_s
        except Exception:
            pass
    
    # Final check and return
    final_outer_center = np.array([0.0, 0.0])
    final_outer_angle = best_outer_angle
    final_s = _compute_min_outer_side(best_inner, final_outer_center, final_outer_angle)
    
    if not _is_valid(best_inner, np.array([final_outer_center[0], final_outer_center[1], final_outer_angle]), final_s):
        best_inner = initial_inner
        final_s = _compute_min_outer_side(best_inner, final_outer_center, final_outer_angle)
    
    inner_hex_data = best_inner.astype(float)
    outer_hex_data = np.array([final_outer_center[0], final_outer_center[1], final_outer_angle])
    outer_hex_side_length = float(final_s)
    
    return inner_hex_data, outer_hex_data, outer_hex_side_length


# EVOLVE-BLOCK-END
