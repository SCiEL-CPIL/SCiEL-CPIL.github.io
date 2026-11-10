# EVOLVE-BLOCK-START
import numpy as np
import time


def _hex_vertices(cx, cy, s, a_deg):
    a = np.deg2rad(a_deg)
    angles = a + np.deg2rad(60.0 * np.arange(6))
    return np.column_stack([cx + s * np.cos(angles), cy + s * np.sin(angles)])


def _hexes_separated(v1, v2, a1_deg, a2_deg):
    """Check if two hexagons are separated. Returns True if separated on at least one of 12 normals."""
    a1 = np.deg2rad(a1_deg)
    a2 = np.deg2rad(a2_deg)
    normals1 = a1 + np.deg2rad(30.0 + 60.0 * np.arange(6))
    normals2 = a2 + np.deg2rad(30.0 + 60.0 * np.arange(6))
    
    all_normals = np.vstack([normals1, normals2])
    cos_n = np.cos(all_normals)
    sin_n = np.sin(all_normals)
    
    for k in range(12):
        n = np.array([cos_n[k], sin_n[k]])
        p1 = v1 @ n
        p2 = v2 @ n
        sep = min(np.min(p2) - np.max(p1), np.min(p1) - np.max(p2))
        if sep > 1e-6:
            return True
    return False


def _max_overlap(v1, v2, a1_deg, a2_deg):
    """Return maximum overlap depth across all 12 normals (positive = intersecting)."""
    a1 = np.deg2rad(a1_deg)
    a2 = np.deg2rad(a2_deg)
    normals1 = a1 + np.deg2rad(30.0 + 60.0 * np.arange(6))
    normals2 = a2 + np.deg2rad(30.0 + 60.0 * np.arange(6))
    
    all_normals = np.vstack([normals1, normals2])
    cos_n = np.cos(all_normals)
    sin_n = np.sin(all_normals)
    
    max_overlap = 0.0
    for k in range(12):
        n = np.array([cos_n[k], sin_n[k]])
        p1 = v1 @ n
        p2 = v2 @ n
        overlap = min(np.max(p2) - np.min(p1), np.max(p1) - np.min(p2))
        if overlap > max_overlap:
            max_overlap = overlap
    return max_overlap


def _min_enclosing_hex_side(vertices_all, outer_angle_deg=0.0):
    """Minimum side of regular hex (centered at origin) enclosing all vertices."""
    oa = np.deg2rad(outer_angle_deg)
    normals = oa + np.deg2rad(30.0 + 60.0 * np.arange(6))
    cos_n = np.cos(normals)
    sin_n = np.sin(normals)
    
    min_side = 0.0
    for p in vertices_all:
        projections = p[0] * cos_n + p[1] * sin_n
        abs_projections = np.abs(projections)
        required_s = np.max(abs_projections) / (np.sqrt(3.0) / 2.0)
        if required_s > min_side:
            min_side = required_s
    return min_side


def _compute_objective(params):
    """Outer hex side + quadratic penalty for overlaps."""
    centers = params[:22].reshape(11, 2)
    angles = params[22:]
    
    vertices_list = []
    for i in range(11):
        v = _hex_vertices(centers[i, 0], centers[i, 1], 1.0, angles[i])
        vertices_list.append(v)
    
    penalty = 0.0
    for i in range(11):
        for j in range(i + 1, 11):
            overlap = _max_overlap(vertices_list[i], vertices_list[j], angles[i], angles[j])
            if overlap > 0:
                penalty += overlap ** 2
    
    all_vertices = np.vstack(vertices_list)
    outer_side = _min_enclosing_hex_side(all_vertices, 0.0)
    
    return outer_side + 50.0 * penalty


def hexagon_packing_11():
    """Optimizes packing of 11 unit hexagons in a regular hexagon via Nelder-Mead."""
    t_start = time.time()
    deadline = t_start + 570.0
    
    dx = 2.5
    dy = 3.0 * np.sqrt(3.0) / 2.0
    initial_centers = np.array([
        [0.0, 0.0],
        [-dx, 0.0], [dx, 0.0],
        [-dx/2, dy], [dx/2, dy],
        [-dx/2, -dy], [dx/2, -dy],
        [-1.5*dx, dy], [1.5*dx, dy],
        [-1.5*dx, -dy], [1.5*dx, -dy],
    ])
    
    x0 = np.hstack([initial_centers.flatten(), np.zeros(11)])
    
    try:
        from scipy.optimize import minimize
        
        best_x = x0.copy()
        best_val = _compute_objective(x0)
        
        n_restarts = 5
        for r in range(n_restarts):
            if time.time() > deadline - 20:
                break
            
            if r == 0:
                x0_trial = x0.copy()
            else:
                rng = np.random.default_rng(1000 + r * 7919)
                x0_trial = x0 + rng.normal(0, 0.5, size=x0.shape)
            
            try:
                result = minimize(
                    _compute_objective, x0_trial,
                    method='Nelder-Mead',
                    options={'maxiter': 3000, 'xatol': 1e-8, 'fatol': 1e-10}
                )
                val = result.fun
                if val < best_val:
                    best_val = val
                    best_x = result.x
            except Exception:
                pass
        
        centers = best_x[:22].reshape(11, 2)
        angles = best_x[22:]
        
        vertices_list = [_hex_vertices(centers[i, 0], centers[i, 1], 1.0, angles[i]) for i in range(11)]
        all_vertices = np.vstack(vertices_list)
        outer_side = _min_enclosing_hex_side(all_vertices, 0.0)
        
        inner_hex_data = np.column_stack([centers, angles])
        outer_hex_data = np.array([0.0, 0.0, 0.0])
        
        return inner_hex_data, outer_hex_data, float(outer_side)
        
    except Exception:
        inner_hex_data = np.column_stack([initial_centers, np.zeros(11)])
        vertices_list = [_hex_vertices(initial_centers[i, 0], initial_centers[i, 1], 1.0, 0.0) for i in range(11)]
        all_vertices = np.vstack(vertices_list)
        outer_side = _min_enclosing_hex_side(all_vertices, 0.0)
        outer_hex_data = np.array([0.0, 0.0, 0.0])
        return inner_hex_data, outer_hex_data, float(outer_side)


# EVOLVE-BLOCK-END
