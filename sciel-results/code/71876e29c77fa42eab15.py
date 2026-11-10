# EVOLVE-BLOCK-START
import numpy as np
import time


def _hex_vertices(cx, cy, s, a_deg):
    a = np.radians(a_deg)
    angles = a + np.deg2rad(60.0 * np.arange(6))
    return np.column_stack([cx + s * np.cos(angles), cy + s * np.sin(angles)])


def _compute_objective(params):
    """Outer hex side + quadratic penalty for overlaps. Vectorized."""
    centers = params[:22].reshape(11, 2)
    angles = params[22:]
    
    # Compute all vertices at once: shape (11, 6, 2)
    a_rad = np.radians(angles)[:, np.newaxis]
    base_angles = np.deg2rad(60.0 * np.arange(6))[np.newaxis, :]
    all_angles = a_rad + base_angles  # (11, 6)
    
    cos_a = np.cos(all_angles)  # (11, 6)
    sin_a = np.sin(all_angles)  # (11, 6)
    
    cx = centers[:, 0:1]  # (11, 1)
    cy = centers[:, 1:2]  # (11, 1)
    
    all_vertices = np.stack([cx + cos_a, cy + sin_a], axis=-1)  # (11, 6, 2)
    
    # Compute overlap penalty for all pairs
    penalty = 0.0
    n = 11
    for i in range(n):
        for j in range(i + 1, n):
            v1 = all_vertices[i]  # (6, 2)
            v2 = all_vertices[j]  # (6, 2)
            
            # Edge normals for hex i (6 normals)
            a1 = a_rad[i, 0]
            normals1 = a1 + np.deg2rad(30.0 + 60.0 * np.arange(6))
            # Edge normals for hex j (6 normals)
            a2 = a_rad[j, 0]
            normals2 = a2 + np.deg2rad(30.0 + 60.0 * np.arange(6))
            
            all_normals = np.concatenate([normals1, normals2])  # (12,)
            cos_n = np.cos(all_normals)  # (12,)
            sin_n = np.sin(all_normals)  # (12,)
            
            # Project vertices onto each normal
            p1 = v1 @ np.stack([cos_n, sin_n], axis=-1)  # (6, 12)
            p2 = v2 @ np.stack([cos_n, sin_n], axis=-1)  # (6, 12)
            
            # For each normal, compute overlap
            # overlap_k = min(max(p2_k) - min(p1_k), max(p1_k) - min(p2_k))
            p1_min = np.min(p1, axis=0)  # (12,)
            p1_max = np.max(p1, axis=0)  # (12,)
            p2_min = np.min(p2, axis=0)  # (12,)
            p2_max = np.max(p2, axis=0)  # (12,)
            
            overlap = np.minimum(p2_max - p1_min, p1_max - p2_min)  # (12,)
            max_overlap = np.max(overlap)
            if max_overlap > 0:
                penalty += max_overlap ** 2
    
    # Compute minimum enclosing hexagon side
    all_vertices_flat = all_vertices.reshape(-1, 2)  # (66, 2)
    
    # Outer hexagon centered at origin with angle 0
    outer_normals = np.deg2rad(30.0 + 60.0 * np.arange(6))
    cos_on = np.cos(outer_normals)  # (6,)
    sin_on = np.sin(outer_normals)  # (6,)
    
    # Project all vertices onto outer normals
    projections = all_vertices_flat @ np.stack([cos_on, sin_on], axis=-1)  # (66, 6)
    abs_projections = np.abs(projections)  # (66, 6)
    
    # For a regular hexagon with side s centered at origin, the inradius is s*sqrt(3)/2
    # A point is inside if |projection onto each edge normal| <= inradius
    # So required s = max(|proj|) / (sqrt(3)/2)
    max_abs_proj = np.max(abs_projections, axis=0)  # (6,)
    required_sides = max_abs_proj / (np.sqrt(3.0) / 2.0)  # (6,)
    outer_side = np.max(required_sides)
    
    return outer_side + 50.0 * penalty


def _check_validity(inner, outer, side):
    """Check if the packing is valid."""
    # All values finite
    if not np.all(np.isfinite(inner)) or not np.all(np.isfinite(outer)):
        return False
    if not np.isfinite(side):
        return False
    if side < np.sqrt(11) - 1e-6:
        return False
    
    # Compute vertices for each inner hexagon
    inner_vertices = []
    for i in range(11):
        v = _hex_vertices(inner[i, 0], inner[i, 1], 1.0, inner[i, 2])
        inner_vertices.append(v)
    
    # Check separation between all pairs
    for i in range(11):
        for j in range(i + 1, 11):
            v1 = inner_vertices[i]
            v2 = inner_vertices[j]
            a1 = np.radians(inner[i, 2])
            a2 = np.radians(inner[j, 2])
            
            normals1 = a1 + np.deg2rad(30.0 + 60.0 * np.arange(6))
            normals2 = a2 + np.deg2rad(30.0 + 60.0 * np.arange(6))
            all_normals = np.concatenate([normals1, normals2])
            
            separated = False
            for k in range(12):
                n = np.array([np.cos(all_normals[k]), np.sin(all_normals[k])])
                p1 = v1 @ n
                p2 = v2 @ n
                sep = min(np.min(p2) - np.max(p1), np.min(p1) - np.max(p2))
                if sep > 1e-6:
                    separated = True
                    break
            if not separated:
                return False
    
    # Check containment in outer hexagon
    ocx, ocy, oa_deg = outer[0], outer[1], outer[2]
    oa = np.radians(oa_deg)
    outer_vertices = _hex_vertices(ocx, ocy, side, oa_deg)
    
    # Shift coordinates by outer center
    shift = np.array([ocx, ocy])
    
    for i in range(11):
        v = inner_vertices[i] - shift
        # Check each vertex against each outer edge
        for e in range(6):
            p1 = outer_vertices[e]
            p2 = outer_vertices[(e + 1) % 6]
            edge_vec = p2 - p1
            # Cross product: cross(p2-p1, p-p1) >= -1e-6
            for p in v:
                cp = edge_vec[0] * (p[1] - p1[1]) - edge_vec[1] * (p[0] - p1[0])
                if cp < -1e-6:
                    return False
    
    return True


def hexagon_packing_11():
    """Optimizes packing of 11 unit hexagons in a regular hexagon."""
    t_start = time.time()
    deadline = t_start + 570.0
    
    # Initial configuration: center hex + 6 around it + 4 more
    dx = 2.5
    dy = 3.0 * np.sqrt(3.0) / 2.0
    
    # Standard hexagonal lattice positions
    initial_centers = np.array([
        [0.0, 0.0],           # center
        [-dx, 0.0],           # left
        [dx, 0.0],            # right
        [-dx/2, dy],          # upper-left
        [dx/2, dy],           # upper-right
        [-dx/2, -dy],         # lower-left
        [dx/2, -dy],          # lower-right
        [-1.5*dx, dy],        # far upper-left
        [1.5*dx, dy],         # far upper-right
        [-1.5*dx, -dy],       # far lower-left
        [1.5*dx, -dy],        # far lower-right
    ])
    
    x0 = np.hstack([initial_centers.flatten(), np.zeros(11)])
    
    try:
        from scipy.optimize import minimize
        
        best_x = x0.copy()
        best_val = _compute_objective(x0)
        
        # Multiple restarts with different initial perturbations
        n_restarts = 8
        for r in range(n_restarts):
            if time.time() > deadline - 30:
                break
            
            if r == 0:
                x0_trial = x0.copy()
            elif r == 1:
                # Slightly perturbed
                rng = np.random.default_rng(42)
                x0_trial = x0 + rng.normal(0, 0.1, size=x0.shape)
            elif r == 2:
                # Different angle initialization
                x0_trial = x0.copy()
                x0_trial[22:] = np.deg2rad(np.pi/6)  # 30 degree rotation
            elif r == 3:
                # More aggressive perturbation
                rng = np.random.default_rng(123)
                x0_trial = x0 + rng.normal(0, 0.3, size=x0.shape)
            elif r == 4:
                # Try with some angles rotated
                x0_trial = x0.copy()
                x0_trial[22:28] = np.deg2rad(np.pi/12)  # 15 degrees for first 6
            elif r == 5:
                rng = np.random.default_rng(456)
                x0_trial = x0 + rng.normal(0, 0.5, size=x0.shape)
            elif r == 6:
                # Tighter initial positions
                x0_trial = x0.copy()
                x0_trial[:22] = x0[:22] * 0.95  # slightly compressed
            else:
                rng = np.random.default_rng(789 + r * 100)
                x0_trial = x0 + rng.normal(0, 0.2, size=x0.shape)
            
            try:
                # Use L-BFGS-B for faster convergence
                result = minimize(
                    _compute_objective, x0_trial,
                    method='L-BFGS-B',
                    options={'maxiter': 2000, 'ftol': 1e-12, 'gtol': 1e-8}
                )
                val = result.fun
                if val < best_val:
                    best_val = val
                    best_x = result.x.copy()
            except Exception:
                pass
        
        # Final polish with Nelder-Mead on the best result
        if time.time() < deadline - 15:
            try:
                result = minimize(
                    _compute_objective, best_x,
                    method='Nelder-Mead',
                    options={'maxiter': 5000, 'xatol': 1e-9, 'fatol': 1e-12}
                )
                val = result.fun
                if val < best_val:
                    best_val = val
                    best_x = result.x.copy()
            except Exception:
                pass
        
        centers = best_x[:22].reshape(11, 2)
        angles = best_x[22:]
        
        # Compute outer side
        all_vertices = []
        for i in range(11):
            v = _hex_vertices(centers[i, 0], centers[i, 1], 1.0, angles[i])
            all_vertices.append(v)
        all_vertices = np.vstack(all_vertices)
        
        outer_normals = np.deg2rad(30.0 + 60.0 * np.arange(6))
        cos_on = np.cos(outer_normals)
        sin_on = np.sin(outer_normals)
        projections = all_vertices @ np.stack([cos_on, sin_on], axis=-1)
        abs_projections = np.abs(projections)
        max_abs_proj = np.max(abs_projections, axis=0)
        required_sides = max_abs_proj / (np.sqrt(3.0) / 2.0)
        outer_side = float(np.max(required_sides))
        
        inner_hex_data = np.column_stack([centers, angles])
        outer_hex_data = np.array([0.0, 0.0, 0.0])
        
        return inner_hex_data, outer_hex_data, outer_side
        
    except Exception:
        inner_hex_data = np.column_stack([initial_centers, np.zeros(11)])
        vertices_list = [_hex_vertices(initial_centers[i, 0], initial_centers[i, 1], 1.0, 0.0) for i in range(11)]
        all_vertices = np.vstack(vertices_list)
        outer_normals = np.deg2rad(30.0 + 60.0 * np.arange(6))
        cos_on = np.cos(outer_normals)
        sin_on = np.sin(outer_normals)
        projections = all_vertices @ np.stack([cos_on, sin_on], axis=-1)
        abs_projections = np.abs(projections)
        max_abs_proj = np.max(abs_projections, axis=0)
        required_sides = max_abs_proj / (np.sqrt(3.0) / 2.0)
        outer_side = float(np.max(required_sides))
        outer_hex_data = np.array([0.0, 0.0, 0.0])
        return inner_hex_data, outer_hex_data, outer_side


# EVOLVE-BLOCK-END
