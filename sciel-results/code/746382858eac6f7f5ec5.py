# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import linprog

def _hex_vertices(cx, cy, d, angle_deg):
    """Return vertices of a regular hexagon centered at (cx,cy) with side d and angle angle_deg."""
    a = np.radians(angle_deg)
    angles = a + np.radians(60 * np.arange(6))
    return np.column_stack([cx + d * np.cos(angles), cy + d * np.sin(angles)])

def _compute_min_s_for_rotation(vertices, rot_deg):
    """
    Given all vertices of inner hexagons (N, 2), find minimum side length s
    of a regular hexagon (rotated by rot_deg) that contains all vertices.
    Uses linear programming with variables [ox, oy, s].
    Constraints: For each edge i of the outer hexagon (normal n_i), 
    n_i . p <= n_i . c + sqrt(3)/2 * s for all points p.
    This is equivalent to: -n_i . c + sqrt(3)/2 * s >= -n_i . p
    Or: n_i . c - sqrt(3)/2 * s <= n_i . p
    Wait, the standard form for linprog is A_ub x <= b_ub.
    Let's derive carefully.
    Outer hexagon center c=(ox, oy), side s, rotation rot.
    Vertices are at c + s * (cos(theta), sin(theta)) where theta = rot + 60k.
    The edges are at distance apothem = s * sqrt(3) / 2 from center.
    The normal to edge k points outward at angle alpha_k = rot + 60k + 30.
    The constraint for a point p to be inside is:
    n_k . (p - c) <= apothem
    n_k . p - n_k . c <= s * sqrt(3) / 2
    -n_k . c + s * sqrt(3) / 2 >= n_k . p
    Let x = [ox, oy, s].
    n_k . c = ox * n_kx + oy * n_ky.
    So: -ox * n_kx - oy * n_ky + s * sqrt(3)/2 >= n_k . p
    Rearranging for linprog (A x <= b):
    ox * n_kx + oy * n_ky - s * sqrt(3)/2 <= -n_k . p
    
    Objective: minimize s -> c_obj = [0, 0, 1].
    """
    rot = np.radians(rot_deg)
    n_pts = len(vertices)
    n_constraints = 6 * n_pts
    A = np.zeros((n_constraints, 3))
    b = np.zeros(n_constraints)
    
    idx = 0
    for i in range(6):
        angle = rot + np.radians(60 * i + 30)
        ci = np.cos(angle)
        si = np.sin(angle)
        for j in range(n_pts):
            # ox * ci + oy * si - s * sqrt(3)/2 <= -(ci * vx + si * vy)
            A[idx, 0] = ci
            A[idx, 1] = si
            A[idx, 2] = -np.sqrt(3) / 2
            b[idx] = -(ci * vertices[j, 0] + si * vertices[j, 1])
            idx += 1
    
    c_obj = np.array([0, 0, 1])
    bounds = [(None, None), (None, None), (0, None)]
    
    try:
        res = linprog(c_obj, A_ub=A, b_ub=b, bounds=bounds, method='highs')
        if res.status == 0:
            s = res.x[2]
            ox = res.x[0]
            oy = res.x[1]
            return s, (ox, oy)
        else:
            return np.inf, None
    except Exception:
        return np.inf, None

def _validate_packing(inner_hex_data, outer_hex_data, s, eps=1e-6):
    """Check if the packing is valid. Returns True if valid."""
    if not np.all(np.isfinite(inner_hex_data)) or not np.all(np.isfinite(outer_hex_data)):
        return False
    if s < np.sqrt(12) - 1e-9:
        return False
    
    ox, oy, oang = outer_hex_data
    outer_verts = _hex_vertices(ox, oy, s, oang)
    
    # Check outer hex vertices distinct
    for i in range(6):
        for j in range(i + 1, 6):
            if np.linalg.norm(outer_verts[i] - outer_verts[j]) < 1e-8:
                return False
    
    # Check inner hex edge lengths
    for i in range(12):
        cx, cy, aang = inner_hex_data[i]
        verts = _hex_vertices(cx, cy, 1, aang)
        for k in range(6):
            v1 = verts[k]
            v2 = verts[(k + 1) % 6]
            edge_len = np.linalg.norm(v2 - v1)
            if abs(edge_len - 1) > 1e-8:
                return False
            
    # Check non-overlap using SAT
    for i in range(12):
        for j in range(i + 1, 12):
            vi = _hex_vertices(inner_hex_data[i, 0], inner_hex_data[i, 1], 1, inner_hex_data[i, 2])
            vj = _hex_vertices(inner_hex_data[j, 0], inner_hex_data[j, 1], 1, inner_hex_data[j, 2])
            
            normals = []
            for v in [vi, vj]:
                for k in range(6):
                    e = v[(k + 1) % 6] - v[k]
                    n = np.array([-e[1], e[0]])
                    n = n / np.linalg.norm(n)
                    normals.append(n)
            
            separated = False
            for n in normals:
                proj_i = np.dot(vi, n)
                proj_j = np.dot(vj, n)
                min_i, max_i = proj_i.min(), proj_i.max()
                min_j, max_j = proj_j.min(), proj_j.max()
                if max_i + eps < min_j or max_j + eps < min_i:
                    separated = True
                    break
            
            if not separated:
                return False
    
    # Check containment
    for i in range(12):
        cx, cy, aang = inner_hex_data[i]
        verts = _hex_vertices(cx, cy, 1, aang)
        for v in verts:
            inside = True
            for k in range(6):
                p1 = outer_verts[k]
                p2 = outer_verts[(k + 1) % 6]
                cross = (p2[0] - p1[0]) * (v[1] - p1[1]) - (p2[1] - p1[1]) * (v[0] - p1[0])
                if cross < -1e-6:
                    inside = False
                    break
            if not inside:
                return False
    
    return True

def hexagon_packing_12():
    start_time = time.time()
    deadline = start_time + 600.0
    sqrt3 = np.sqrt(3)
    
    # Flower pattern: center + 6 neighbors + 5 from next ring
    # This is a known compact configuration
    centers_base = np.array([
        [0, 0],
        [2, 0],
        [1, sqrt3],
        [-1, sqrt3],
        [-2, 0],
        [-1, -sqrt3],
        [1, -sqrt3],
        [3, sqrt3],
        [0, 2 * sqrt3],
        [-3, sqrt3],
        [-3, -sqrt3],
        [0, -2 * sqrt3]
    ])
    
    best_s = np.inf
    best_inner = None
    best_outer = None
    
    def try_configuration(centers, theta_deg, phi_deg):
        nonlocal best_s, best_inner, best_outer
        if time.time() > deadline - 5:
            return
        
        # Build cluster vertices
        a = np.radians(theta_deg)
        unit_angles = np.radians(60 * np.arange(6))
        unit_verts = np.column_stack([np.cos(unit_angles), np.sin(unit_angles)])
        
        cos_a, sin_a = np.cos(a), np.sin(a)
        rot_matrix = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
        rot_unit_verts = unit_verts @ rot_matrix.T
        
        all_verts = np.zeros((72, 2))
        for j in range(12):
            for k in range(6):
                all_verts[j * 6 + k] = centers[j] + rot_unit_verts[k]
        
        s, center = _compute_min_s_for_rotation(all_verts, phi_deg)
        if s < best_s:
            inner_hex_data = np.column_stack([centers, np.full(12, theta_deg)])
            outer_hex_data = np.array([center[0], center[1], phi_deg])
            if _validate_packing(inner_hex_data, outer_hex_data, s):
                best_s = s
                best_inner = inner_hex_data.copy()
                best_outer = outer_hex_data.copy()

    # Phase 1: Coarse grid search over rotations
    # theta: rotation of inner hexagons
    # phi: rotation of outer hexagon
    n_theta = 30
    n_phi = 30
    theta_vals = np.linspace(0, 60, n_theta)
    phi_vals = np.linspace(0, 60, n_phi)
    
    for theta_deg in theta_vals:
        if time.time() > deadline - 100:
            break
        for phi_deg in phi_vals:
            if time.time() > deadline - 100:
                break
            try_configuration(centers_base, theta_deg, phi_deg)
            
    # Phase 2: Local search to refine positions and rotations
    if best_inner is not None and time.time() < deadline - 50:
        current_inner = best_inner.copy()
        current_s = best_s
        current_outer = best_outer.copy()
        
        # We will optimize centers and theta
        # phi is determined by the LP for a given set of vertices
        
        for iteration in range(50):
            if time.time() > deadline - 10:
                break
            improved = False
            for i in range(12):
                if time.time() > deadline - 10:
                    break
                # Try small perturbations in x, y for hex i
                for dx in [-0.05, 0, 0.05]:
                    for dy in [-0.05, 0, 0.05]:
                        new_centers = current_inner[:, :2].copy()
                        new_centers[i, 0] += dx
                        new_centers[i, 1] += dy
                        
                        # Keep theta same for now, try a few theta variations
                        for dtheta in [-1, 0, 1]:
                            new_theta = current_inner[0, 2] + dtheta
                            new_inner_candidate = np.column_stack([new_centers, np.full(12, new_theta)])
                            
                            # Quick check: do we still have overlaps?
                            # We can skip full validation here and let try_configuration do it,
                            # but try_configuration is expensive.
                            # Let's just call try_configuration which validates.
                            # However, try_configuration recomputes phi via LP.
                            
                            # To speed up, we can estimate if this is promising.
                            # But for simplicity and correctness, we call try_configuration.
                            # We limit the number of calls per iteration.
                            if dx == 0 and dy == 0 and dtheta == 0:
                                continue
                                
                            # We need to find the best phi for this new config.
                            # try_configuration does this.
                            # But try_configuration updates global best only if valid and better.
                            # We want to accept improvements locally to guide the search.
                            
                            # Let's modify try_configuration to return the s value for local tracking
                            # Or just rely on the global best update.
                            
                            # To make local search work, we need to know if the new config is valid.
                            # We'll run a lightweight check:
                            # 1. Build verts
                            # 2. Check SAT overlaps
                            # 3. If no overlaps, run LP for s
                            # 4. If s < current_s, accept.
                            
                            a = np.radians(new_theta)
                            unit_angles = np.radians(60 * np.arange(6))
                            unit_verts = np.column_stack([np.cos(unit_angles), np.sin(unit_angles)])
                            cos_a, sin_a = np.cos(a), np.sin(a)
                            rot_matrix = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
                            rot_unit_verts = unit_verts @ rot_matrix.T
                            
                            all_verts = np.zeros((72, 2))
                            for j in range(12):
                                for k in range(6):
                                    all_verts[j * 6 + k] = new_centers[j] + rot_unit_verts[k]
                                    
                            # Check overlaps
                            overlap = False
                            for k in range(12):
                                for l in range(k+1, 12):
                                    vk = _hex_vertices(new_centers[k,0], new_centers[k,1], 1, new_theta)
                                    vl = _hex_vertices(new_centers[l,0], new_centers[l,1], 1, new_theta)
                                    # Simplified SAT check
                                    normals = []
                                    for v in [vk, vl]:
                                        for m in range(6):
                                            e = v[(m + 1) % 6] - v[m]
                                            n = np.array([-e[1], e[0]])
                                            n = n / np.linalg.norm(n)
                                            normals.append(n)
                                    sep = False
                                    for n in normals:
                                        proj_k = np.dot(vk, n)
                                        proj_l = np.dot(vl, n)
                                        if proj_k.max() + 1e-6 < proj_l.min() or proj_l.max() + 1e-6 < proj_k.min():
                                            sep = True
                                            break
                                    if not sep:
                                        overlap = True
                                        break
                                if overlap:
                                    break
                            
                            if overlap:
                                continue
                            
                            # Find best phi
                            best_phi_s = np.inf
                            best_phi = 0
                            for phi in np.linspace(0, 60, 10):
                                s, center = _compute_min_s_for_rotation(all_verts, phi)
                                if s < best_phi_s:
                                    best_phi_s = s
                                    best_phi = phi
                                    
                            if best_phi_s < current_s - 1e-8:
                                # Accept
                                current_s = best_phi_s
                                current_inner[:, :2] = new_centers
                                current_inner[:, 2] = new_theta
                                improved = True
                                
                                # Update global best if valid
                                outer_candidate = np.array([center[0], center[1], best_phi])
                                if _validate_packing(current_inner, outer_candidate, current_s):
                                    best_s = current_s
                                    best_inner = current_inner.copy()
                                    best_outer = outer_candidate.copy()
                                    
            if not improved:
                break

    # Fallback if no valid packing found (should not happen with flower seed)
    if best_inner is None:
        try_configuration(centers_base, 0, 0)
        
    return best_inner, best_outer, best_s

# EVOLVE-BLOCK-END
