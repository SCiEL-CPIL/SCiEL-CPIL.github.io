# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import minimize


def _hex_vertices(cx, cy, s, a_deg):
    a = np.radians(a_deg)
    angs = a + np.deg2rad(60.0 * np.arange(6))
    return np.column_stack([cx + s * np.cos(angs), cy + s * np.sin(angs)])


def _hex_normals(a_deg):
    a = np.radians(a_deg)
    edge_angles = a + np.deg2rad(60.0 * np.arange(6)) + np.deg2rad(30.0)
    normals = np.column_stack([np.cos(edge_angles), np.sin(edge_angles)])
    return np.vstack([normals, -normals])


def _separation_ok(v1, v2, n1, n2):
    """Check if two hexagons are separated on at least one of their 12 edge normals."""
    # Project both hexagons onto each of the 24 normals (12 from each)
    # For each normal, check if the projected intervals are disjoint
    for normals in (n1, n2):
        proj1 = v1 @ normals.T  # (6, 12)
        proj2 = v2 @ normals.T  # (6, 12)
        min1 = proj1.min(axis=0)  # (12,)
        max1 = proj1.max(axis=0)  # (12,)
        min2 = proj2.min(axis=0)  # (12,)
        max2 = proj2.max(axis=0)  # (12,)
        # Separation if max1 < min2 - tol OR max2 < min1 - tol for ANY normal
        if np.any(max1 < min2 - 1e-6) or np.any(max2 < min1 - 1e-6):
            return True
    return False


def _hexagons_separated(v1, v2, a1, a2):
    n1 = _hex_normals(a1)
    n2 = _hex_normals(a2)
    return _separation_ok(v1, v2, n1, n2)


def _point_in_hexagon(p, outer_verts):
    """Check if point p is inside (or on boundary of) the outer hexagon."""
    for i in range(6):
        p1 = outer_verts[i]
        p2 = outer_verts[(i + 1) % 6]
        edge = p2 - p1
        to_p = p - p1
        cross = edge[0] * to_p[1] - edge[1] * to_p[0]
        if cross < -1e-6:
            return False
    return True


def _all_inner_in_outer(inner_verts_list, outer_verts):
    """Check all vertices of all inner hexagons are inside the outer hexagon."""
    for verts in inner_verts_list:
        for p in verts:
            if not _point_in_hexagon(p, outer_verts):
                return False
    return True


def _build_hexagons(r1, r2, theta1, theta2):
    """
    Build 11 hexagons: center + 6 inner ring + 4 outer ring.
    """
    hexagons = []
    hexagons.append((0.0, 0.0, 0.0))
    for i in range(6):
        ang = np.radians(theta1 + 60.0 * i)
        cx = r1 * np.cos(ang)
        cy = r1 * np.sin(ang)
        hexagons.append((cx, cy, 0.0))
    for i in range(4):
        ang = np.radians(theta2 + 90.0 * i)
        cx = r2 * np.cos(ang)
        cy = r2 * np.sin(ang)
        hexagons.append((cx, cy, 0.0))
    return hexagons


def _compute_minimal_outer_hex(hexagons):
    """Compute minimal enclosing regular hexagon for the given inner hexagons."""
    all_verts = []
    for (cx, cy, ang) in hexagons:
        verts = _hex_vertices(cx, cy, 1.0, ang)
        all_verts.append(verts)
    all_verts = np.vstack(all_verts)
    
    best_s = np.inf
    best_a = 0.0
    
    for a_deg in np.linspace(0, 60, 61):
        a = np.radians(a_deg)
        edge_angles = a + np.deg2rad(60.0 * np.arange(6)) + np.deg2rad(30.0)
        normals = np.column_stack([np.cos(edge_angles), np.sin(edge_angles)])
        
        proj = all_verts @ normals.T
        max_proj = proj.max(axis=0)
        min_proj = proj.min(axis=0)
        
        required_apothem = np.maximum(max_proj, -min_proj)
        apothem = required_apothem.max()
        s = apothem / (np.sqrt(3) / 2)
        
        if s < best_s:
            best_s = s
            best_a = a_deg
    
    return (0.0, 0.0, best_a, best_s)


def _objective(params):
    """
    Objective function for optimization.
    params: (22,) array = [x1,y1, x2,y2, ..., x11,y11, outer_angle_deg]
    Returns: side length of minimal enclosing hexagon, or penalty if invalid.
    """
    centers = params[:22].reshape(11, 2)
    outer_angle_deg = params[22]
    
    # All inner hexagons have rotation 0
    all_verts = np.zeros((66, 2))
    for i in range(11):
        cx, cy = centers[i]
        verts = _hex_vertices(cx, cy, 1.0, 0.0)
        all_verts[i*6:(i+1)*6] = verts
    
    # Check separation between all pairs
    for i in range(11):
        for j in range(i + 1, 11):
            v1 = all_verts[i*6:(i+1)*6]
            v2 = all_verts[j*6:(j+1)*6]
            if not _hexagons_separated(v1, v2, 0.0, 0.0):
                # Penalty for overlap: return large value
                return 100.0
    
    # Compute minimal enclosing hexagon for given outer angle
    a = np.radians(outer_angle_deg)
    edge_angles = a + np.deg2rad(60.0 * np.arange(6)) + np.deg2rad(30.0)
    normals = np.column_stack([np.cos(edge_angles), np.sin(edge_angles)])
    
    proj = all_verts @ normals.T
    max_proj = proj.max(axis=0)
    min_proj = proj.min(axis=0)
    
    required_apothem = np.maximum(max_proj, -min_proj)
    apothem = required_apothem.max()
    s = apothem / (np.sqrt(3) / 2)
    
    # Check containment
    outer_verts = _hex_vertices(0.0, 0.0, s, outer_angle_deg)
    for i in range(11):
        for p in all_verts[i*6:(i+1)*6]:
            if not _point_in_hexagon(p, outer_verts):
                return 100.0
    
    return s


def _validate_and_return(centers, outer_angle_deg, s):
    """Validate the solution and return the proper format."""
    hexagons = []
    for i in range(11):
        hexagons.append((float(centers[i, 0]), float(centers[i, 1]), 0.0))
    
    # Verify all constraints
    all_verts = []
    for (cx, cy, ang) in hexagons:
        verts = _hex_vertices(cx, cy, 1.0, ang)
        all_verts.append(verts)
    
    # Check separation
    for i in range(11):
        for j in range(i + 1, 11):
            if not _hexagons_separated(all_verts[i], all_verts[j], 0.0, 0.0):
                return None
    
    # Check containment
    outer_verts = _hex_vertices(0.0, 0.0, s, outer_angle_deg)
    for i in range(11):
        for p in all_verts[i]:
            if not _point_in_hexagon(p, outer_verts):
                return None
    
    inner_hex_data = np.array(hexagons)
    outer_hex_data = np.array([0.0, 0.0, outer_angle_deg])
    
    # Ensure s >= sqrt(11)
    s = max(s, np.sqrt(11.0))
    
    return inner_hex_data, outer_hex_data, float(s)


def hexagon_packing_11():
    """
    Constructs a packing of 11 disjoint unit regular hexagons inside a larger regular hexagon.
    Returns (inner_hex_data, outer_hex_data, outer_hex_side_length).
    """
    start_time = time.time()
    deadline = start_time + 580.0  # Safety margin before 630s limit
    
    best_result = None
    best_s = np.inf
    
    # Initial estimate from symmetric flower packing
    r1 = np.sqrt(3)
    r2 = 2 * np.sqrt(3)
    
    initial_centers = np.zeros((11, 2))
    # Center
    initial_centers[0] = [0.0, 0.0]
    # Inner ring: 6 hexagons
    for i in range(6):
        ang = np.radians(60.0 * i)
        initial_centers[i + 1] = [r1 * np.cos(ang), r1 * np.sin(ang)]
    # Outer ring: 4 hexagons
    for i in range(4):
        ang = np.radians(90.0 * i)
        initial_centers[i + 7] = [r2 * np.cos(ang), r2 * np.sin(ang)]
    
    # Try multiple restarts with different initial conditions
    n_restarts = 5
    for restart in range(n_restarts):
        if time.time() > deadline:
            break
        
        # Perturb initial conditions for each restart
        if restart == 0:
            centers_init = initial_centers.copy()
        else:
            centers_init = initial_centers + 0.1 * np.random.randn(11, 2)
        
        # Random outer angle
        outer_angle_init = np.random.uniform(0, 60)
        
        params_init = np.concatenate([centers_init.flatten(), [outer_angle_init]])
        
        # Run Nelder-Mead optimization
        try:
            result = minimize(
                _objective,
                params_init,
                method='Nelder-Mead',
                options={'maxiter': 2000, 'xatol': 1e-8, 'fatol': 1e-8}
            )
        except Exception:
            continue
        
        if time.time() > deadline:
            break
        
        if result.fun < best_s:
            best_s = result.fun
            centers = result.x[:22].reshape(11, 2)
            outer_angle = result.x[22]
            
            validated = _validate_and_return(centers, outer_angle, best_s)
            if validated is not None:
                best_result = validated
    
    # Fallback: use initial symmetric packing
    if best_result is None:
        hexagons = _build_hexagons(r1, r2, 0.0, 0.0)
        ox, oy, oa, s = _compute_minimal_outer_hex(hexagons)
        
        # Validate
        all_verts = []
        for (cx, cy, ang) in hexagons:
            verts = _hex_vertices(cx, cy, 1.0, ang)
            all_verts.append(verts)
        
        valid = True
        for i in range(11):
            for j in range(i + 1, 11):
                if not _hexagons_separated(all_verts[i], all_verts[j], 0.0, 0.0):
                    valid = False
                    break
            if not valid:
                break
        
        outer_verts = _hex_vertices(ox, oy, s, oa)
        if valid:
            for i in range(11):
                for p in all_verts[i]:
                    if not _point_in_hexagon(p, outer_verts):
                        valid = False
                        break
                if not valid:
                    break
        
        if valid:
            inner_hex_data = np.array(hexagons)
            outer_hex_data = np.array([ox, oy, oa])
            s = max(s, np.sqrt(11.0))
            return inner_hex_data, outer_hex_data, float(s)
    
    return best_result


# EVOLVE-BLOCK-END
