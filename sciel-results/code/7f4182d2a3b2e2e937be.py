import numpy as np
import time
from scipy.optimize import minimize, differential_evolution

N = 12
DEADLINE = 145.0  # Safety margin for 150s limit

# Unit hexagon geometry (pointy-top: vertices at 30, 90, 150, 210, 270, 330 degrees)
# This gives flat sides facing 0, 60, 120, 180, 240, 300
UNIT_VERTS_PT = np.array([
    [np.cos(np.radians(60 * i + 30)), np.sin(np.radians(60 * i + 30))] for i in range(6)
])

# Unit hexagon geometry (flat-top: vertices at 0, 60, 120, 180, 240, 300)
UNIT_VERTS_FT = np.array([
    [np.cos(np.radians(60 * i)), np.sin(np.radians(60 * i))] for i in range(6)
])

# Outward normals for the 6 edges of a regular hexagon (pointing to edge midpoints)
# For a hexagon with vertices at angles theta_i, edge midpoints are at theta_i + 30
OUTER_NORMALS = np.array([
    [np.cos(np.radians(60 * i + 30)), np.sin(np.radians(60 * i + 30))] for i in range(6)
])


def get_hex_vertices(center, angle_deg, verts_template=UNIT_VERTS_FT):
    """Get vertices of a unit hexagon at given center and rotation."""
    a_rad = np.radians(angle_deg)
    c = np.cos(a_rad)
    s = np.sin(a_rad)
    R = np.array([[c, -s], [s, c]])
    return verts_template @ R.T + np.array(center)


def compute_required_side(centers, angles, verts_templates, outer_center=(0.0, 0.0), outer_angle=0.0):
    """
    Compute the minimum outer side length needed to contain all inner hexagons.
    centers: list of (x, y)
    angles: list of rotation angles in degrees
    verts_templates: list of vertex templates (each 6x2)
    """
    oa_rad = np.radians(outer_angle)
    c_o = np.cos(-oa_rad)
    s_o = np.sin(-oa_rad)
    R_o = np.array([[c_o, -s_o], [s_o, c_o]])
    
    max_extent = 0.0
    for i in range(len(centers)):
        v = get_hex_vertices(centers[i], angles[i], verts_templates[i])
        v_rel = v - np.array(outer_center)
        v_rot = v_rel @ R_o.T
        projs = v_rot @ OUTER_NORMALS.T
        abs_projs = np.abs(projs)
        max_proj = np.max(abs_projs, axis=1)
        max_extent = max(max_extent, float(np.max(max_proj)))
    
    # Apothem = side * sqrt(3)/2  =>  side = apothem / (sqrt(3)/2)
    return max_extent / (np.sqrt(3) / 2)


def sat_min_separation(v1, v2):
    """
    Compute minimum separation between two convex polygons using SAT.
    Returns negative value if overlapping.
    """
    edges1 = np.roll(v1, -1, axis=0) - v1
    edges2 = np.roll(v2, -1, axis=0) - v2
    axes = np.vstack([edges1, edges2])
    norms = np.linalg.norm(axes, axis=1, keepdims=True)
    axes = axes / (norms + 1e-15)
    
    proj1 = v1 @ axes.T
    proj2 = v2 @ axes.T
    
    sep = np.minimum(proj1.min(axis=0) - proj2.max(axis=0),
                    proj2.min(axis=0) - proj1.max(axis=0))
    
    return float(np.max(sep))


def check_validity(centers, angles, verts_templates, outer_side, outer_center=(0.0, 0.0), outer_angle=0.0, tol=1e-6):
    """
    Check if packing is valid: no overlaps, all inside outer hexagon.
    """
    n = len(centers)
    
    # Precompute all vertices
    all_verts = []
    for i in range(n):
        all_verts.append(get_hex_vertices(centers[i], angles[i], verts_templates[i]))
    
    # 1. Check pairwise separation (SAT)
    for i in range(n):
        v1 = all_verts[i]
        for j in range(i + 1, n):
            v2 = all_verts[j]
            sep = sat_min_separation(v1, v2)
            if sep < -tol:
                return False
    
    # 2. Check containment
    apothem = outer_side * np.sqrt(3) / 2
    oa_rad = np.radians(outer_angle)
    c_o = np.cos(-oa_rad)
    s_o = np.sin(-oa_rad)
    R_o = np.array([[c_o, -s_o], [s_o, c_o]])
    
    for i in range(n):
        v = all_verts[i]
        v_rel = v - np.array(outer_center)
        v_rot = v_rel @ R_o.T
        projs = v_rot @ OUTER_NORMALS.T
        abs_projs = np.abs(projs)
        max_proj = np.max(abs_projs, axis=1)
        
        if np.max(max_proj) > apothem + tol:
            return False
            
    return True


def generate_symmetric_config(r1, r2, ring1_angle, ring2_angle, outer_angle, use_pointy=False):
    """
    Generate a symmetric configuration:
    - 1 hexagon at center
    - 6 hexagons on ring 1 (radius r1, evenly spaced at 60 degrees)
    - 5 hexagons on ring 2 (radius r2, evenly spaced at 72 degrees, offset by 36 degrees)
    
    Returns: (centers, angles, verts_templates, outer_angle)
    """
    verts_template = UNIT_VERTS_PT if use_pointy else UNIT_VERTS_FT
    
    centers = [(0.0, 0.0)]
    angles = [0.0]
    templates = [verts_template]
    
    # Ring 1: 6 hexagons
    for k in range(6):
        theta = np.radians(60 * k + ring1_angle)
        x = r1 * np.cos(theta)
        y = r1 * np.sin(theta)
        centers.append((x, y))
        angles.append(ring1_angle)  # All same rotation
        templates.append(verts_template)
    
    # Ring 2: 5 hexagons
    for k in range(5):
        theta = np.radians(72 * k + 36 + ring2_angle)
        x = r2 * np.cos(theta)
        y = r2 * np.sin(theta)
        centers.append((x, y))
        angles.append(ring2_angle)  # All same rotation
        templates.append(verts_template)
    
    return centers, angles, templates, outer_angle


def objective_symmetric(params, use_pointy=False):
    """
    Objective function for the symmetric configuration.
    params: [r1, r2, ring1_angle, ring2_angle, outer_angle]
    """
    r1, r2, ring1_angle, ring2_angle, outer_angle = params
    
    # Ensure positive radii
    r1 = max(0.1, r1)
    r2 = max(0.1, r2)
    
    centers, angles, templates, oa = generate_symmetric_config(
        r1, r2, ring1_angle, ring2_angle, outer_angle, use_pointy
    )
    
    # Compute required side
    req_side = compute_required_side(centers, angles, templates, (0.0, 0.0), oa)
    
    # Check for overlaps
    n = len(centers)
    all_verts = []
    for i in range(n):
        all_verts.append(get_hex_vertices(centers[i], angles[i], templates[i]))
    
    min_sep = float('inf')
    for i in range(n):
        for j in range(i + 1, n):
            sep = sat_min_separation(all_verts[i], all_verts[j])
            if sep < min_sep:
                min_sep = sep
    
    overlap_penalty = 0.0
    if min_sep < 0:
        overlap_penalty = -min_sep * 100.0
    
    return req_side + overlap_penalty


def optimize_symmetric(use_pointy=False, time_budget=20.0):
    """
    Optimize the symmetric configuration using differential evolution and local refinement.
    """
    start_time = time.time()
    
    # Parameter bounds
    # r1: distance to ring 1 (should be around sqrt(3) ~ 1.73 for touching)
    # r2: distance to ring 2 (should be larger, around 3-4)
    # angles: 0-360 degrees
    bounds = [
        (1.0, 3.0),      # r1
        (2.5, 5.0),      # r2
        (0.0, 360.0),    # ring1_angle
        (0.0, 360.0),    # ring2_angle
        (0.0, 60.0)      # outer_angle (by symmetry, only need 0-60)
    ]
    
    x0 = np.array([1.73, 3.5, 0.0, 0.0, 0.0])
    
    best_x = x0.copy()
    best_val = objective_symmetric(x0, use_pointy)
    
    # Phase 1: Differential Evolution (global search)
    elapsed = time.time() - start_time
    de_time = min(10.0, time_budget * 0.5)
    
    if de_time > 2.0:
        try:
            result_de = differential_evolution(
                lambda x: objective_symmetric(x, use_pointy),
                bounds,
                seed=42,
                maxiter=50,
                popsize=15,
                tol=1e-4,
                polish=False,
                update_interval=1
            )
            elapsed = time.time() - start_time
            if result_de.fun < best_val:
                best_val = result_de.fun
                best_x = result_de.x
        except Exception:
            pass
    
    # Phase 2: Local refinement with Nelder-Mead
    elapsed = time.time() - start_time
    nm_time = min(8.0, time_budget - elapsed - 2.0)
    
    if nm_time > 1.0:
        try:
            def obj_nm(x):
                if time.time() - start_time > start_time + time_budget - 1:
                    return 1e10
                return objective_symmetric(x, use_pointy)
            
            result_nm = minimize(
                obj_nm, best_x, method='Nelder-Mead',
                options={'maxiter': 300, 'xatol': 1e-6, 'fatol': 1e-6, 'disp': False}
            )
            if result_nm.fun < best_val:
                best_val = result_nm.fun
                best_x = result_nm.x
        except Exception:
            pass
    
    # Extract best configuration
    r1, r2, ring1_angle, ring2_angle, outer_angle = best_x
    r1 = max(0.1, r1)
    r2 = max(0.1, r2)
    
    centers, angles, templates, oa = generate_symmetric_config(
        r1, r2, ring1_angle, ring2_angle, outer_angle, use_pointy
    )
    
    req_side = compute_required_side(centers, angles, templates, (0.0, 0.0), oa)
    
    # Validate
    if check_validity(centers, angles, templates, req_side, (0.0, 0.0), oa):
        return centers, angles, templates, oa, req_side, True
    else:
        return centers, angles, templates, oa, req_side, False


def generate_lattice_configs():
    """
    Generate diverse hexagonal lattice configurations as fallbacks.
    """
    configs = []
    
    # Pattern 1: 3-4-3-2
    p1 = [
        (0, 0), (1, 0), (2, 0),
        (-1, 1), (0, 1), (1, 1), (2, 1),
        (0, 2), (1, 2), (2, 2),
        (0, 3), (1, 3)
    ]
    
    # Pattern 2: 2-4-4-2
    p2 = [
        (0, 0), (1, 0),
        (-1, 1), (0, 1), (1, 1), (2, 1),
        (-1, 2), (0, 2), (1, 2), (2, 2),
        (0, 3), (1, 3)
    ]
    
    # Pattern 3: 1-3-4-3-1
    p3 = [
        (0, 0),
        (-1, 1), (0, 1), (1, 1),
        (-2, 2), (-1, 2), (0, 2), (1, 2),
        (-1, 3), (0, 3), (1, 3),
        (0, 4)
    ]
    
    # Pattern 4: 4-4-4
    p4 = [
        (-1, 0), (0, 0), (1, 0), (2, 0),
        (-1, 1), (0, 1), (1, 1), (2, 1),
        (-1, 2), (0, 2), (1, 2), (2, 2)
    ]
    
    # Pattern 5: 3-3-3-3
    p5 = [
        (-1, 0), (0, 0), (1, 0),
        (-1, 1), (0, 1), (1, 1),
        (-1, 2), (0, 2), (1, 2),
        (-1, 3), (0, 3), (1, 3)
    ]
    
    patterns = [p1, p2, p3, p4, p5]
    
    spacings = [0.98, 1.0, 1.02]
    rotations = [0.0, 30.0]
    
    for p in patterns:
        for s in spacings:
            for rot in rotations:
                centers = []
                a1 = np.array([np.sqrt(3) * s, 0.0])
                a2 = np.array([np.sqrt(3) / 2.0 * s, 1.5 * s])
                
                for i, j in p:
                    pos = i * a1 + j * a2
                    centers.append((pos[0], pos[1]))
                
                # Center of mass correction
                com = np.mean([c for c in centers], axis=0)
                centered = [(c[0] - com[0], c[1] - com[1]) for c in centers]
                
                angles = [rot] * len(centered)
                templates = [UNIT_VERTS_FT] * len(centered)
                configs.append((centered, angles, templates, 0.0))
    
    return configs


def optimize_lattice_config(centers, angles, templates, outer_angle, time_budget=5.0):
    """
    Optimize a lattice configuration using Nelder-Mead.
    """
    start_time = time.time()
    
    # Flatten parameters: x, y for each hexagon
    x0 = []
    for i in range(len(centers)):
        x0.append(centers[i][0])
        x0.append(centers[i][1])
    
    def obj(x):
        if time.time() - start_time > start_time + time_budget - 1:
            return 1e10
        
        new_centers = []
        for i in range(len(centers)):
            new_centers.append((x[2*i], x[2*i+1]))
        
        req_side = compute_required_side(new_centers, angles, templates, (0.0, 0.0), outer_angle)
        
        # Check for overlaps
        n = len(new_centers)
        all_verts = []
        for i in range(n):
            all_verts.append(get_hex_vertices(new_centers[i], angles[i], templates[i]))
        
        min_sep = float('inf')
        for i in range(n):
            for j in range(i + 1, n):
                sep = sat_min_separation(all_verts[i], all_verts[j])
                if sep < min_sep:
                    min_sep = sep
        
        overlap_penalty = 0.0
        if min_sep < 0:
            overlap_penalty = -min_sep * 100.0
        
        return req_side + overlap_penalty
    
    try:
        result = minimize(
            obj, np.array(x0), method='Nelder-Mead',
            options={'maxiter': 200, 'xatol': 1e-5, 'fatol': 1e-5, 'disp': False}
        )
        
        new_centers = []
        for i in range(len(centers)):
            new_centers.append((result.x[2*i], result.x[2*i+1]))
        
        req_side = compute_required_side(new_centers, angles, templates, (0.0, 0.0), outer_angle)
        
        if check_validity(new_centers, angles, templates, req_side, (0.0, 0.0), outer_angle):
            return new_centers, angles, templates, outer_angle, req_side, True
        else:
            return new_centers, angles, templates, outer_angle, req_side, False
    except Exception:
        return centers, angles, templates, outer_angle, float('inf'), False


def construct():
    start_time = time.time()
    
    best_score = float('inf')
    best_result = None
    
    # Phase 1: Symmetric configuration search
    for use_pointy in [False, True]:
        elapsed = time.time() - start_time
        if elapsed > DEADLINE - 30:
            break
        
        time_budget = min(25.0, DEADLINE - elapsed - 25)
        if time_budget < 5.0:
            continue
        
        try:
            centers, angles, templates, oa, req_side, valid = optimize_symmetric(
                use_pointy, time_budget
            )
            
            if valid and req_side < best_score:
                best_score = req_side
                inner_list = [[centers[i][0], centers[i][1], angles[i]] for i in range(len(centers))]
                best_result = (inner_list, (0.0, 0.0), req_side, oa)
        except Exception:
            pass
    
    # Phase 2: Lattice configuration search (fallback)
    elapsed = time.time() - start_time
    if elapsed < DEADLINE - 20:
        configs = generate_lattice_configs()
        
        # Shuffle for diversity
        np.random.seed(42)
        np.random.shuffle(configs)
        
        for i, config in enumerate(configs):
            elapsed = time.time() - start_time
            if elapsed > DEADLINE - 10:
                break
            
            centers, angles, templates, oa = config
            time_budget = min(5.0, DEADLINE - elapsed - 5)
            
            if time_budget < 1.0:
                continue
            
            try:
                new_centers, new_angles, new_templates, new_oa, req_side, valid = optimize_lattice_config(
                    centers, angles, templates, oa, time_budget
                )
                
                if valid and req_side < best_score:
                    best_score = req_side
                    inner_list = [[new_centers[j][0], new_centers[j][1], new_angles[j]] for j in range(len(new_centers))]
                    best_result = (inner_list, (0.0, 0.0), req_side, new_oa)
            except Exception:
                pass
    
    if best_result is not None:
        return best_result
    
    # Fallback: Simple circular arrangement
    inner_default = []
    for i in range(N):
        theta = 2 * np.pi * i / N
        x = 3.5 * np.cos(theta)
        y = 3.5 * np.sin(theta)
        inner_default.append([x, y, 0.0])
    
    angles_default = [0.0] * N
    templates_default = [UNIT_VERTS_FT] * N
    required_side = compute_required_side(
        [(c[0], c[1]) for c in inner_default],
        angles_default,
        templates_default,
        (0.0, 0.0),
        0.0
    )
    return inner_default, (0.0, 0.0), required_side, 0.0


if __name__ == "__main__":
    result = construct()
    print(f"Best Side: {result[2]}")
