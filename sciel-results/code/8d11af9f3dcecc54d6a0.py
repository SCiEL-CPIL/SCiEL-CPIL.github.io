"""
Hexagon packing: 12 unit hexagons in a regular hexagon, minimize outer side length.
Strategy: Formulate as low-dimensional global optimization over rigid honeycomb cluster
+ global rotation + outer orientation. Use multi-start Nelder-Mead with diverse
honeycomb layouts. The cluster is a rigid body (all hexagons share same orientation),
reducing the problem to 3-4 DOF optimization.
"""
import numpy as np
import time
from scipy.optimize import minimize

N = 12
DEADLINE = 148.0  # Safety margin before 150s kill

# Unit hexagon vertices (pointy-top, side length 1)
# Vertices at angles 30, 90, 150, 210, 270, 330 degrees
UNIT_VERTS = np.array([
    [np.cos(np.radians(60 * i + 30)), np.sin(np.radians(60 * i + 30))] for i in range(6)
])

# Edge normals (outward) for containment check
# For a regular hexagon centered at origin, the 6 supporting lines have normals at 0,60,...,300
EDGE_NORMALS = np.array([
    [np.cos(np.radians(60 * i)), np.sin(np.radians(60 * i))] for i in range(6)
])


def get_cluster_vertices(local_centers, global_pos, global_angle_deg):
    """
    Compute global vertices for a rigid cluster.
    local_centers: (N, 2) array of local positions (relative to cluster center)
    global_pos: (2,) array
    global_angle_deg: float
    Returns: (N, 6, 2) array of global vertices
    """
    a_rad = np.radians(global_angle_deg)
    c = np.cos(a_rad)
    s = np.sin(a_rad)
    R = np.array([[c, -s], [s, c]])
    
    # Rotate local centers
    rotated_centers = local_centers @ R.T + global_pos
    
    # Generate vertices
    verts = np.zeros((N, 6, 2))
    for i in range(N):
        verts[i] = UNIT_VERTS @ R.T + rotated_centers[i]
    return verts


def compute_required_side(verts, outer_center, outer_angle_deg):
    """
    Compute minimum outer side length to contain vertices.
    """
    oa_rad = np.radians(outer_angle_deg)
    c_o = np.cos(-oa_rad)
    s_o = np.sin(-oa_rad)
    R_o = np.array([[c_o, -s_o], [s_o, c_o]])
    
    max_extent = 0.0
    # Flatten vertices to (N*6, 2) for faster processing
    flat_verts = verts.reshape(-1, 2)
    v_rel = flat_verts - outer_center
    v_rot = v_rel @ R_o.T
    projs = v_rot @ EDGE_NORMALS.T
    abs_projs = np.abs(projs)
    max_proj = np.max(abs_projs, axis=1)
    max_extent = np.max(max_proj)
    
    # Apothem = side * sqrt(3)/2 => side = apothem / (sqrt(3)/2)
    return max_extent / (np.sqrt(3) / 2)


def check_cluster_validity(local_centers, global_pos, global_angle_deg, outer_center, outer_side, outer_angle_deg, tol=1e-6):
    """
    Check if the rigid cluster is valid:
    1. No internal overlaps (pre-computed for base layout, but check just in case)
    2. Containment in outer hexagon
    """
    verts = get_cluster_vertices(local_centers, global_pos, global_angle_deg)
    
    # 1. Check Internal Overlaps (SAT)
    for i in range(N):
        v1 = verts[i]
        for j in range(i + 1, N):
            v2 = verts[j]
            edges1 = np.roll(v1, -1, axis=0) - v1
            edges2 = np.roll(v2, -1, axis=0) - v2
            axes = np.vstack([edges1, edges2])
            norms = np.linalg.norm(axes, axis=1, keepdims=True)
            axes = axes / (norms + 1e-15)
            
            proj1 = v1 @ axes.T
            proj2 = v2 @ axes.T
            
            sep = np.minimum(proj1.min(axis=0) - proj2.max(axis=0),
                            proj2.min(axis=0) - proj1.max(axis=0))
            
            if np.max(sep) < -tol:
                return False
    
    # 2. Check Containment
    apothem = outer_side * np.sqrt(3) / 2
    oa_rad = np.radians(outer_angle_deg)
    c_o = np.cos(-oa_rad)
    s_o = np.sin(-oa_rad)
    R_o = np.array([[c_o, -s_o], [s_o, c_o]])
    
    flat_verts = verts.reshape(-1, 2)
    v_rel = flat_verts - outer_center
    v_rot = v_rel @ R_o.T
    projs = v_rot @ EDGE_NORMALS.T
    abs_projs = np.abs(projs)
    max_proj = np.max(abs_projs, axis=1)
    
    if np.max(max_proj) > apothem + tol:
        return False
    
    return True


def generate_honeycomb_layouts():
    """
    Generate diverse compact honeycomb cluster layouts for 12 hexagons.
    Each layout is a set of axial coordinates that form a valid non-overlapping cluster.
    Returns list of (N, 2) arrays of local centers (centered at origin).
    """
    a1 = np.array([np.sqrt(3), 0])
    a2 = np.array([np.sqrt(3) / 2, 1.5])
    
    layouts = []
    
    # Layout 1: Center + 6 ring + 5 from ring 2 (compact hexagonal)
    coords1 = [
        (0, 0),
        (1, 0), (-1, 0), (0, 1), (0, -1), (1, -1), (-1, 1),
        (2, -1), (-2, 1), (0, 2), (0, -2), (2, 0)
    ]
    
    # Layout 2: 3-4-3-2 rows
    coords2 = [
        (0, 0),
        (-1, 0), (1, 0), (-1, 1), (0, 1), (1, 1),
        (-2, 1), (-2, 2), (-1, 2), (0, 2), (1, 2),
        (-1, 3), (0, 3)
    ]
    
    # Layout 3: 2-3-4-3 rows
    coords3 = [
        (0, 0), (1, 0),
        (-1, 1), (0, 1), (1, 1),
        (-2, 2), (-1, 2), (0, 2), (1, 2),
        (-1, 3), (0, 3), (1, 3)
    ]
    
    # Layout 4: 1-3-4-3-1 (diamond-like)
    coords4 = [
        (0, 0),
        (-1, 1), (0, 1), (1, 1),
        (-2, 2), (-1, 2), (0, 2), (1, 2),
        (-1, 3), (0, 3), (1, 3),
        (0, 4)
    ]
    
    # Layout 5: 4-4-4 (three rows of 4)
    coords5 = [
        (-1, 0), (0, 0), (1, 0), (2, 0),
        (-1, 1), (0, 1), (1, 1), (2, 1),
        (-1, 2), (0, 2), (1, 2), (2, 2)
    ]
    
    # Layout 6: 2-4-4-2
    coords6 = [
        (0, 0), (1, 0),
        (-1, 1), (0, 1), (1, 1), (2, 1),
        (-1, 2), (0, 2), (1, 2), (2, 2),
        (0, 3), (1, 3)
    ]
    
    # Layout 7: 3-3-3-3 (four rows of 3, staggered)
    coords7 = [
        (0, 0), (1, 0), (2, 0),
        (0, 1), (1, 1), (2, 1),
        (0, 2), (1, 2), (2, 2),
        (0, 3), (1, 3), (2, 3)
    ]
    
    # Layout 8: 2-3-3-2-2
    coords8 = [
        (0, 0), (1, 0),
        (-1, 1), (0, 1), (1, 1),
        (-2, 2), (-1, 2), (0, 2),
        (-1, 3), (0, 3), (1, 3)
    ]
    
    # Layout 9: Asymmetric compact
    coords9 = [
        (0, 0),
        (1, 0), (-1, 0), (0, 1), (0, -1), (1, -1), (-1, 1),
        (2, 0), (-2, 0), (1, 1), (-1, -1), (2, -1)
    ]
    
    # Layout 10: Another asymmetric
    coords10 = [
        (0, 0),
        (1, 0), (-1, 0), (0, 1), (0, -1), (1, -1), (-1, 1),
        (2, -1), (-2, 1), (1, 2), (-1, -2), (0, 2)
    ]
    
    for coords in [coords1, coords2, coords3, coords4, coords5, coords6, coords7, coords8, coords9, coords10]:
        local_centers = []
        for q, r in coords:
            x = np.sqrt(3) * (q + r / 2.0)
            y = 1.5 * r
            local_centers.append([x, y])
        
        local_centers = np.array(local_centers)
        
        # Verify no overlaps
        valid = True
        for i in range(N):
            for j in range(i + 1, N):
                dist = np.linalg.norm(local_centers[i] - local_centers[j])
                if dist < np.sqrt(3) - 1e-9:
                    valid = False
                    break
            if not valid:
                break
        
        if valid:
            # Center the layout around origin
            cx = np.mean(local_centers[:, 0])
            cy = np.mean(local_centers[:, 1])
            local_centers -= np.array([cx, cy])
            layouts.append(local_centers)
    
    return layouts


def objective_3d(params, local_centers, outer_center, outer_angle_deg):
    """
    Objective: minimize outer side length.
    params: [x, y, angle_deg] - cluster position and rotation
    """
    x, y, angle = params
    verts = get_cluster_vertices(local_centers, np.array([x, y]), angle)
    side = compute_required_side(verts, outer_center, outer_angle_deg)
    return side


def objective_4d(params, local_centers, outer_center):
    """
    Objective: minimize outer side length.
    params: [x, y, angle_deg, outer_angle_deg]
    """
    x, y, angle, outer_angle = params
    verts = get_cluster_vertices(local_centers, np.array([x, y]), angle)
    side = compute_required_side(verts, outer_center, outer_angle)
    return side


def construct():
    start_time = time.time()
    np.random.seed(42)
    
    outer_center = np.array([0.0, 0.0])
    
    # Generate diverse honeycomb layouts
    layouts = generate_honeycomb_layouts()
    
    if not layouts:
        # Fallback: simple grid
        a1 = np.array([np.sqrt(3), 0])
        a2 = np.array([np.sqrt(3) / 2, 1.5])
        coords = [(i, j) for i in range(3) for j in range(4)]
        local_centers = np.array([[np.sqrt(3) * (q + r/2.0), 1.5*r] for q, r in coords])
        local_centers -= local_centers.mean(axis=0)
        layouts = [local_centers]
    
    best_side = float('inf')
    best_params_3d = None
    best_layout_idx = 0
    best_outer_angle = 0.0
    
    # Phase 1: For each layout, do coarse sampling + local refinement
    for layout_idx, local_centers in enumerate(layouts):
        if time.time() > DEADLINE - 20:
            break
        
        # Coarse random sampling of (x, y, angle)
        num_samples = 500
        best_local_side = float('inf')
        best_local_params = None
        
        for i in range(num_samples):
            if time.time() > DEADLINE - 15:
                break
            
            x = np.random.uniform(-1.5, 1.5)
            y = np.random.uniform(-1.5, 1.5)
            angle = np.random.uniform(0, 60)  # Hex symmetry
            
            params = np.array([x, y, angle])
            side = objective_3d(params, local_centers, outer_center, 0.0)
            
            if side < best_local_side:
                best_local_side = side
                best_local_params = params
        
        # Local refinement with Nelder-Mead
        if best_local_params is not None:
            res = minimize(
                objective_3d,
                best_local_params,
                args=(local_centers, outer_center, 0.0),
                method='Nelder-Mead',
                options={'maxiter': 500, 'xatol': 1e-6, 'fatol': 1e-6}
            )
            if res.fun < best_local_side:
                best_local_side = res.fun
                best_local_params = res.x
        
        # Also try optimizing outer angle jointly
        if best_local_params is not None and time.time() < DEADLINE - 10:
            params_4d = np.append(best_local_params, 0.0)
            res4 = minimize(
                objective_4d,
                params_4d,
                args=(local_centers, outer_center),
                method='Nelder-Mead',
                options={'maxiter': 300, 'xatol': 1e-5, 'fatol': 1e-5}
            )
            if res4.fun < best_local_side:
                best_local_side = res4.fun
                best_local_params = res4.x[:3]
                best_outer_angle_for_this = res4.x[3]
            else:
                best_outer_angle_for_this = 0.0
        
        # Update global best
        if best_local_side < best_side:
            best_side = best_local_side
            best_params_3d = best_local_params
            best_layout_idx = layout_idx
            best_outer_angle = best_outer_angle_for_this if best_local_side < best_side else 0.0
    
    # Phase 2: Multi-start refinement on the best layout
    local_centers = layouts[best_layout_idx]
    
    if best_params_3d is not None:
        # Try multiple random restarts + refinement
        for restart in range(8):
            if time.time() > DEADLINE - 5:
                break
            
            x = np.random.uniform(-1.5, 1.5)
            y = np.random.uniform(-1.5, 1.5)
            angle = np.random.uniform(0, 60)
            params = np.array([x, y, angle])
            
            res = minimize(
                objective_3d,
                params,
                args=(local_centers, outer_center, best_outer_angle),
                method='Nelder-Mead',
                options={'maxiter': 400, 'xatol': 1e-6, 'fatol': 1e-6}
            )
            if res.fun < best_side:
                best_side = res.fun
                best_params_3d = res.x
        
        # Final joint optimization of all 4 parameters
        if time.time() < DEADLINE - 3:
            params_4d = np.append(best_params_3d, best_outer_angle)
            res_final = minimize(
                objective_4d,
                params_4d,
                args=(local_centers, outer_center),
                method='Nelder-Mead',
                options={'maxiter': 500, 'xatol': 1e-7, 'fatol': 1e-7}
            )
            if res_final.fun < best_side:
                best_side = res_final.fun
                best_params_3d = res_final.x[:3]
                best_outer_angle = res_final.x[3]
    
    # Final validation and construction
    if best_params_3d is None:
        best_params_3d = np.array([0.0, 0.0, 0.0])
        best_side = objective_3d(best_params_3d, local_centers, outer_center, best_outer_angle)
    
    x, y, angle = best_params_3d
    
    # Calculate final vertices to return
    verts = get_cluster_vertices(local_centers, np.array([x, y]), angle)
    
    # Extract centers and angles for the return format
    a_rad = np.radians(angle)
    c = np.cos(a_rad)
    s = np.sin(a_rad)
    R = np.array([[c, -s], [s, c]])
    
    global_centers = local_centers @ R.T + np.array([x, y])
    
    inner_list = []
    for i in range(N):
        inner_list.append([global_centers[i, 0], global_centers[i, 1], angle])
    
    # Double check validity
    if not check_cluster_validity(local_centers, np.array([x, y]), angle, outer_center, best_side, best_outer_angle):
        # If invalid, try a safe fallback
        inner_list = []
        for i in range(N):
            theta = 2 * np.pi * i / N
            x_f = 3.5 * np.cos(theta)
            y_f = 3.5 * np.sin(theta)
            inner_list.append([x_f, y_f, 0.0])
        best_side = compute_required_side(
            get_cluster_vertices(np.array([[x_f, y_f] for x_f, y_f, _ in inner_list]), np.array([0, 0]), 0),
            outer_center, best_outer_angle
        )
    
    return inner_list, (0.0, 0.0), best_side, best_outer_angle


if __name__ == "__main__":
    result = construct()
    print(f"Best outer side length: {result[2]:.6f}")
    print(f"Number of inner hexagons: {len(result[0])}")
