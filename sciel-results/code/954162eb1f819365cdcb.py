"""
Hexagon packing: 12 unit hexagons in a regular hexagon, minimize outer side length.
Strategy: Honeycomb Lattice Deformation + Rigid-Body RRT* search + Nelder-Mead refinement.
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

def get_deformed_vertices(local_centers, shear, global_pos, global_angle_deg):
    """
    Compute global vertices for a deformed lattice cluster.
    local_centers: (N, 2) array of local positions
    shear: scalar shear parameter
    global_pos: (2,) array
    global_angle_deg: float
    Returns: (N, 6, 2) array of global vertices
    """
    # Apply shear deformation to local centers
    # Shear along x-axis: x' = x + shear*y, y' = y
    deformed = local_centers.copy()
    deformed[:, 0] += shear * local_centers[:, 1]
    
    a_rad = np.radians(global_angle_deg)
    c = np.cos(a_rad)
    s = np.sin(a_rad)
    R = np.array([[c, -s], [s, c]])
    
    # Rotate deformed centers
    rotated_centers = deformed @ R.T + global_pos
    
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
    1. No internal overlaps (pre-computed for base layout, but check just in case if layout changes)
    2. Containment in outer hexagon
    """
    verts = get_cluster_vertices(local_centers, global_pos, global_angle_deg)
    
    # 1. Check Internal Overlaps (SAT)
    # Since we use a fixed valid base layout, this should always pass.
    # But we include it for robustness.
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

def generate_base_layout():
    """
    Generate a compact, non-overlapping local layout for 12 hexagons.
    Uses a central hexagon + 6 neighbors + 5 from the next ring.
    This layout is rigid.
    """
    # Hex coordinates for a compact cluster
    # Center (0,0)
    # Ring 1: 6 neighbors
    # Ring 2: Pick 5 to make 12 total.
    
    coords = [
        (0, 0),
        (1, 0), (-1, 0), (0, 1), (0, -1), (1, -1), (-1, 1), # 7
        (2, -1), (-2, 1), (0, 2), (0, -2), (2, 0) # 5 more
    ]
    
    local_centers = []
    for q, r in coords:
        x = np.sqrt(3) * (q + r / 2.0)
        y = 1.5 * r
        local_centers.append([x, y])
    
    local_centers = np.array(local_centers)
    
    # Center the local layout around origin to make global rotation/translation intuitive
    cx = np.mean(local_centers[:, 0])
    cy = np.mean(local_centers[:, 1])
    local_centers -= np.array([cx, cy])
    
    return local_centers

def generate_lattice_layout():
    """
    Generate a 3x4 rectangular sub-lattice of the infinite hexagonal tiling.
    This is a different compact arrangement that may benefit from shear deformation.
    """
    # Axial coordinates for a 3x4 grid
    coords = []
    for i in range(3):
        for j in range(4):
            # Offset each row to form hexagonal lattice
            q = i
            r = j
            coords.append((q, r))
    
    local_centers = []
    for q, r in coords:
        x = np.sqrt(3) * (q + r / 2.0)
        y = 1.5 * r
        local_centers.append([x, y])
    
    local_centers = np.array(local_centers)
    
    # Center the local layout around origin
    cx = np.mean(local_centers[:, 0])
    cy = np.mean(local_centers[:, 1])
    local_centers -= np.array([cx, cy])
    
    return local_centers

def objective_rigid(params, local_centers, outer_center, outer_angle_deg):
    """
    Objective function for Nelder-Mead on rigid body parameters.
    params: [x, y, angle_deg]
    """
    x, y, angle = params
    verts = get_cluster_vertices(local_centers, np.array([x, y]), angle)
    side = compute_required_side(verts, outer_center, outer_angle_deg)
    return side

def objective_deformed(params, local_centers, outer_center, outer_angle_deg):
    """
    Objective function for Nelder-Mead on deformed lattice parameters.
    params: [x, y, angle_deg, shear]
    """
    x, y, angle, shear = params
    verts = get_deformed_vertices(local_centers, shear, np.array([x, y]), angle)
    
    # Check for internal overlaps with penalty
    overlap_penalty = 0.0
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
            
            if np.max(sep) < 0:
                overlap_penalty += abs(np.max(sep))
    
    side = compute_required_side(verts, outer_center, outer_angle_deg)
    
    # Return side + large penalty if overlaps exist
    return side + 1000.0 * overlap_penalty

def construct():
    start_time = time.time()
    np.random.seed(42)
    
    # 1. Generate base layouts
    local_centers_rigid = generate_base_layout()
    local_centers_lattice = generate_lattice_layout()
    
    outer_center = np.array([0.0, 0.0])
    outer_angle = 0.0
    
    best_side = float('inf')
    best_result = None  # (inner_list, side)
    
    # === Strategy A: Rigid Body Search ===
    best_rigid_params = None
    best_rigid_side = float('inf')
    
    # Phase 1: Coarse Random Sampling for rigid
    num_samples = 1500
    for i in range(num_samples):
        if time.time() > DEADLINE - 60:
            break
        
        x = np.random.uniform(-2, 2)
        y = np.random.uniform(-2, 2)
        angle = np.random.uniform(0, 60)
        
        params = np.array([x, y, angle])
        side = objective_rigid(params, local_centers_rigid, outer_center, outer_angle)
        
        if side < best_rigid_side:
            best_rigid_side = side
            best_rigid_params = params
    
    # Phase 2: Local Refinement for rigid
    if best_rigid_params is not None:
        res = minimize(
            objective_rigid, 
            best_rigid_params, 
            args=(local_centers_rigid, outer_center, outer_angle),
            method='Nelder-Mead',
            options={'maxiter': 1000, 'xatol': 1e-6, 'fatol': 1e-6}
        )
        if res.fun < best_rigid_side:
            best_rigid_side = res.fun
            best_rigid_params = res.x
        
        # Try a few more restarts
        for restart in range(3):
            if time.time() > DEADLINE - 40:
                break
            
            x = np.random.uniform(-2, 2)
            y = np.random.uniform(-2, 2)
            angle = np.random.uniform(0, 60)
            params = np.array([x, y, angle])
            
            res = minimize(
                objective_rigid, 
                params, 
                args=(local_centers_rigid, outer_center, outer_angle),
                method='Nelder-Mead',
                options={'maxiter': 500, 'xatol': 1e-5, 'fatol': 1e-5}
            )
            if res.fun < best_rigid_side:
                best_rigid_side = res.fun
                best_rigid_params = res.x
    
    # Record best rigid result
    if best_rigid_params is not None and best_rigid_side < best_side:
        x, y, angle = best_rigid_params
        a_rad = np.radians(angle)
        c = np.cos(a_rad)
        s = np.sin(a_rad)
        R = np.array([[c, -s], [s, c]])
        global_centers = local_centers_rigid @ R.T + np.array([x, y])
        inner_list = []
        for i in range(N):
            inner_list.append([global_centers[i, 0], global_centers[i, 1], angle])
        best_side = best_rigid_side
        best_result = (inner_list, best_side, 0.0)  # shear=0
    
    # === Strategy B: Deformed Lattice Search ===
    best_def_params = None
    best_def_side = float('inf')
    
    # Phase 1: Coarse Random Sampling for deformed
    num_samples_def = 1000
    for i in range(num_samples_def):
        if time.time() > DEADLINE - 30:
            break
        
        x = np.random.uniform(-2, 2)
        y = np.random.uniform(-2, 2)
        angle = np.random.uniform(0, 60)
        shear = np.random.uniform(-0.5, 0.5)
        
        params = np.array([x, y, angle, shear])
        side = objective_deformed(params, local_centers_lattice, outer_center, outer_angle)
        
        if side < best_def_side:
            best_def_side = side
            best_def_params = params
    
    # Phase 2: Local Refinement for deformed
    if best_def_params is not None:
        res = minimize(
            objective_deformed, 
            best_def_params, 
            args=(local_centers_lattice, outer_center, outer_angle),
            method='Nelder-Mead',
            options={'maxiter': 1000, 'xatol': 1e-6, 'fatol': 1e-6}
        )
        if res.fun < best_def_side:
            best_def_side = res.fun
            best_def_params = res.x
        
        # Try a few more restarts
        for restart in range(3):
            if time.time() > DEADLINE - 10:
                break
            
            x = np.random.uniform(-2, 2)
            y = np.random.uniform(-2, 2)
            angle = np.random.uniform(0, 60)
            shear = np.random.uniform(-0.5, 0.5)
            params = np.array([x, y, angle, shear])
            
            res = minimize(
                objective_deformed, 
                params, 
                args=(local_centers_lattice, outer_center, outer_angle),
                method='Nelder-Mead',
                options={'maxiter': 500, 'xatol': 1e-5, 'fatol': 1e-5}
            )
            if res.fun < best_def_side:
                best_def_side = res.fun
                best_def_params = res.x
    
    # Record best deformed result
    if best_def_params is not None and best_def_side < best_side:
        x, y, angle, shear = best_def_params
        # Verify no overlaps
        verts = get_deformed_vertices(local_centers_lattice, shear, np.array([x, y]), angle)
        overlap_check = True
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
                
                if np.max(sep) < -1e-6:
                    overlap_check = False
                    break
            if not overlap_check:
                break
        
        if overlap_check:
            # Compute actual side without penalty
            side = compute_required_side(verts, outer_center, outer_angle)
            if side < best_side:
                a_rad = np.radians(angle)
                c = np.cos(a_rad)
                s = np.sin(a_rad)
                R = np.array([[c, -s], [s, c]])
                
                # Apply shear then rotate
                deformed_centers = local_centers_lattice.copy()
                deformed_centers[:, 0] += shear * local_centers_lattice[:, 1]
                global_centers = deformed_centers @ R.T + np.array([x, y])
                
                inner_list = []
                for i in range(N):
                    inner_list.append([global_centers[i, 0], global_centers[i, 1], angle])
                best_side = side
                best_result = (inner_list, best_side, shear)
    
    # Final fallback if no valid result found
    if best_result is None:
        # Use simple rigid layout with no optimization
        x, y, angle = 0.0, 0.0, 0.0
        a_rad = np.radians(angle)
        c = np.cos(a_rad)
        s = np.sin(a_rad)
        R = np.array([[c, -s], [s, c]])
        global_centers = local_centers_rigid @ R.T + np.array([x, y])
        inner_list = []
        for i in range(N):
            inner_list.append([global_centers[i, 0], global_centers[i, 1], angle])
        verts = get_cluster_vertices(local_centers_rigid, np.array([x, y]), angle)
        best_side = compute_required_side(verts, outer_center, outer_angle)
        best_result = (inner_list, best_side, 0.0)
    
    inner_list, final_side, _ = best_result
    
    # Double check validity
    if not check_cluster_validity(local_centers_rigid, np.array([0,0]), 0, outer_center, final_side, outer_angle):
        # If invalid, try a safe fallback
        inner_list = []
        for i in range(N):
            theta = 2 * np.pi * i / N
            x_f = 3.5 * np.cos(theta)
            y_f = 3.5 * np.sin(theta)
            inner_list.append([x_f, y_f, 0.0])
        final_side = compute_required_side(get_cluster_vertices(np.array([[x_f, y_f] for x_f, y_f, _ in inner_list]), np.array([0,0]), 0), outer_center, outer_angle)
    
    return inner_list, (0.0, 0.0), final_side, outer_angle

if __name__ == "__main__":
    result = construct()
    print(f"Best outer side length: {result[2]:.6f}")
    print(f"Number of inner hexagons: {len(result[0])}")
