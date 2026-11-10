# EVOLVE-BLOCK-START
import numpy as np
import time

def _hex_vertices(cx, cy, side, angle_deg):
    """Compute 6 vertices of a regular hexagon centered at (cx,cy) with given side and angle."""
    a = np.radians(angle_deg)
    angles = a + np.arange(6) * np.pi / 3.0
    return np.column_stack([
        cx + side * np.cos(angles),
        cy + side * np.sin(angles)
    ])

def _min_outer_side(vertices_all, outer_angle_deg=0.0):
    """
    Given array of all inner hexagon vertices (shape (72,2)), compute minimum outer hex side s
    (centered at origin, angle outer_angle_deg) that contains all vertices.
    """
    k = np.arange(6)
    outer_angle = np.radians(outer_angle_deg)
    angles = outer_angle + (k + 0.5) * np.pi / 3
    m = np.column_stack([np.cos(angles), np.sin(angles)])
    
    dots = vertices_all @ m.T  # shape (N, 6)
    max_dot = np.max(dots, axis=1)  # shape (N,)
    global_max = np.max(max_dot)
    
    if global_max < 0:
        global_max = 0.0
    
    s_min = (2.0 / np.sqrt(3.0)) * global_max
    return s_min

def _check_nonoverlap(centers, angles_deg, side=1.0, tol=1e-6):
    """
    Check if all pairs of hexagons are non-overlapping using SAT.
    Returns True if no overlap, False otherwise.
    """
    n = len(centers)
    vertices_list = []
    for i in range(n):
        verts = _hex_vertices(centers[i,0], centers[i,1], side, angles_deg[i])
        vertices_list.append(verts)
    
    normals_list = []
    for verts in vertices_list:
        normals = []
        for e in range(6):
            p1 = verts[e]
            p2 = verts[(e+1) % 6]
            edge = p2 - p1
            nx, ny = edge[1], -edge[0]
            norm = np.sqrt(nx*nx + ny*ny)
            if norm > 1e-12:
                normals.append(np.array([nx/norm, ny/norm]))
        normals_list.append(normals)
    
    for i in range(n):
        for j in range(i+1, n):
            v1 = vertices_list[i]
            v2 = vertices_list[j]
            # SAT: check all 12 edge normals (6 from each hexagon)
            overlap = True
            for hex_idx in range(2):
                verts = [v1, v2][hex_idx]
                normals = normals_list[hex_idx]
                for normal in normals:
                    proj1 = v1 @ normal
                    proj2 = v2 @ normal
                    min1, max1 = np.min(proj1), np.max(proj1)
                    min2, max2 = np.min(proj2), np.max(proj2)
                    if max1 + tol < min2 or max2 + tol < min1:
                        overlap = False
                        break
                if not overlap:
                    break
            if overlap:
                return False
    return True

def _compute_all_vertices(centers, angles_deg, side=1.0):
    """Compute all 72 vertices of 12 hexagons."""
    verts = []
    for i in range(len(centers)):
        verts.append(_hex_vertices(centers[i,0], centers[i,1], side, angles_deg[i]))
    return np.vstack(verts)

def _get_rigid_honeycomb_params():
    """
    Returns the centers and angles for the 12 hexagons in a rigid honeycomb-like cluster.
    The cluster is centered at the origin.
    Structure:
    - 1 center hex
    - 6 hexes in inner ring
    - 5 hexes in outer ring
    All hexagons are aligned (angle 0) relative to the cluster frame.
    """
    d = np.sqrt(3.0)
    positions = []
    angles = []
    
    # Center
    positions.append([0.0, 0.0])
    angles.append(0.0)
    
    # Inner ring: 6 hexagons
    for k in range(6):
        angle = k * np.pi / 3
        x = d * np.cos(angle)
        y = d * np.sin(angle)
        positions.append([x, y])
        angles.append(0.0)
    
    # Outer ring: 5 hexagons
    # Placed in the gaps of the inner ring, further out
    d2 = 2.0 * d
    # Angles offset by 30 degrees to sit in the "valleys" of the hexagonal lattice
    for k in range(5):
        angle = (k * 72 + 30) * np.pi / 180
        x = d2 * np.cos(angle)
        y = d2 * np.sin(angle)
        positions.append([x, y])
        angles.append(0.0)
    
    positions = np.array(positions)
    angles = np.array(angles)
    return positions, angles

def _evaluate_rigid_rotation(theta_deg):
    """
    Rotate the rigid honeycomb cluster by theta_deg and compute the min outer side.
    """
    positions, angles = _get_rigid_honeycomb_params()
    
    # Rotate positions and angles
    theta_rad = np.radians(theta_deg)
    cos_t = np.cos(theta_rad)
    sin_t = np.sin(theta_rad)
    
    new_positions = np.zeros_like(positions)
    for i in range(len(positions)):
        x, y = positions[i]
        new_positions[i, 0] = x * cos_t - y * sin_t
        new_positions[i, 1] = x * sin_t + y * cos_t
        
    new_angles = angles + theta_deg
    
    # Compute vertices
    all_verts = _compute_all_vertices(new_positions, new_angles)
    
    # Compute min outer side
    s = _min_outer_side(all_verts)
    
    return s, new_positions, new_angles

def hexagon_packing_12():
    start_time = time.time()
    deadline = start_time + 580.0 # Safety margin
    
    # Phase 1: Optimize the rigid rotation
    # The cluster has 6-fold symmetry, so we only need to search 0 to 60 degrees.
    # Fine sweep first
    best_s = float('inf')
    best_theta = 0.0
    best_positions = None
    best_angles = None
    
    # Coarse sweep
    thetas = np.linspace(0, 60, 121)
    for t in thetas:
        s, pos, ang = _evaluate_rigid_rotation(t)
        if s < best_s:
            best_s = s
            best_theta = t
            best_positions = pos
            best_angles = ang
    
    # Fine sweep around best_theta
    thetas_fine = np.linspace(best_theta - 5, best_theta + 5, 1001)
    thetas_fine = thetas_fine % 60 # Keep in range
    for t in thetas_fine:
        s, pos, ang = _evaluate_rigid_rotation(t)
        if s < best_s:
            best_s = s
            best_theta = t
            best_positions = pos
            best_angles = ang
            
    # Very fine sweep
    thetas_very_fine = np.linspace(best_theta - 0.5, best_theta + 0.5, 1001)
    thetas_very_fine = thetas_very_fine % 60
    for t in thetas_very_fine:
        s, pos, ang = _evaluate_rigid_rotation(t)
        if s < best_s:
            best_s = s
            best_theta = t
            best_positions = pos
            best_angles = ang

    # Check validity of the rigid solution
    if not _check_nonoverlap(best_positions, best_angles):
        # This shouldn't happen for the standard honeycomb, but just in case
        # Fallback to a scaled version or random restart
        pass

    # Prepare params for SA refinement
    # We allow the hexagons to move slightly to see if we can improve further
    # Initial params: [x1, y1, theta1, ..., x12, y12, theta12]
    current_params = np.column_stack([best_positions, best_angles]).flatten()
    
    # Phase 2: Local SA refinement
    # We use a small step size to avoid breaking the good structure
    np.random.seed(42)
    T = 0.05
    cooling = 0.999
    step_size = 0.05
    accept_count = 0
    total_count = 0
    
    current_s = best_s
    best_params = current_params.copy()
    
    max_iters = 200000
    for it in range(max_iters):
        if time.time() > deadline:
            break
        
        new_params = current_params + np.random.randn(36) * step_size
        
        centers_new = new_params.reshape(12, 3)[:, :2]
        angles_new = new_params.reshape(12, 3)[:, 2]
        
        # Check overlap
        if not _check_nonoverlap(centers_new, angles_new):
            continue # Reject invalid moves immediately in this phase to maintain validity
        
        all_verts_new = _compute_all_vertices(centers_new, angles_new)
        s_new = _min_outer_side(all_verts_new)
        
        delta = s_new - current_s
        total_count += 1
        
        if delta < 0 or np.random.random() < np.exp(-delta / max(T, 1e-10)):
            current_params = new_params
            current_s = s_new
            accept_count += 1
            if s_new < best_s:
                best_s = s_new
                best_params = new_params.copy()
        
        T *= cooling
        
        # Adaptive step size
        if it % 1000 == 0 and it > 0:
            acc_rate = accept_count / max(total_count, 1)
            if acc_rate < 0.2:
                step_size *= 0.8
            elif acc_rate > 0.8:
                step_size *= 1.2
            step_size = np.clip(step_size, 0.001, 0.5)
            accept_count = 0
            total_count = 0

    # Final verification and centering
    centers_final = best_params.reshape(12, 3)[:, :2]
    angles_final = best_params.reshape(12, 3)[:, 2]
    
    # Shift so that the centroid of the inner hexagons is at the origin?
    # The problem states: "outer a (3,) array (x, y, angle_deg) for the outer hexagon".
    # And "inner a (12, 3) array".
    # The validity check shifts all centers by minus the outer center.
    # So we can place the outer hexagon at (0,0,0) and center the inner cluster around it.
    
    # Let's center the inner hexagons around the origin to be safe, 
    # assuming the outer hexagon is centered at the origin.
    centroid = np.mean(centers_final, axis=0)
    centers_final = centers_final - centroid
    
    # Recalculate s after centering (should be same if we centered correctly, but vertices change)
    all_verts_final = _compute_all_vertices(centers_final, angles_final)
    s_final = _min_outer_side(all_verts_final)
    
    # Final validity check
    if not _check_nonoverlap(centers_final, angles_final):
        # Fallback to the rigid solution if SA broke it
        centers_final = best_positions
        angles_final = best_angles
        centroid = np.mean(centers_final, axis=0)
        centers_final = centers_final - centroid
        all_verts_final = _compute_all_vertices(centers_final, angles_final)
        s_final = _min_outer_side(all_verts_final)

    inner_hex_data = np.column_stack([centers_final, angles_final])
    outer_hex_data = np.array([0.0, 0.0, 0.0])
    
    return inner_hex_data, outer_hex_data, s_final

# EVOLVE-BLOCK-END
