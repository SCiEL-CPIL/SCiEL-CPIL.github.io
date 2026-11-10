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
    Given array of all inner hexagon vertices (shape (N, 2)), compute minimum outer hex side s
    (centered at origin, with given angle) that contains all vertices.
    """
    k = np.arange(6)
    base_angle = np.radians(outer_angle_deg)
    # The apothem direction angles for the outer hexagon
    # The condition for a point (x,y) to be inside a hexagon with side s and angle a
    # is max over i of dot((x,y), n_i) <= s * apothem_factor
    # apothem = s * sqrt(3)/2
    # So s >= max(dot) / (sqrt(3)/2) = 2/sqrt(3) * max(dot)
    angles = base_angle + (k + 0.5) * np.pi / 3
    m = np.column_stack([np.cos(angles), np.sin(angles)])
    
    dots = vertices_all @ m.T
    max_dot = np.max(dots, axis=1)
    global_max = np.max(max_dot)
    
    if global_max < 0:
        global_max = 0.0
    
    s_min = (2.0 / np.sqrt(3.0)) * global_max
    return s_min

def _check_nonoverlap(centers, side=1.0, tol=1e-6):
    """
    Check if all pairs of hexagons are non-overlapping using vectorized SAT.
    Returns True if no overlap, False otherwise.
    Assumes all hexagons have angle 0.
    """
    n = len(centers)
    if n == 0:
        return True
    
    # Precompute vertices for all hexagons (angle 0)
    all_verts = np.zeros((n, 6, 2))
    for i in range(n):
        all_verts[i] = _hex_vertices(centers[i,0], centers[i,1], side, 0.0)
        
    # Normals for angle 0 hexagon
    # Edges are at 0, 60, ... degrees? No, vertices are at 0, 60...
    # Edge 0 connects v0 and v1.
    # v0 = (1, 0), v1 = (0.5, sqrt(3)/2)
    # Edge vector = (-0.5, sqrt(3)/2). Normal = (sqrt(3)/2, 0.5) which is angle 30 deg.
    # Actually, for a hexagon with vertices at 0, 60, 120...
    # The supporting lines (normals) are at 30, 90, 150... degrees.
    # Let's just compute normals from vertices.
    all_normals = np.zeros((n, 6, 2))
    for i in range(n):
        verts = all_verts[i]
        for e in range(6):
            p1 = verts[e]
            p2 = verts[(e+1) % 6]
            edge = p2 - p1
            nx, ny = edge[1], -edge[0]
            norm = np.sqrt(nx*nx + ny*ny)
            if norm > 1e-12:
                all_normals[i, e] = [nx/norm, ny/norm]

    for i in range(n):
        for j in range(i+1, n):
            v1 = all_verts[i]
            v2 = all_verts[j]
            overlap = True
            # Check normals of hex i
            proj1 = v1 @ all_normals[i].T  # (6, 6)
            proj2 = v2 @ all_normals[i].T  # (6, 6)
            min1 = np.min(proj1, axis=0)
            max1 = np.max(proj1, axis=0)
            min2 = np.min(proj2, axis=0)
            max2 = np.max(proj2, axis=0)
            separated = (max1 + tol < min2) | (max2 + tol < min1)
            if np.any(separated):
                overlap = False
            else:
                # Check normals of hex j (same as i since angle is 0 for all)
                # proj1 = v1 @ all_normals[j].T
                # proj2 = v2 @ all_normals[j].T
                # min1 = np.min(proj1, axis=0)
                # max1 = np.max(proj1, axis=0)
                # min2 = np.min(proj2, axis=0)
                # max2 = np.max(proj2, axis=0)
                # separated = (max1 + tol < min2) | (max2 + tol < min1)
                # if np.any(separated):
                #     overlap = False
                pass # Already checked one set of normals, which is sufficient for identical orientations?
                # Wait, SAT requires checking normals of BOTH polygons.
                # Since they have the same orientation, the set of normals is the same.
                # So checking normals of i is sufficient.
            
            if overlap:
                return False
    return True

def _compute_all_vertices(centers, side=1.0):
    """Compute all 72 vertices of 12 hexagons (angle 0)."""
    verts = []
    for i in range(len(centers)):
        verts.append(_hex_vertices(centers[i,0], centers[i,1], side, 0.0))
    return np.vstack(verts)

def _initial_lattice():
    """
    Create initial lattice arrangement of 12 hexagons.
    Center + 6 neighbors + 5 from second ring.
    """
    d = np.sqrt(3.0)  # distance between adjacent hex centers
    positions = []
    
    # Center
    positions.append([0.0, 0.0])
    
    # First ring: 6 hexagons
    for k in range(6):
        angle = k * np.pi / 3
        x = d * np.cos(angle)
        y = d * np.sin(angle)
        positions.append([x, y])
    
    # Second ring: We need 5 more.
    # The second ring in a honeycomb has 12 positions.
    # They are at distance 2*d? No.
    # Distance from center to second ring center is d * sqrt(7) ?
    # Let's use axial coordinates.
    # Center (0,0)
    # Ring 1: (1,0), (1,-1), (0,-1), (-1,0), (-1,1), (0,1)
    # Ring 2 positions: (2,0), (2,-1), (2,-2), (1,-2), (0,-2), (-1,-1), (-2,0), (-2,1), (-2,2), (-1,2), (0,2), (1,1)
    # Convert axial (q,r) to cartesian (x,y) for flat-top hexagons?
    # Standard mapping: x = d * (q + r/2), y = d * sqrt(3)/2 * r ? 
    # Wait, the distance between centers is d.
    # If we use axial coordinates where neighbors are at distance 1 in axial space,
    # the cartesian distance is d.
    # Let's stick to the previous angular placement which was correct.
    # The 12 positions in the second ring are at angles 0, 30, 60... ? No.
    # The vertices of the hexagon formed by ring 1 centers are at d.
    # The ring 2 centers are further out.
    
    # Let's just use the known coordinates for the "flower" of 7, and add 5 more.
    # A compact arrangement for 12 is often 1 + 6 + 5.
    # The 5 should be placed to fill the "corners" of the hexagonal cluster.
    
    # Let's generate the full ring 2 and pick 5 that are furthest from each other or symmetric.
    # Actually, the most compact packing of 12 hexagons is not necessarily the full ring 2 subset.
    # But it's a good start.
    
    # Coordinates for Ring 2 (axial):
    # (2,0), (2,-1), (2,-2), (1,-2), (0,-2), (-1,-1), (-2,0), (-2,1), (-2,2), (-1,2), (0,2), (1,1)
    # Let's convert these to cartesian.
    # Using the standard hex grid mapping where distance between adjacent is d:
    # x = d * (q + r/2)
    # y = d * (sqrt(3)/2) * r
    
    ring2_axial = [
        (2, 0), (2, -1), (2, -2), (1, -2), (0, -2), (-1, -1),
        (-2, 0), (-2, 1), (-2, 2), (-1, 2), (0, 2), (1, 1)
    ]
    
    # We want to pick 5. To maintain symmetry, we might pick every other one?
    # 12 positions, pick 5. That breaks 6-fold symmetry.
    # Maybe we should shift the whole cluster?
    # For now, just pick the first 5? Or the ones at angles 0, 72, 144, 216, 288?
    # Let's calculate cartesian coords for all 12 and pick 5 that are well distributed.
    
    ring2_cart = []
    for q, r in ring2_axial:
        x = d * (q + r/2.0)
        y = d * (np.sqrt(3.0)/2.0) * r
        ring2_cart.append([x, y])
    
    # Sort by angle
    ring2_cart.sort(key=lambda p: np.arctan2(p[1], p[0]))
    
    # Pick 5 evenly spaced from the 12
    # Indices: 0, 2, 4, 7, 9 ?
    # Let's just pick 5 that are furthest from the center? They are all same distance.
    # Let's pick indices 0, 2, 5, 7, 10 to be roughly symmetric.
    indices = [0, 2, 5, 7, 10]
    for idx in indices:
        positions.append(ring2_cart[idx])
        
    return np.array(positions)

def _local_search(centers, deadline, outer_angle_deg=0.0):
    """
    Perform local search to minimize outer side length.
    """
    current_centers = centers.copy()
    all_verts = _compute_all_vertices(current_centers)
    current_s = _min_outer_side(all_verts, outer_angle_deg)
    
    best_centers = current_centers.copy()
    best_s = current_s
    
    step_size = 0.05
    T = 0.01
    cooling = 0.999
    
    np.random.seed(42)
    
    n_steps = 0
    while time.time() < deadline - 5:
        # Perturb one hexagon
        i = np.random.randint(0, 12)
        new_centers = current_centers.copy()
        new_centers[i] += np.random.randn(2) * step_size
        
        if not _check_nonoverlap(new_centers):
            continue
        
        all_verts_new = _compute_all_vertices(new_centers)
        s_new = _min_outer_side(all_verts_new, outer_angle_deg)
        
        delta = s_new - current_s
        if delta < 0 or np.random.random() < np.exp(-delta / T):
            current_centers = new_centers
            current_s = s_new
            if s_new < best_s:
                best_s = s_new
                best_centers = new_centers.copy()
        
        T *= cooling
        if n_steps % 1000 == 0:
            step_size *= 0.95
            step_size = max(step_size, 0.005)
        n_steps += 1
        
    return best_centers, best_s

def hexagon_packing_12():
    start_time = time.time()
    deadline = start_time + 620.0
    
    # 1. Get initial lattice positions
    centers = _initial_lattice()
    
    # 2. Local search with outer angle 0
    centers, s = _local_search(centers, deadline, 0.0)
    
    # 3. Try other outer angles
    best_overall_s = s
    best_overall_centers = centers
    best_overall_angle = 0.0
    
    # The local search was expensive. We can reuse the best centers found and just
    # check if a different outer angle yields a better s, or do a quick refinement.
    # However, the optimal inner arrangement depends on the outer angle.
    # Let's do a few more local searches with different outer angles if time permits.
    
    angles_to_try = [0.0, 15.0, 30.0, 45.0]
    
    for ang in angles_to_try:
        if time.time() > deadline - 100:
            break
        
        # Start from the best centers found so far
        c_start = best_overall_centers.copy()
        c_opt, s_opt = _local_search(c_start, deadline - 10, ang)
        
        if s_opt < best_overall_s:
            best_overall_s = s_opt
            best_overall_centers = c_opt
            best_overall_angle = ang
            
    # Final verification
    centers_final = best_overall_centers
    angles_final = np.zeros(12)
    
    # Center the configuration
    centroid = np.mean(centers_final, axis=0)
    centers_final = centers_final - centroid
    
    # Recalculate s with the best outer angle
    all_verts_final = _compute_all_vertices(centers_final)
    s_final = _min_outer_side(all_verts_final, best_overall_angle)
    
    # Ensure validity
    if not _check_nonoverlap(centers_final):
        # Fallback
        centers_final, angles_final = _initial_lattice(), np.zeros(12)
        centroid = np.mean(centers_final, axis=0)
        centers_final = centers_final - centroid
        all_verts_final = _compute_all_vertices(centers_final)
        s_final = _min_outer_side(all_verts_final, 0.0)
        best_overall_angle = 0.0

    inner_hex_data = np.column_stack([centers_final, angles_final])
    outer_hex_data = np.array([0.0, 0.0, best_overall_angle])
    
    return inner_hex_data, outer_hex_data, s_final

