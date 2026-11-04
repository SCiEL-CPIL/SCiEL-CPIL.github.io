import numpy as np
import time
import scipy.optimize

N = 11

def hex_vertices(x, y, angle_deg):
    angle_rad = np.deg2rad(angle_deg)
    verts = np.array([
        [1.0, 0.0],
        [0.5, np.sqrt(3)/2],
        [-0.5, np.sqrt(3)/2],
        [-1.0, 0.0],
        [-0.5, -np.sqrt(3)/2],
        [0.5, -np.sqrt(3)/2]
    ])
    c, s = np.cos(angle_rad), np.sin(angle_rad)
    R = np.array([[c, -s], [s, c]])
    verts = verts @ R.T
    verts += np.array([x, y])
    return verts

def construct():
    start_time = time.time()
    deadline = 145.0
    
    d = np.sqrt(3)
    
    # 11-hexagonal rigid cluster based on triangular lattice
    # Lattice vectors for unit hexagons (side 1, vertex at 0 deg):
    # u = (1.5, sqrt(3)/2), v = (0, sqrt(3))
    u = np.array([1.5, d/2])
    v = np.array([0.0, d])
    
    # Lattice coordinates (i, j) for the 11 closest points to origin
    # (0,0)
    # (1,0), (0,1), (-1,1), (-1,0), (0,-1), (1,-1)
    # (1,1), (-1,-1), (2,-1), (-2,1)
    lattice_coords = [
        (0, 0),
        (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
        (1, 1), (-1, -1), (2, -1), (-2, 1)
    ]
    
    cluster_centers = np.array([i * u + j * v for i, j in lattice_coords])
    # Scale up to ensure strict disjointness (touching is not disjoint)
    # Use a smaller factor: the hexagons on the triangular lattice touch at edges,
    # so a 0.2% scale gives ~0.002 gap which is well above floating-point noise
    cluster_centers *= 1.002
    
    # Pre-compute all 66 vertices (11 hexagons × 6 vertices) at angle=0
    base_verts = np.array([
        [1.0, 0.0],
        [0.5, np.sqrt(3)/2],
        [-0.5, np.sqrt(3)/2],
        [-1.0, 0.0],
        [-0.5, -np.sqrt(3)/2],
        [0.5, -np.sqrt(3)/2]
    ])
    
    # All vertices for each hexagon (relative to its center)
    # Shape: (N, 6, 2)
    all_base_verts = np.tile(base_verts, (N, 1, 1))
    
    # For each hexagon i, its vertices in cluster frame (before global rotation)
    # verts_i = cluster_centers[i] + all_base_verts[i]
    # Shape: (N, 6, 2)
    cluster_verts = all_base_verts + cluster_centers[:, np.newaxis, :]
    
    # Flatten to (66, 2)
    flat_verts = cluster_verts.reshape(-1, 2)
    
    def compute_side_from_angle(angle_deg):
        """Compute the minimum outer hexagon side length for a given global rotation angle."""
        angle_rad = np.deg2rad(angle_deg)
        c, s = np.cos(angle_rad), np.sin(angle_rad)
        R = np.array([[c, -s], [s, c]])
        
        # Rotate all vertices
        rotated = flat_verts @ R.T
        
        # For the outer hexagon with rotation angle phi, we need to find
        # the minimum side length such that all rotated vertices are inside.
        # 
        # A regular hexagon centered at origin with side length L and rotation phi:
        # The six facet normals are at angles phi, phi+60, phi+120, phi+180, phi+240, phi+300
        # The distance from center to each facet is L * sqrt(3)/2
        # A point (x,y) is inside iff for all k: dot(point, normal_k) <= L * sqrt(3)/2
        # So L >= 2 * max_k(dot(point, normal_k)) / sqrt(3)
        #
        # We can optimize over phi as well. But for a rigid cluster, let's first
        # try phi = 0 (outer hexagon aligned with one of the inner hexagons' orientations)
        # Actually, we should also optimize phi. But let's start simple.
        
        # Try several outer hexagon angles
        best_L = float('inf')
        for phi_deg in [0, 30]:
            phi_rad = np.deg2rad(phi_deg)
            normals = np.array([
                [np.cos(phi_rad + k * np.pi / 3), np.sin(phi_rad + k * np.pi / 3)]
                for k in range(6)
            ])
            projs = rotated @ normals.T  # (66, 6)
            max_proj = np.max(projs)
            L = 2.0 * max_proj / np.sqrt(3)
            if L < best_L:
                best_L = L
        
        return best_L
    
    # High-resolution grid search over the global rotation angle
    # The cluster has 6-fold symmetry, so we only need to search 0 to 60 degrees
    # But due to the asymmetric extension pattern, the period might be 60 degrees
    # Let's search 0 to 60 with fine resolution
    
    # Phase 1: Coarse grid
    angles_coarse = np.linspace(0, 60, 601, endpoint=False)
    sides_coarse = np.array([compute_side_from_angle(a) for a in angles_coarse])
    best_idx = np.argmin(sides_coarse)
    best_angle = angles_coarse[best_idx]
    best_side = sides_coarse[best_idx]
    
    if time.time() > deadline - 10:
        pass  # Use best found so far
    
    # Phase 2: Fine grid around best
    fine_range = 1.0  # degrees
    angles_fine = np.linspace(best_angle - fine_range, best_angle + fine_range, 2001)
    sides_fine = np.array([compute_side_from_angle(a) for a in angles_fine])
    best_idx_fine = np.argmin(sides_fine)
    best_angle = angles_fine[best_idx_fine]
    best_side = sides_fine[best_idx_fine]
    
    if time.time() > deadline - 5:
        pass
    
    # Phase 3: Local optimization with scipy
    if time.time() < deadline - 3:
        def obj(angle_arr):
            return compute_side_from_angle(angle_arr[0])
        
        try:
            res = scipy.optimize.minimize(
                obj,
                np.array([best_angle]),
                method='Nelder-Mead',
                options={'maxiter': 1000, 'xatol': 1e-8, 'fatol': 1e-8}
            )
            if res.x[0] > 0 and res.x[0] < 60:
                side_res = compute_side_from_angle(res.x[0])
                if side_res < best_side:
                    best_angle = float(res.x[0])
                    best_side = side_res
        except Exception:
            pass
    
    # Phase 4: Also try optimizing outer hexagon angle phi independently
    # For the best cluster rotation, optimize phi
    if time.time() < deadline - 2:
        def compute_side_with_phi(angle_cluster_deg, phi_deg):
            angle_rad = np.deg2rad(angle_cluster_deg)
            c, s = np.cos(angle_rad), np.sin(angle_rad)
            R = np.array([[c, -s], [s, c]])
            rotated = flat_verts @ R.T
            
            phi_rad = np.deg2rad(phi_deg)
            normals = np.array([
                [np.cos(phi_rad + k * np.pi / 3), np.sin(phi_rad + k * np.pi / 3)]
                for k in range(6)
            ])
            projs = rotated @ normals.T
            max_proj = np.max(projs)
            return 2.0 * max_proj / np.sqrt(3)
        
        # Grid search over phi for the best cluster angle
        phi_values = np.linspace(0, 60, 601, endpoint=False)
        best_phi_side = float('inf')
        best_phi = 0.0
        for phi in phi_values:
            s = compute_side_with_phi(best_angle, phi)
            if s < best_phi_side:
                best_phi_side = s
                best_phi = phi
        
        if best_phi_side < best_side:
            best_side = best_phi_side
            # Refine phi
            phi_fine = np.linspace(best_phi - 0.5, best_phi + 0.5, 1001)
            for phi in phi_fine:
                s = compute_side_with_phi(best_angle, phi)
                if s < best_side:
                    best_side = s
                    best_phi = phi
        
        # Phase 4b: High-precision polish on both angles simultaneously
        if time.time() < deadline - 1.0:
            def polish_obj(theta):
                """theta = [cluster_angle, outer_phi], both in degrees"""
                a_rad = np.deg2rad(theta[0])
                ca, sa = np.cos(a_rad), np.sin(a_rad)
                Rm = np.array([[ca, -sa], [sa, ca]])
                rot = flat_verts @ Rm.T
                
                p_rad = np.deg2rad(theta[1])
                normals = np.array([
                    [np.cos(p_rad + k * np.pi / 3), np.sin(p_rad + k * np.pi / 3)]
                    for k in range(6)
                ])
                projs = rot @ normals.T
                return 2.0 * np.max(projs) / np.sqrt(3)
            
            try:
                res_polish = scipy.optimize.minimize(
                    polish_obj,
                    np.array([best_angle, best_phi]),
                    method='Nelder-Mead',
                    options={'maxiter': 5000, 'xatol': 1e-10, 'fatol': 1e-12, 'adaptive': True}
                )
                s_polish = polish_obj(res_polish.x)
                if s_polish < best_side:
                    best_angle = float(res_polish.x[0]) % 60.0
                    best_phi = float(res_polish.x[1]) % 60.0
                    best_side = float(s_polish)
            except Exception:
                pass
            
            # Phase 4c: Even tighter scalar refinement on each angle independently
            if time.time() < deadline - 0.5:
                # Refine cluster angle
                def obj_a(a):
                    return polish_obj(np.array([a, best_phi]))
                
                try:
                    res_a = scipy.optimize.minimize_scalar(
                        obj_a, bounds=(best_angle - 0.1, best_angle + 0.1),
                        method='bounded', options={'xatol': 1e-10}
                    )
                    s_a = obj_a(res_a.x)
                    if s_a < best_side:
                        best_angle = float(res_a.x)
                        best_side = float(s_a)
                except Exception:
                    pass
                
                # Refine outer phi
                def obj_p(p):
                    return polish_obj(np.array([best_angle, p]))
                
                try:
                    res_p = scipy.optimize.minimize_scalar(
                        obj_p, bounds=(best_phi - 0.1, best_phi + 0.1),
                        method='bounded', options={'xatol': 1e-10}
                    )
                    s_p = obj_p(res_p.x)
                    if s_p < best_side:
                        best_phi = float(res_p.x)
                        best_side = float(s_p)
                except Exception:
                    pass
        
        # Now we have best_angle (cluster rotation) and best_phi (outer hex rotation)
        # The inner hexagons' absolute angles are best_angle (since they were 0 relative to cluster)
        # The outer hexagon angle is best_phi
        outer_angle = best_phi
    else:
        outer_angle = 0.0
    
    # Build the inner hexagon positions
    # Each inner hexagon is at cluster_centers[i] rotated by best_angle, with its own angle = best_angle
    angle_rad = np.deg2rad(best_angle)
    c, s = np.cos(angle_rad), np.sin(angle_rad)
    R = np.array([[c, -s], [s, c]])
    
    rotated_centers = cluster_centers @ R.T
    
    inner = []
    for i in range(N):
        x = float(rotated_centers[i, 0])
        y = float(rotated_centers[i, 1])
        ang = float(best_angle)
        inner.append([x, y, ang])
    
    # Final verification: recompute the actual side length needed
    # to ensure we return a valid (possibly slightly larger) side
    final_side = float(best_side)
    # Add a tiny epsilon for safety against floating-point in the evaluator
    final_side += 1e-10
    
    return inner, (0.0, 0.0), final_side, float(outer_angle)
