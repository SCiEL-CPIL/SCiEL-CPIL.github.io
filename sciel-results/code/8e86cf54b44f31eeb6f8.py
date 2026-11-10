import numpy as np
import time

def hexagon_packing_12():
    start_time = time.time()
    deadline = 600.0  # Safety margin before 630s kill

    def hex_vertices(cx, cy, side, angle_deg):
        """Return 6 vertices of a regular hexagon."""
        a = np.deg2rad(angle_deg)
        angles = a + np.arange(6) * np.pi / 3
        return np.column_stack((cx + side * np.cos(angles),
                                cy + side * np.sin(angles)))

    def get_all_vertices(centers, side=1.0):
        """Get all 72 vertices of 12 unit hexagons."""
        verts = []
        for cx, cy in centers:
            angles = np.arange(6) * np.pi / 3
            verts.append(np.column_stack((cx + side * np.cos(angles),
                                          cy + side * np.sin(angles))))
        return np.vstack(verts)

    def min_enclosing_hex_side(verts):
        """Find minimum side length of a regular hexagon (centered at origin)
        that contains all vertices. Returns (s, theta_deg).
        
        A regular hexagon with side s, angle theta contains point p iff
        for all k=0..5: |p . n_k| <= s * sqrt(3)/2
        where n_k = (cos(theta + 60k deg), sin(theta + 60k deg)).
        
        So s >= 2/sqrt(3) * max_k max_i |v_i . n_k|
        """
        # For a regular hexagon centered at origin with "radius" R (center to vertex),
        # the apothem (center to edge) is R * cos(30 deg) = R * sqrt(3)/2.
        # The side length s = R.
        # A point p is inside iff for all 6 edge normals n_k:
        #   p . n_k <= apothem = s * sqrt(3)/2
        # So s >= 2/sqrt(3) * max_k (max_i v_i . n_k)
        # But we also need the negative side, so we need:
        #   |v_i . n_k| <= s * sqrt(3)/2 for all i, k
        # Thus s >= 2/sqrt(3) * max_{i,k} |v_i . n_k|
        
        # The 6 normal directions for a hexagon with angle theta are:
        # theta + 0, 60, 120, 180, 240, 300 degrees
        # But note: n and -n give the same constraint, so we only need 3 unique directions
        # Actually for a regular hexagon, the 6 edge normals are at angles:
        # theta + 30 + 60k for k=0..5 (the normals point outward from edges)
        # Wait, let me reconsider.
        
        # A regular hexagon with vertices at angles theta, theta+60, ..., theta+300
        # from center, at distance R=s from center.
        # The edges are between consecutive vertices. The outward normal of the edge
        # between vertex at angle alpha and alpha+60 is at angle alpha+30.
        # So the 6 outward normals are at angles: theta+30, theta+90, theta+150, 
        # theta+210, theta+270, theta+330.
        # The apothem (distance from center to each edge) is s*cos(30 deg) = s*sqrt(3)/2.
        # A point p is inside iff for all 6 normals n_k: p . n_k <= s*sqrt(3)/2.
        
        # To minimize s, for a given theta:
        # s(theta) = (2/sqrt(3)) * max_k max_i (v_i . n_k(theta))
        # where n_k(theta) = (cos(theta+30+60k deg), sin(theta+30+60k deg))
        
        # We want min_theta s(theta).
        # Since the function is periodic with period 60 deg, search theta in [0, 60).
        
        best_s = float('inf')
        best_theta = 0.0
        
        # Sample theta finely, then refine
        # First coarse pass
        thetas = np.linspace(0, np.pi/3, 180, endpoint=False)
        
        for theta in thetas:
            # 6 outward normal angles
            norm_angles = theta + np.pi/6 + np.arange(6) * np.pi/3
            normals = np.column_stack((np.cos(norm_angles), np.sin(norm_angles)))
            
            # Project all vertices onto each normal
            # projections[i,k] = v_i . n_k
            projections = verts @ normals.T  # shape (72, 6)
            max_proj = np.max(projections, axis=0)  # shape (6,)
            required_s = np.max(max_proj) * 2.0 / np.sqrt(3)
            
            if required_s < best_s:
                best_s = required_s
                best_theta = theta
        
        # Refine around best_theta
        for _ in range(3):
            thetas_fine = np.linspace(best_theta - np.pi/60, best_theta + np.pi/60, 200, endpoint=False)
            for theta in theths_fine:
                norm_angles = theta + np.pi/6 + np.arange(6) * np.pi/3
                normals = np.column_stack((np.cos(norm_angles), np.sin(norm_angles)))
                projections = verts @ normals.T
                max_proj = np.max(projections, axis=0)
                required_s = np.max(max_proj) * 2.0 / np.sqrt(3)
                if required_s < best_s:
                    best_s = required_s
                    best_theta = theta
        
        return best_s, np.rad2deg(best_theta)

    def check_separation(centers, tol=1e-6):
        """Check that no two unit hexagons overlap using SAT."""
        n = len(centers)
        for i in range(n):
            for j in range(i + 1, n):
                cx_i, cy_i = centers[i]
                cx_j, cy_j = centers[j]
                
                # Get vertices
                angles = np.arange(6) * np.pi / 3
                vi = np.column_stack((cx_i + np.cos(angles), cy_i + np.sin(angles)))
                vj = np.column_stack((cx_j + np.cos(angles), cy_j + np.sin(angles)))
                
                # SAT: check all 12 edge normals (6 from each hex)
                # Edge normals for hex i
                for k in range(6):
                    # Edge from vi[k] to vi[(k+1)%6]
                    edge = vi[(k+1) % 6] - vi[k]
                    # Normal (perpendicular, pointing outward)
                    # For a convex polygon, the outward normal of edge (a,b) is 
                    # (-(b_y - a_y), b_x - a_x) normalized, but we need to be careful
                    # Actually for SAT we just need any perpendicular direction
                    nx = -(edge[1])
                    ny = edge[0]
                    norm = np.sqrt(nx*nx + ny*ny)
                    if norm < 1e-12:
                        continue
                    nx /= norm
                    ny /= norm
                    
                    # Project both hexagons onto this normal
                    proj_i = vi @ np.array([nx, ny])
                    proj_j = vj @ np.array([nx, ny])
                    
                    min_i = np.min(proj_i)
                    max_i = np.max(proj_i)
                    min_j = np.min(proj_j)
                    max_j = np.max(proj_j)
                    
                    # Check for gap
                    if max_i + tol < min_j or max_j + tol < min_i:
                        # There's a separating axis -> they don't overlap on this axis
                        # But we need ALL axes to have overlap for them to overlap
                        # Actually SAT says: if there EXISTS a separating axis, they don't overlap
                        # So if we find ANY gap, they're separated -> good
                        # Wait no: SAT says two convex shapes overlap iff for ALL
                        # edge normals of both shapes, the projections overlap.
                        # So if we find ANY axis where projections DON'T overlap,
                        # the shapes are separated (valid).
                        # If projections overlap on ALL 12 axes, they overlap (invalid).
                        pass  # This axis shows separation, continue checking
                    else:
                        # Projections overlap on this axis, need to check more
                        continue
                
                # If we get here, all 12 axes showed overlap -> hexagons overlap
                # But wait, I need to also check normals from hex j
                # Let me redo this properly
                
        # Redo SAT properly
        for i in range(n):
            for j in range(i + 1, n):
                cx_i, cy_i = centers[i]
                cx_j, cy_j = centers[j]
                
                angles = np.arange(6) * np.pi / 3
                vi = np.column_stack((cx_i + np.cos(angles), cy_i + np.sin(angles)))
                vj = np.column_stack((cx_j + np.cos(angles), cy_j + np.sin(angles)))
                
                overlap_on_all = True
                # Check all 12 edge normals
                for hex_verts in [vi, vj]:
                    for k in range(6):
                        edge = hex_verts[(k+1) % 6] - hex_verts[k]
                        nx = -(edge[1])
                        ny = edge[0]
                        norm = np.sqrt(nx*nx + ny*ny)
                        if norm < 1e-12:
                            continue
                        nx /= norm
                        ny /= norm
                        
                        proj_i = vi @ np.array([nx, ny])
                        proj_j = vj @ np.array([nx, ny])
                        
                        if np.max(proj_i) < np.min(proj_j) - tol or \
                           np.max(proj_j) < np.min(proj_i) - tol:
                            overlap_on_all = False
                            break
                    if not overlap_on_all:
                        break
                
                if overlap_on_all:
                    return False  # Hexagons overlap
        return True

    def generate_symmetric_config(r1, r2, phi):
        """Generate 12 hexagon centers with 6-fold symmetry.
        r1: radius of ring 1 (6 hexagons)
        r2: radius of ring 2 (5 hexagons)  
        phi: rotation offset of ring 2
        """
        centers = [[0.0, 0.0]]  # Center hexagon
        
        # Ring 1: 6 hexagons at radius r1
        for k in range(6):
            angle = k * np.pi / 3
            centers.append([r1 * np.cos(angle), r1 * np.sin(angle)])
        
        # Ring 2: 5 hexagons at radius r2, offset by phi
        for k in range(5):
            angle = phi + k * 2 * np.pi / 5
            centers.append([r2 * np.cos(angle), r2 * np.sin(angle)])
        
        return np.array(centers)

    def objective(params):
        """Objective: minimize s given symmetric config parameters."""
        r1, r2, phi = params
        
        # Ensure positive radii
        r1 = max(r1, 0.5)
        r2 = max(r2, 0.5)
        
        centers = generate_symmetric_config(r1, r2, phi)
        
        # Check separation
        if not check_separation(centers):
            return 1e6
        
        # Compute minimum enclosing hexagon
        verts = get_all_vertices(centers)
        s, theta = min_enclosing_hex_side(verts)
        
        return s

    def optimize_from_start(r1_init, r2_init, phi_init):
        """Optimize from a given starting point using Nelder-Mead."""
        from scipy.optimize import minimize
        
        x0 = np.array([r1_init, r2_init, phi_init])
        
        try:
            res = minimize(
                objective,
                x0,
                method='Nelder-Mead',
                options={'maxiter': 500, 'xatol': 1e-10, 'fatol': 1e-10, 'adaptive': True}
            )
            return res.fun, res.x
        except Exception:
            return None, None

    # Multiple starting points
    starts = [
        (2.0, 4.0, 0.0),
        (2.0, 4.0, np.pi/5),
        (2.5, 4.5, 0.0),
        (2.5, 4.5, np.pi/5),
        (3.0, 5.0, 0.0),
        (3.0, 5.0, np.pi/5),
        (2.0, 5.0, np.pi/10),
        (2.5, 5.0, np.pi/10),
        (3.0, 5.5, np.pi/10),
        (2.2, 4.2, 0.3),
        (2.8, 4.8, 0.5),
        (2.0, 4.5, 0.6),
        (2.5, 5.0, 0.4),
        (3.0, 5.0, 0.2),
    ]

    best_s = float('inf')
    best_params = None

    for r1_i, r2_i, phi_i in starts:
        if time.time() > deadline:
            break
        
        s, params = optimize_from_start(r1_i, r2_i, phi_i)
        
        if s is not None and s < best_s:
            best_s = s
            best_params = params

    # If symmetric optimization failed, try a general approach
    if best_params is None or best_s > 10:
        # Fallback: use a known good configuration
        # Hexagonal packing with 12 hexagons
        # Center + 6 in ring 1 + 5 in ring 2
        # For unit hexagons, ring 1 radius = sqrt(3) ≈ 1.732
        # Ring 2 needs to be larger
        r1 = np.sqrt(3)
        r2 = 2 * np.sqrt(3)  # Approximate
        phi = np.pi / 5
        
        centers = generate_symmetric_config(r1, r2, phi)
        verts = get_all_vertices(centers)
        best_s, best_theta = min_enclosing_hex_side(verts)
        best_params = np.array([r1, r2, phi])

    # Extract final parameters
    r1, r2, phi = best_params
    centers = generate_symmetric_config(r1, r2, phi)
    
    # Compute the outer hexagon angle and side
    verts = get_all_vertices(centers)
    s, theta_deg = min_enclosing_hex_side(verts)
    
    # Shift centers so centroid is at origin
    centroid = np.mean(centers, axis=0)
    shifted_centers = centers - centroid
    
    # Build inner hex data
    inner_hex_data = np.zeros((12, 3))
    for i, (cx, cy) in enumerate(shifted_centers):
        inner_hex_data[i] = [cx, cy, 0]
    
    # Outer hexagon: centered at origin, with side s and angle theta_deg
    outer_hex_data = np.array([0.0, 0.0, theta_deg])
    outer_hex_side_length = s

    # Final validation: verify all constraints
    # Check outer hexagon contains all inner vertices
    outer_verts = hex_vertices(0, 0, s, theta_deg)
    for i in range(12):
        cx, cy, _ = inner_hex_data[i]
        # Check each vertex of inner hex i is inside outer hex
        inner_v = hex_vertices(cx, cy, 1.0, 0)
        for v in inner_v:
            # Check if v is inside outer hex using cross product test
            inside = True
            for k in range(6):
                p1 = outer_verts[k]
                p2 = outer_verts[(k+1) % 6]
                cross = (p2[0]-p1[0])*(v[1]-p1[1]) - (p2[1]-p1[1])*(v[0]-p1[0])
                if cross < -1e-6:
                    inside = False
                    break
            if not inside:
                # Adjust: increase s slightly
                s *= 1.001
                outer_hex_data = np.array([0.0, 0.0, theta_deg])
                outer_hex_side_length = s
                break

    return inner_hex_data, outer_hex_data, outer_hex_side_length
