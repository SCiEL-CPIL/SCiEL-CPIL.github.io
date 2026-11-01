import numpy as np
import time
from scipy.optimize import differential_evolution

N = 12

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s limit

    # Hexagonal lattice positions for 12 unit hexagons (centered hex ring radius 2)
    # Using axial coordinates (q, r) -> Cartesian for flat-top hexagons
    # The 12 positions: center + ring of 6 + outer 5 (to make 12)
    # Actually for a compact arrangement of 12:
    #   Center (1) + Ring 1 (6) + Ring 2 partial (5) = 12
    # Let's use a standard hex grid with spacing = sqrt(3) (flat-to-flat distance)
    
    sqrt3 = np.sqrt(3.0)
    
    # Axial to Cartesian (flat-top): x = sqrt3*(q + r/2), y = 1.5*r
    def axial_to_cartesian(q, r):
        return np.array([sqrt3 * (q + r / 2.0), 1.5 * r])
    
    # 12 positions: center + 6 in ring 1 + 5 in ring 2
    axial_coords = [
        (0, 0),   # center
        (1, 0), (1, -1), (0, -1), (-1, 0), (-1, 1), (0, 1),  # ring 1
        (2, 0), (2, -1), (1, -2), (-1, 2), (-2, 1)  # 5 from ring 2
    ]
    
    # Wait, that's 12 but let me verify they're all distinct and form a nice cluster
    # Let me use a more standard compact arrangement:
    # Actually, the classic "hex number" 12 = 1 + 6 + 5 (partial ring 2)
    # But a better compact shape might be a 3x4 rhombus or similar.
    # Let's use the standard hexagonal number layout:
    # Row -2: q=-1,0,1 (3 hexes)
    # Row -1: q=-1,0,1,2 (4 hexes) 
    # Row 0: q=-1,0,1,2 (4 hexes)
    # Row 1: q=0,1 (2 hexes) -- not symmetric
    # 
    # Better: use the actual centered hexagonal number H(2) = 19, take 12 of them
    # Or use a 2x3x2 block in hex grid
    #
    # Simplest compact: hexagonal lattice with the 12 closest points to origin
    # In hex grid, the 12 nearest neighbors (excluding center) + center = 13, so take 12
    # Actually let's just use a well-known compact 12-hex arrangement:
    
    # Use axial coordinates for a compact 12-hex shape:
    # A 3x4 parallelogram in hex grid:
    # (0,0), (1,0), (2,0), (3,0)
    # (0,1), (1,1), (2,1), (3,1)
    # (0,2), (1,2), (2,2), (3,2)
    # That's 12 hexes in a 4x3 block
    
    axial_coords = []
    for r in range(3):
        for q in range(4):
            axial_coords.append((q, r))
    
    # Convert to Cartesian
    base_positions = np.array([axial_to_cartesian(q, r) for q, r in axial_coords])
    
    # Center the configuration
    centroid = base_positions.mean(axis=0)
    base_positions = base_positions - centroid
    
    def hexagon_vertices(cx, cy, side, angle_deg):
        angle_rad = np.radians(angle_deg)
        angles = angle_rad + np.radians(np.arange(0, 360, 60))
        return np.column_stack([
            cx + side * np.cos(angles),
            cy + side * np.sin(angles)
        ])

    def compute_outer_side(params):
        """
        params: [cx, cy, rotation_deg, distortion]
        cx, cy: outer hexagon center
        rotation_deg: outer hexagon rotation
        distortion: small perturbation factor applied to lattice (0 = perfect lattice)
        
        Returns the side length of the outer hexagon needed to contain all inner hexagons.
        """
        cx, cy, rot_deg, distortion = params
        
        # Apply distortion to lattice positions (slight scaling of radial components)
        positions = base_positions.copy()
        if distortion != 0.0:
            # Radial distortion: scale each position by (1 + distortion * r_normalized)
            dists = np.linalg.norm(positions, axis=1)
            max_dist = dists.max()
            if max_dist > 1e-10:
                scale_factors = 1.0 + distortion * (dists / max_dist)
                positions = positions * scale_factors[:, np.newaxis]
        
        # Compute all vertices of all 12 inner hexagons
        all_verts = np.zeros((N, 6, 2))
        for i in range(N):
            all_verts[i] = hexagon_vertices(positions[i, 0], positions[i, 1], 1.0, 0.0)
        
        verts_flat = all_verts.reshape(-1, 2)
        
        # Rotate outer hexagon: we need to find min side of rotated hexagon containing all points
        # For a regular hexagon with center (cx, cy), rotation theta, side S:
        # A point p is inside if for all 6 face normals n_k(theta):
        #   (p - center) . n_k <= S * sqrt(3) / 2
        # So S >= max_k [(p - center) . n_k] / (sqrt(3)/2)
        # We want min over theta of max_k max_p [(p - center) . n_k(theta)] / (sqrt(3)/2)
        
        # Relative positions
        rel = verts_flat - np.array([cx, cy])
        
        # 6 face normals for rotation rot_deg
        theta = np.radians(rot_deg)
        normals = np.zeros((6, 2))
        for k in range(6):
            angle = theta + np.radians(k * 60)
            normals[k] = [np.cos(angle), np.sin(angle)]
        
        # For each normal, compute max dot product
        max_dots = np.zeros(6)
        for k in range(6):
            dots = rel.dot(normals[k])
            max_dots[k] = np.max(dots)
        
        # Side length = max(max_dots) / (sqrt(3)/2)
        side = np.max(max_dots) / (sqrt3 / 2.0)
        
        return side

    def objective(params):
        """Objective for differential evolution: minimize outer side length."""
        return compute_outer_side(params)

    # Define bounds for differential evolution
    # cx, cy: outer center, search in [-5, 5]
    # rot_deg: 0 to 60 (hexagon has 6-fold symmetry)
    # distortion: -0.3 to 0.3
    bounds = [
        (-5.0, 5.0),   # cx
        (-5.0, 5.0),   # cy
        (0.0, 60.0),   # rotation
        (-0.3, 0.3)    # distortion
    ]

    # Run differential evolution
    best_params = None
    best_side = float('inf')
    
    try:
        result = differential_evolution(
            objective,
            bounds,
            seed=42,
            maxiter=200,
            tol=1e-8,
            popsize=15,
            mutation=(0.5, 1.0),
            recombination=0.8,
            polish=True,
            updating='immediate',
            init='latinhypercube',
            seed=42,
            callback=None,
            workers=1
        )
        
        if result.success or result.fun < best_side:
            best_params = result.x
            best_side = result.fun
            
    except Exception as e:
        pass
    
    # If DE didn't find a good solution, try a few manual rotations
    if best_params is None or best_side > 10.0:
        best_params = [0.0, 0.0, 0.0, 0.0]
        best_side = compute_outer_side(best_params)
    
    # Fine-tune with local optimization around best solution
    from scipy.optimize import minimize
    
    try:
        res = minimize(
            compute_outer_side,
            best_params,
            method='Nelder-Mead',
            options={'maxiter': 500, 'xatol': 1e-8, 'fatol': 1e-8}
        )
        if res.fun < best_side:
            best_side = res.fun
            best_params = res.x
    except Exception:
        pass
    
    # Try a few more rotations for fine-tuning
    for rot_try in np.linspace(0, 60, 13):
        if time.time() > deadline - 10:
            break
        test_params = [best_params[0], best_params[1], rot_try, best_params[3]]
        test_side = compute_outer_side(test_params)
        if test_side < best_side:
            best_side = test_side
            best_params = test_params
    
    # Final refinement
    try:
        res = minimize(
            compute_outer_side,
            best_params,
            method='Nelder-Mead',
            options={'maxiter': 1000, 'xatol': 1e-10, 'fatol': 1e-10}
        )
        if res.fun < best_side:
            best_side = res.fun
            best_params = res.x
    except Exception:
        pass

    # Extract final parameters
    cx, cy, rot_deg, distortion = best_params
    
    # Compute final inner positions
    positions = base_positions.copy()
    if distortion != 0.0:
        dists = np.linalg.norm(positions, axis=1)
        max_dist = dists.max()
        if max_dist > 1e-10:
            scale_factors = 1.0 + distortion * (dists / max_dist)
            positions = positions * scale_factors[:, np.newaxis]
    
    # Verify validity (should be valid by construction due to lattice spacing)
    # But let's double-check no overlaps
    def check_disjoint(positions):
        for i in range(N):
            for j in range(i + 1, N):
                p1 = hexagon_vertices(positions[i, 0], positions[i, 1], 1.0, 0.0)
                p2 = hexagon_vertices(positions[j, 0], positions[j, 1], 1.0, 0.0)
                
                # SAT check
                disjoint = False
                for i_sat in range(6):
                    j_sat = (i_sat + 1) % 6
                    edge = p1[j_sat] - p1[i_sat]
                    normal = np.array([-edge[1], edge[0]])
                    norm = np.linalg.norm(normal)
                    if norm < 1e-10:
                        continue
                    normal /= norm
                    proj1 = p1.dot(normal)
                    proj2 = p2.dot(normal)
                    if np.max(proj1) < np.min(proj2) - 1e-6 or np.max(proj2) < np.min(proj1) - 1e-6:
                        disjoint = True
                        break
                
                if not disjoint:
                    for i_sat in range(6):
                        j_sat = (i_sat + 1) % 6
                        edge = p2[j_sat] - p2[i_sat]
                        normal = np.array([-edge[1], edge[0]])
                        norm = np.linalg.norm(normal)
                        if norm < 1e-10:
                            continue
                        normal /= norm
                        proj1 = p1.dot(normal)
                        proj2 = p2.dot(normal)
                        if np.max(proj1) < np.min(proj2) - 1e-6 or np.max(proj2) < np.min(proj1) - 1e-6:
                            disjoint = True
                            break
                    if not disjoint:
                        return False
        return True
    
    if not check_disjoint(positions):
        # Fallback: use a more spread out configuration
        # Scale up the lattice
        for scale in [1.1, 1.2, 1.3, 1.5]:
            test_pos = base_positions * scale
            if check_disjoint(test_pos):
                positions = test_pos
                break
    
    # Recompute final side length
    final_side = compute_outer_side([cx, cy, rot_deg, distortion])
    
    # If the computed side is too large, try to optimize center and rotation more carefully
    if final_side > 8.0:
        # Brute force over rotation and center
        best_final_side = final_side
        best_final_params = [cx, cy, rot_deg, distortion]
        
        for rot in np.linspace(0, 60, 37):
            for cx_try in np.linspace(-3, 3, 13):
                for cy_try in np.linspace(-3, 3, 13):
                    if time.time() > deadline - 5:
                        break
                    test_side = compute_outer_side([cx_try, cy_try, rot, distortion])
                    if test_side < best_final_side:
                        best_final_side = test_side
                        best_final_params = [cx_try, cy_try, rot, distortion]
        
        final_side = best_final_side
        cx, cy, rot_deg, distortion = best_final_params
    
    # Build output
    inner = []
    for i in range(N):
        inner.append([float(positions[i, 0]), float(positions[i, 1]), 0.0])
    
    return inner, (float(cx), float(cy)), float(final_side), float(rot_deg)

# EVOLVE-BLOCK-END
