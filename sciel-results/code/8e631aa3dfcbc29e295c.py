import numpy as np
import time
import scipy.optimize

N = 11

def hex_vertices_relative(angle_rad):
    """
    Returns the 6 vertices of a unit regular hexagon centered at origin,
    rotated by angle_rad.
    """
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
    return verts @ R.T

def construct():
    start_time = time.time()
    deadline = 145.0  # Safety margin before 150s limit
    
    # --- 1. Define Initial Cluster Topology (Flower Pattern) ---
    # This topology is known to be efficient for packing hexagons.
    # 1 center, 6 in first ring, 4 in second ring (gaps filled).
    # Note: We start with a valid, disjoint configuration.
    # Disjointness is ensured by a small scaling factor.
    
    # Base centers in a "flat-top" orientation relative to the cluster frame
    # The distance between centers of adjacent touching hexagons is sqrt(3).
    # We scale by 1.001 to ensure strict disjointness (gap ~0.001 * sqrt(3) ~ 0.0017).
    scale = 1.001
    
    # Center 0
    c0 = np.array([0.0, 0.0])
    
    # Ring 1: 6 hexagons at distance sqrt(3)
    r1 = np.sqrt(3) * scale
    ring1_angles = np.deg2rad([0, 60, 120, 180, 240, 300])
    ring1_centers = []
    for ang in ring1_angles:
        ring1_centers.append([r1 * np.cos(ang), r1 * np.sin(ang)])
    
    # Ring 2: 4 hexagons at distance 3*sqrt(3) is too far.
    # The "flower" of 7 has radius sqrt(3)+1.
    # To add 4 more, we place them in the "bays" between ring 1 hexagons.
    # The distance from center to the "bay" center is roughly 2 * sqrt(3) * cos(30) = 3?
    # Let's use the standard triangular lattice coordinates scaled.
    # Lattice vectors for touching hexagons:
    # u = (1.5, sqrt(3)/2), v = (0, sqrt(3))
    # Points:
    # (0,0)
    # (1,0), (0,1), (-1,1), (-1,0), (0,-1), (1,-1) -> These are the 6 neighbors.
    # Next layer neighbors of the neighbors that are not already present:
    # (2,0), (1,1), (-1,2), (-2,1), (-2,0), (-1,-1), (0,-2), (1,-2)
    # Wait, the "Flower" of 11 usually refers to the 1 center + 6 + 4 specific ones.
    # Let's stick to the triangular lattice subset that forms a compact cluster.
    # The set of 11 points:
    # (0,0)
    # (1,0), (0,1), (-1,1), (-1,0), (0,-1), (1,-1)
    # (2,-1), (-2,1), (1,1), (-1,-1) -- wait, (1,1) and (-1,-1) are already in ring 1?
    # Let's check lattice coords for Ring 1:
    # (1,0) -> dist sqrt(3)
    # (0,1) -> dist sqrt(3)
    # (-1,1) -> dist sqrt(3)
    # (-1,0) -> dist sqrt(3)
    # (0,-1) -> dist sqrt(3)
    # (1,-1) -> dist sqrt(3)
    # These are the 6 neighbors.
    
    # The next closest points in the triangular lattice to the origin that are not in Ring 1:
    # (2,0): dist 3
    # (1,1): dist 3
    # (-1,2): dist 3
    # (-2,1): dist 3
    # (-2,0): dist 3
    # (-1,-1): dist 3
    # (0,-2): dist 3
    # (1,-2): dist 3
    # There are 8 such points. The "Flower" of 11 likely takes 4 of these to maintain symmetry?
    # Or maybe it takes the 4 that align with the axes?
    # Actually, a common optimal packing for 11 hexagons is the "flower" plus 4 more.
    # Let's use the 4 points at angles 30, 150, 210, 330 degrees? No, those are gaps.
    # The points at distance 3 are at angles:
    # (3,0) -> 0 deg
    # (1.5, 2.598) -> 60 deg
    # etc.
    
    # Let's just use the 11 points from the triangular lattice that are closest to the origin.
    # Sorted by distance:
    # 0: (0,0)
    # 1-6: (1,0), (0,1), (-1,1), (-1,0), (0,-1), (1,-1)
    # 7-14: (2,0), (1,1), (-1,2), (-2,1), (-2,0), (-1,-1), (0,-2), (1,-2)
    # All 8 are at distance 3.
    # To make a symmetric cluster of 11, we can pick 4 of them.
    # However, the inspiration suggests a specific "flower" pattern.
    # Let's use the 4 points that form a square-like symmetry? No, hex symmetry.
    # Let's pick (2,0), (-2,0), (1,1), (-1,-1)? No.
    # Let's pick the 4 that are furthest apart?
    # Actually, for the optimization to work, the initial configuration just needs to be valid.
    # The optimizer will move them.
    # Let's use the 4 points: (2,0), (-2,0), (1,-2), (-1,2)?
    # Let's just take the first 4 from the sorted list of distance-3 points.
    
    u = np.array([1.5, np.sqrt(3)/2])
    v = np.array([0.0, np.sqrt(3)])
    
    lattice_coords = [
        (0, 0),
        (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
        (2, 0), (1, 1), (-1, 2), (-2, 1) # Picked 4 arbitrarily from the 8 available
    ]
    
    base_centers = np.array([i * u + j * v for i, j in lattice_coords])
    centers = base_centers * scale
    
    # --- 2. Pre-compute Base Vertices ---
    # We will compute vertices on the fly in the objective function for flexibility of rotation.
    # Base vertices at angle 0
    base_verts = np.array([
        [1.0, 0.0],
        [0.5, np.sqrt(3)/2],
        [-0.5, np.sqrt(3)/2],
        [-1.0, 0.0],
        [-0.5, -np.sqrt(3)/2],
        [0.5, -np.sqrt(3)/2]
    ])

    def compute_side(params):
        """
        params: [theta_global, phi_0, ..., phi_10, theta_outer]
        theta_global: rotation of the entire cluster (positions and inner hex orientations)
        phi_i: additional rotation of inner hex i relative to cluster
        theta_outer: rotation of the outer hexagon
        
        Returns: Minimum side length of the outer hexagon.
        """
        theta_global = params[0]
        phis = params[1:1+N]
        theta_outer = params[1+N]
        
        # Global rotation matrix
        c_g, s_g = np.cos(theta_global), np.sin(theta_global)
        R_g = np.array([[c_g, -s_g], [s_g, c_g]])
        
        # Rotate centers
        rotated_centers = centers @ R_g.T
        
        # Outer hexagon normals
        # The outer hexagon is centered at origin (we assume WLOG, or we can optimize center too? 
        # The task allows moving the outer hexagon. But for a symmetric cluster, centering it 
        # at the centroid is optimal. We'll fix center at 0 for simplicity and rely on 
        # rotation/side optimization. If the cluster is not symmetric, we might lose optimality.
        # However, the cluster is roughly symmetric.
        # Let's also allow the outer hexagon center to be optimized? 
        # The interface returns (inner, center, side, angle).
        # The evaluator checks if inner hexes are inside the outer hex defined by center, side, angle.
        # So we CAN optimize the outer center.
        # But that adds 2 more variables.
        # Let's stick to center=(0,0) for the outer hex for now, as the cluster is centered there.
        
        phi_rad = np.deg2rad(theta_outer)
        normals = np.array([
            [np.cos(phi_rad + k * np.pi / 3), np.sin(phi_rad + k * np.pi / 3)]
            for k in range(6)
        ])
        
        max_proj = 0.0
        
        for i in range(N):
            # Total rotation of inner hex i
            theta_i = theta_global + phis[i]
            c_i, s_i = np.cos(theta_i), np.sin(theta_i)
            R_i = np.array([[c_i, -s_i], [s_i, c_i]])
            
            # Vertices of inner hex i
            verts = base_verts @ R_i.T + rotated_centers[i]
            
            # Project onto outer normals
            dots = verts @ normals.T
            local_max = np.max(dots)
            if local_max > max_proj:
                max_proj = local_max
        
        # Side length L such that apothem = max_proj
        # Apothem = L * sqrt(3) / 2
        L = 2.0 * max_proj / np.sqrt(3)
        
        return L

    def smooth_objective(params, t=50.0):
        """
        Smooth approximation using log-sum-exp for the max projection.
        """
        theta_global = params[0]
        phis = params[1:1+N]
        theta_outer = params[1+N]
        
        c_g, s_g = np.cos(theta_global), np.sin(theta_global)
        R_g = np.array([[c_g, -s_g], [s_g, c_g]])
        rotated_centers = centers @ R_g.T
        
        phi_rad = np.deg2rad(theta_outer)
        normals = np.array([
            [np.cos(phi_rad + k * np.pi / 3), np.sin(phi_rad + k * np.pi / 3)]
            for k in range(6)
        ])
        
        all_dots = []
        for i in range(N):
            theta_i = theta_global + phis[i]
            c_i, s_i = np.cos(theta_i), np.sin(theta_i)
            R_i = np.array([[c_i, -s_i], [s_i, c_i]])
            verts = base_verts @ R_i.T + rotated_centers[i]
            dots = verts @ normals.T # shape (6, 6)
            all_dots.append(dots)
        
        all_dots = np.vstack(all_dots) # shape (66, 6)
        
        # Max projection per normal
        max_per_normal = np.max(all_dots, axis=0) # shape (6,)
        
        # Log-sum-exp approximation of max over the 6 normals
        m = np.max(max_per_normal)
        log_sum_exp = m + np.log(np.sum(np.exp(max_per_normal - m)))
        smooth_max = log_sum_exp / t
        
        return 2.0 * smooth_max / np.sqrt(3)

    # --- 3. Optimization ---
    
    # Initial guess
    # Global rotation 0, individual rotations 0, outer rotation 0
    x0 = np.zeros(1 + N + 1)
    
    best_score = float('inf')
    best_params = None
    
    # Phase 1: Multi-start Powell
    n_starts = 10
    rng = np.random.default_rng(42)
    
    for i in range(n_starts):
        if time.time() > deadline - 30:
            break
        
        if i == 0:
            x_start = x0.copy()
        elif i == 1:
            # Rotate outer by 30 deg
            x_start = np.zeros(1 + N + 1)
            x_start[-1] = np.pi/6
        elif i == 2:
            # Rotate all inner by 30 deg
            x_start = np.zeros(1 + N + 1)
            x_start[1:1+N] = np.pi/6
        else:
            # Random perturbations
            x_start = x0.copy()
            x_start += rng.uniform(-0.1, 0.1, size=1 + N + 1)
        
        try:
            res = scipy.optimize.minimize(
                compute_side, 
                x_start, 
                method='Powell',
                options={'maxiter': 1000, 'disp': False, 'ftol': 1e-12, 'xtol': 1e-12}
            )
            if res.fun < best_score:
                best_score = res.fun
                best_params = res.x.copy()
        except Exception:
            pass
    
    if best_params is None:
        # Fallback to initial
        best_params = x0
        best_score = compute_side(x0)

    # Phase 2: Nelder-Mead with Smooth Objective
    if time.time() < deadline - 15:
        try:
            res2 = scipy.optimize.minimize(
                smooth_objective, 
                best_params, 
                method='Nelder-Mead',
                options={'xatol': 1e-10, 'fatol': 1e-12, 'maxiter': 5000, 'adaptive': True}
            )
            true_score = compute_side(res2.x)
            if true_score < best_score:
                best_score = true_score
                best_params = res2.x.copy()
        except Exception:
            pass

    # Phase 3: Coordinate Descent / Scalar Refinement
    if time.time() < deadline - 5:
        # Refine each parameter individually using bounded scalar minimization
        for j in range(1 + N + 1):
            if time.time() > deadline - 3:
                break
            
            current_val = best_params[j]
            
            # Define bounds based on parameter type
            if j == 0 or j == 1 + N: # Global or Outer rotation
                bounds = (current_val - 0.5, current_val + 0.5)
            else: # Inner rotation
                bounds = (current_val - 0.5, current_val + 0.5)
            
            def obj_j(x):
                p = best_params.copy()
                p[j] = x
                return compute_side(p)
            
            try:
                res_j = scipy.optimize.minimize_scalar(
                    obj_j, 
                    bounds=bounds, 
                    method='bounded', 
                    options={'xatol': 1e-10}
                )
                if res_j.fun < best_score:
                    best_score = res_j.fun
                    best_params[j] = res_j.x
            except Exception:
                pass

    # --- 4. Construct Output ---
    
    theta_global = best_params[0]
    phis = best_params[1:1+N]
    theta_outer = best_params[1+N]
    
    # Rotate centers by global angle
    c_g, s_g = np.cos(theta_global), np.sin(theta_global)
    R_g = np.array([[c_g, -s_g], [s_g, c_g]])
    final_centers = centers @ R_g.T
    
    inner = []
    for i in range(N):
        x = float(final_centers[i, 0])
        y = float(final_centers[i, 1])
        ang_deg = float(np.rad2deg(theta_global + phis[i]))
        inner.append([x, y, ang_deg])
    
    # Final side length with safety margin
    final_side = float(best_score) + 1e-9
    
    # Outer hexagon center is (0,0)
    outer_center = (0.0, 0.0)
    outer_angle_deg = float(np.rad2deg(theta_outer))
    
    return inner, outer_center, final_side, outer_angle_deg

