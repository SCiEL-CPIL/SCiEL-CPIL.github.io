import numpy as np
import time

def _hex_verts(cx, cy, angle_rad, s=1.0):
    """Vertices of a regular hexagon."""
    angles = angle_rad + np.arange(6) * np.pi / 3
    return np.column_stack([cx + s * np.cos(angles), cy + s * np.sin(angles)])

def _min_side_for_cluster(centers, angles, outer_angle_rad):
    """
    For a fixed cluster and outer hexagon rotation, compute the minimum
    outer hexagon side length S such that all inner hexagons fit inside.
    
    The outer hexagon (centered at origin, side S, rotation outer_angle_rad)
    has 6 edges. Each edge is a half-plane constraint.
    
    For edge k with outward normal n_k, the constraint is:
        for all inner hexagon vertices v: dot(v, n_k) <= S * sqrt(3) / 2
    
    So S >= max over all inner vertices and all 6 edges of:
        dot(v, n_k) * 2 / sqrt(3)
    """
    # Outer hexagon edge normals (pointing outward)
    # For a hexagon with vertices at angles (outer_angle + 60*i), 
    # the edge between vertex i and vertex i+1 has outward normal at angle (outer_angle + 60*i + 30)
    normal_angles = outer_angle_rad + np.pi / 6 + np.arange(6) * np.pi / 3
    normals = np.column_stack([np.cos(normal_angles), np.sin(normal_angles)])
    
    # Compute all inner hexagon vertices
    all_verts = []
    for i in range(len(centers)):
        v = _hex_verts(centers[i, 0], centers[i, 1], angles[i])
        all_verts.append(v)
    all_verts = np.vstack(all_verts)
    
    # For each normal, compute max dot product over all inner vertices
    # S_min = max_k (max_v dot(v, n_k)) * 2 / sqrt(3)
    # Also need to handle the case where all dots are negative (vertex on opposite side)
    # Actually, the constraint is dot(v, n_k) <= S * apothem = S * sqrt(3)/2
    # So S >= dot(v, n_k) / (sqrt(3)/2) = 2*dot(v, n_k)/sqrt(3)
    # We need S >= max over all k,v of 2*dot(v,n_k)/sqrt(3)
    
    # Compute dot products: (60, 2) @ (6, 2)^T -> (60, 6)
    dots = all_verts @ normals.T  # shape (60, 6)
    max_dots = np.max(dots, axis=0)  # shape (6,)
    
    S_min = np.max(max_dots) * 2.0 / np.sqrt(3.0)
    
    return S_min

def _check_separation(centers, angles, eps=1e-6):
    """
    Check that all pairs of inner hexagons are separated.
    Two hexagons are separated if on at least one of their 12 edge normals,
    the projected intervals are apart by more than eps.
    
    For two convex polygons A and B, they are separated iff there exists a
    separating axis. For two regular hexagons with potentially different
    orientations, the separating axis must be parallel to one of their edges.
    """
    n = len(centers)
    verts_list = [_hex_verts(centers[i, 0], centers[i, 1], angles[i]) for i in range(n)]
    
    for i in range(n):
        for j in range(i + 1, n):
            A = verts_list[i]
            B = verts_list[j]
            
            # Get edge normals for both hexagons
            # For hexagon with vertices v[0..5], edge from v[k] to v[k+1]
            # Normal is perpendicular to edge, pointing outward
            def get_edge_normals(verts):
                normals = []
                for k in range(6):
                    e = verts[(k+1) % 6] - verts[k]
                    # Outward normal: rotate edge by -90 deg (or +90, need to check)
                    # For a CCW polygon, outward normal of edge (v[k]->v[k+1]) is (dy, -dx) normalized
                    # Actually for CCW, inward is (dy, -dx), outward is (-dy, dx)
                    # Let's compute: for hexagon centered at origin with vertex 0 at angle 0,
                    # vertex 1 at 60 deg. Edge from v0 to v1.
                    # v0 = (1,0), v1 = (0.5, sqrt(3)/2)
                    # edge = (-0.5, sqrt(3)/2)
                    # outward normal should point away from center, so roughly in direction (0.5, sqrt(3)/2) normalized? 
                    # Actually the outward normal of this edge points in the direction of angle 30 deg.
                    # (-dy, dx) = (-sqrt(3)/2, -0.5) -> that's inward
                    # (dy, -dx) = (sqrt(3)/2, 0.5) -> normalized: (sqrt(3)/2, 1/2) which is angle 30 deg. Yes!
                    e_norm = np.linalg.norm(e)
                    normal = np.array([e[1], -e[0]]) / e_norm
                    normals.append(normal)
                return np.array(normals)
            
            normals_A = get_edge_normals(A)
            normals_B = get_edge_normals(B)
            all_normals = np.vstack([normals_A, normals_B])
            
            # Project both polygons onto each normal
            separated = False
            for normal in all_normals:
                proj_A = A @ normal
                proj_B = B @ normal
                min_A = np.min(proj_A)
                max_A = np.max(proj_A)
                min_B = np.min(proj_B)
                max_B = np.max(proj_B)
                
                # Check if intervals are apart
                if max_A < min_B - eps or max_B < min_A - eps:
                    separated = True
                    break
            
            if not separated:
                return False
    
    return True

def _required_outer_side(centers, angles):
    """
    Find the minimum outer hexagon side length over all rotation angles.
    This is a 1D convex piecewise-linear minimization.
    """
    # The function S(theta) = max over 6 normals of (max over 60 vertices of dot(v, n(theta)))
    # Each term is of the form |a*cos(theta) + b*sin(theta)| = R*cos(theta - phi)
    # So S(theta) is the max of 360 cosine-like functions, which is convex in a periodic sense.
    
    # We can do a fine grid search + refinement
    # Since the function is periodic with period 60 deg (pi/3), search over [0, pi/3)
    
    # First, coarse grid
    n_coarse = 360
    thetas = np.linspace(0, np.pi / 3, n_coarse, endpoint=False)
    S_vals = np.array([_min_side_for_cluster(centers, angles, t) for t in thetas])
    
    # Find minimum
    idx = np.argmin(S_vals)
    S_best = S_vals[idx]
    
    # Refine around the minimum with finer grid
    theta_lo = thetas[(idx - 1 + n_coarse) % n_coarse]
    theta_hi = thetas[(idx + 1) % n_coarse]
    
    n_fine = 1000
    thetas_fine = np.linspace(theta_lo, theta_hi, n_fine)
    S_vals_fine = np.array([_min_side_for_cluster(centers, angles, t) for t in thetas_fine])
    
    idx_fine = np.argmin(S_vals_fine)
    S_best = S_vals_fine[idx_fine]
    theta_best = thetas_fine[idx_fine]
    
    return S_best, theta_best

def hexagon_packing_11():
    start_time = time.time()
    deadline = start_time + 620  # Safety margin
    
    sqrt3 = np.sqrt(3)
    
    # Define several candidate cluster configurations
    # Each is (centers, angles) for 11 unit hexagons
    
    candidates = []
    
    # Configuration 1: Triangular lattice 7-hex flower + 4 satellites
    # 7-hex flower: center + 6 neighbors at distance sqrt(3)
    c1 = np.array([
        [0, 0],
        [sqrt3, 0],
        [sqrt3/2, 1.5],
        [-sqrt3/2, 1.5],
        [-sqrt3, 0],
        [-sqrt3/2, -1.5],
        [sqrt3/2, -1.5],
        # 4 more hexagons in the gaps
        [0, 3.0],
        [0, -3.0],
        [2.5*sqrt3, 0],
        [-2.5*sqrt3, 0]
    ])
    a1 = np.zeros(11)
    candidates.append((c1, a1))
    
    # Configuration 2: Hexagonal arrangement - 1 center, 6 ring, 4 outer
    c2 = np.array([
        [0, 0],
        [sqrt3, 0],
        [sqrt3/2, 1.5],
        [-sqrt3/2, 1.5],
        [-sqrt3, 0],
        [-sqrt3/2, -1.5],
        [sqrt3/2, -1.5],
        [1.5*sqrt3, 1.5],
        [-1.5*sqrt3, 1.5],
        [1.5*sqrt3, -1.5],
        [-1.5*sqrt3, -1.5]
    ])
    a2 = np.zeros(11)
    candidates.append((c2, a2))
    
    # Configuration 3: Linear arrangement with offsets
    c3 = np.array([
        [0, 0],
        [sqrt3, 0],
        [2*sqrt3, 0],
        [3*sqrt3, 0],
        [-sqrt3, 0],
        [-2*sqrt3, 0],
        [sqrt3/2, 1.5],
        [-sqrt3/2, 1.5],
        [sqrt3/2, -1.5],
        [-sqrt3/2, -1.5],
        [0, 3.0]
    ])
    a3 = np.zeros(11)
    candidates.append((c3, a3))
    
    # Configuration 4: Compact 3x4 grid-like arrangement
    # Hexagonal grid: rows offset
    c4 = np.array([
        [0, 0],
        [sqrt3, 0],
        [2*sqrt3, 0],
        [sqrt3/2, 1.5],
        [1.5*sqrt3, 1.5],
        [2.5*sqrt3, 1.5],
        [-sqrt3/2, 1.5],
        [-1.5*sqrt3, 1.5],
        [sqrt3/2, -1.5],
        [1.5*sqrt3, -1.5],
        [2.5*sqrt3, -1.5]
    ])
    a4 = np.zeros(11)
    candidates.append((c4, a4))
    
    # Configuration 5: Star pattern
    angles_star = np.linspace(0, 2*np.pi, 6, endpoint=False)
    c5 = np.array([[0, 0]] + 
        [[2.5*sqrt3*np.cos(a), 2.5*sqrt3*np.sin(a)] for a in angles_star] +
        [[1.25*sqrt3*np.cos(a + np.pi/6), 1.25*sqrt3*np.sin(a + np.pi/6)] for a in angles_star[:4]])
    a5 = np.zeros(11)
    candidates.append((c5, a5))
    
    # Configuration 6: Dense triangular cluster (rows of 3,4,3,1)
    # Row 0: 3 hexagons
    # Row 1: 4 hexagons (offset)
    # Row 2: 3 hexagons
    # Row 3: 1 hexagon
    dy = 1.5  # vertical spacing
    dx = sqrt3  # horizontal spacing
    c6 = np.array([
        [-dx, 0], [0, 0], [dx, 0],  # Row 0: 3
        [-1.5*dx, dy], [-0.5*dx, dy], [0.5*dx, dy], [1.5*dx, dy],  # Row 1: 4
        [-dx, 2*dy], [0, 2*dy], [dx, 2*dy],  # Row 2: 3
        [0, 3*dy]  # Row 3: 1
    ])
    a6 = np.zeros(11)
    candidates.append((c6, a6))
    
    # Configuration 7: Compact with some rotated hexagons for better packing
    c7 = np.array([
        [0, 0],
        [sqrt3, 0],
        [sqrt3/2, 1.5],
        [-sqrt3/2, 1.5],
        [-sqrt3, 0],
        [-sqrt3/2, -1.5],
        [sqrt3/2, -1.5],
        [1.5*sqrt3, 1.5],
        [-1.5*sqrt3, 1.5],
        [1.5*sqrt3, -1.5],
        [-1.5*sqrt3, -1.5]
    ])
    # Rotate some outer hexagons by 30 degrees
    a7 = np.zeros(11)
    a7[7:] = np.deg2rad(30)
    candidates.append((c7, a7))
    
    # Configuration 8: Tighter cluster
    c8 = np.array([
        [0, 0],
        [sqrt3, 0],
        [sqrt3/2, 1.5],
        [-sqrt3/2, 1.5],
        [-sqrt3, 0],
        [-sqrt3/2, -1.5],
        [sqrt3/2, -1.5],
        [sqrt3, 3.0],
        [-sqrt3, 3.0],
        [sqrt3, -3.0],
        [-sqrt3, -3.0]
    ])
    a8 = np.zeros(11)
    candidates.append((c8, a8))
    
    # Evaluate all candidates
    best_S = np.inf
    best_centers = None
    best_angles = None
    best_outer_angle = 0
    
    for ci, (centers, angles) in enumerate(candidates):
        if time.time() > deadline:
            break
        
        # Check separation
        if not _check_separation(centers, angles):
            continue
        
        # Find optimal outer hexagon rotation and side
        S, theta = _required_outer_side(centers, angles)
        
        if S < best_S:
            best_S = S
            best_centers = centers.copy()
            best_angles = angles.copy()
            best_outer_angle = theta
    
    # If no valid configuration found, use a fallback
    if best_centers is None:
        # Fallback: simple grid
        best_centers = np.array([
            [0, 0],
            [sqrt3, 0],
            [2*sqrt3, 0],
            [sqrt3/2, 1.5],
            [1.5*sqrt3, 1.5],
            [2.5*sqrt3, 1.5],
            [-sqrt3/2, 1.5],
            [-1.5*sqrt3, 1.5],
            [sqrt3/2, -1.5],
            [1.5*sqrt3, -1.5],
            [2.5*sqrt3, -1.5]
        ])
        best_angles = np.zeros(11)
        best_S, best_outer_angle = _required_outer_side(best_centers, best_angles)
    
    # Try local refinement on the best configuration
    if time.time() < deadline - 10:
        # Perturb centers slightly and re-evaluate
        best_S_refined = best_S
        best_centers_refined = best_centers.copy()
        best_angles_refined = best_angles.copy()
        best_outer_angle_refined = best_outer_angle
        
        # Try small perturbations
        for trial in range(50):
            if time.time() > deadline - 5:
                break
            
            # Random small perturbation
            scale = 0.1 * (1.0 - trial / 50.0)
            pert_centers = best_centers + scale * np.random.randn(11, 2) * 0.1
            pert_angles = best_angles + scale * np.random.randn(11) * 0.05
            
            if not _check_separation(pert_centers, pert_angles):
                continue
            
            S_trial, theta_trial = _required_outer_side(pert_centers, pert_angles)
            
            if S_trial < best_S_refined:
                best_S_refined = S_trial
                best_centers_refined = pert_centers.copy()
                best_angles_refined = pert_angles.copy()
                best_outer_angle_refined = theta_trial
        
        if best_S_refined < best_S:
            best_S = best_S_refined
            best_centers = best_centers_refined
            best_angles = best_angles_refined
            best_outer_angle = best_outer_angle_refined
    
    # Final validation
    if not _check_separation(best_centers, best_angles):
        # Use a definitely valid fallback
        best_centers = np.array([
            [0, 0],
            [sqrt3, 0],
            [2*sqrt3, 0],
            [sqrt3/2, 1.5],
            [1.5*sqrt3, 1.5],
            [2.5*sqrt3, 1.5],
            [-sqrt3/2, 1.5],
            [-1.5*sqrt3, 1.5],
            [sqrt3/2, -1.5],
            [1.5*sqrt3, -1.5],
            [2.5*sqrt3, -1.5]
        ])
        best_angles = np.zeros(11)
        best_S, best_outer_angle = _required_outer_side(best_centers, best_angles)
    
    # Ensure S >= sqrt(11)
    min_S = np.sqrt(11)
    if best_S < min_S:
        best_S = min_S
    
    # Construct return values
    # inner: (11, 3) with (x, y, angle_deg)
    inner = np.column_stack([best_centers, np.rad2deg(best_angles)])
    
    # outer: (3,) with (x, y, angle_deg)
    outer_angle_deg = np.rad2deg(best_outer_angle)
    outer = np.array([0.0, 0.0, outer_angle_deg])
    
    # side: float
    side = float(best_S)
    
    return inner, outer, side
