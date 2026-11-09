import numpy as np
import time
from scipy.optimize import minimize

def _hex_vertices(cx, cy, d, a):
    a_rad = np.deg2rad(a)
    angles = a_rad + np.arange(6) * np.pi / 3
    return np.column_stack([cx + d * np.cos(angles), cy + d * np.sin(angles)])

def _min_outer_side(positions, angle_deg):
    """
    Compute the minimum side length of a regular hexagon with the given angle
    that encloses all unit hexagons at the given positions.
    """
    unit_hex_verts = _hex_vertices(0, 0, 1, 0)
    
    # All inner hexagon vertices
    all_verts = positions[:, np.newaxis, :] + unit_hex_verts[np.newaxis, :, :]
    all_verts_flat = all_verts.reshape(-1, 2)
    
    a_rad = np.deg2rad(angle_deg)
    
    # Outward normals of the outer hexagon edges
    # Edge i connects vertex i and i+1.
    # Outward normal angle is a + 60i + 30 degrees.
    normals = []
    for i in range(6):
        n_angle = a_rad + i * np.pi/3 + np.pi/6
        n_unit = np.array([np.cos(n_angle), np.sin(n_angle)])
        normals.append(n_unit)
    normals = np.array(normals)
    
    # Project all vertices onto normals
    projections = all_verts_flat @ normals.T # (36, 6)
    max_projections = np.max(projections, axis=0) # (6,)
    
    # s * sqrt(3)/2 >= max_proj  =>  s >= 2 * max_proj / sqrt(3)
    min_s = np.max(max_projections) * 2 / np.sqrt(3)
    
    return min_s

def _check_validity(inner, outer, s):
    """
    Validates the solution according to the task contract.
    Returns True if valid, False otherwise.
    """
    # Check finite
    if not np.all(np.isfinite(inner)) or not np.all(np.isfinite(outer)) or not np.isfinite(s):
        return False
    
    if s < np.sqrt(12) - 1e-9:
        return False
        
    # Check outer vertices distinct
    outer_verts = _hex_vertices(outer[0], outer[1], s, outer[2])
    for i in range(6):
        for j in range(i+1, 6):
            if np.linalg.norm(outer_verts[i] - outer_verts[j]) < 1e-9:
                return False
                
    # Check unit edge length of inner hexagons
    for i in range(12):
        cx, cy, ang = inner[i]
        verts = _hex_vertices(cx, cy, 1, ang)
        for k in range(6):
            v1 = verts[k]
            v2 = verts[(k+1)%6]
            if abs(np.linalg.norm(v2 - v1) - 1.0) > 1e-8:
                return False

    # Check separation (SAT)
    inner_verts_list = []
    for i in range(12):
        cx, cy, ang = inner[i]
        inner_verts_list.append(_hex_vertices(cx, cy, 1, ang))
        
    for i in range(12):
        for j in range(i+1, 12):
            va = inner_verts_list[i]
            vb = inner_verts_list[j]
            
            separated = False
            # Check axes from A
            for k in range(6):
                v1 = va[k]
                v2 = va[(k+1)%6]
                e = v2 - v1
                n = np.array([-e[1], e[0]])
                norm = np.linalg.norm(n)
                if norm < 1e-12: continue
                n = n / norm
                proj_a = np.dot(va, n)
                proj_b = np.dot(vb, n)
                if np.max(proj_a) < np.min(proj_b) - 1e-6 or np.max(proj_b) < np.min(proj_a) - 1e-6:
                    separated = True
                    break
            if separated: continue
            
            # Check axes from B
            for k in range(6):
                v1 = vb[k]
                v2 = vb[(k+1)%6]
                e = v2 - v1
                n = np.array([-e[1], e[0]])
                norm = np.linalg.norm(n)
                if norm < 1e-12: continue
                n = n / norm
                proj_a = np.dot(va, n)
                proj_b = np.dot(vb, n)
                if np.max(proj_a) < np.min(proj_b) - 1e-6 or np.max(proj_b) < np.min(proj_a) - 1e-6:
                    separated = True
                    break
            if not separated:
                return False
                
    # Check inner vertices inside outer
    for i in range(12):
        for v in inner_verts_list[i]:
            inside = True
            for k in range(6):
                p1 = outer_verts[k]
                p2 = outer_verts[(k+1)%6]
                e = p2 - p1
                cross_val = e[0] * (v[1] - p1[1]) - e[1] * (v[0] - p1[0])
                if cross_val < -1e-6:
                    inside = False
                    break
            if not inside:
                return False
                
    return True

def hexagon_packing_12():
    start_time = time.time()
    deadline = start_time + 580 # Safety margin
    
    best_score = 0.0
    best_inner = None
    best_outer = None
    best_s = None
    
    # Define the rigid cluster of 12 hexagons in a hexagonal lattice
    # Center at (0,0)
    # Ring 1: 6 hexagons at distance sqrt(3) from center? No, edge-to-edge distance is 1.5.
    # Vertices of unit hexagon at angle 0:
    # (1, 0), (0.5, sqrt(3)/2), (-0.5, sqrt(3)/2), (-1, 0), (-0.5, -sqrt(3)/2), (0.5, -sqrt(3)/2)
    # To pack 12 hexagons in a hexagonal shape (1 in center, 6 around, 5 more to complete a larger hexagon?)
    # A hexagon of side 2 has 1 + 6 + 12 = 19 hexagons.
    # A hexagon of side 1 has 1 hexagon.
    # We need 12.
    # Common packing: 1 center, 6 neighbors, and 5 of the next ring? Or a 3x3 square-ish block?
    # Let's use the "hexagonal number" arrangement minus some corners or a specific compact shape.
    # The most compact shape for 12 is often a 3x3 block of a triangular lattice, which forms a hexagon-like shape.
    # Let's define positions in a coordinate system where the lattice vectors are u = (1.5, 0) and v = (0.75, 1.5*sqrt(3)/2) = (0.75, 1.299)
    # Actually, simpler:
    # Center (0,0)
    # Neighbors at distance 1.5:
    # (1.5, 0), (0.75, 1.299), (-0.75, 1.299), (-1.5, 0), (-0.75, -1.299), (0.75, -1.299)
    # Next ring (distance 3 from center along axes, or 1.5 from neighbors):
    # (3, 0), (2.25, 2.598), (0, 2.598), (-2.25, 2.598), (-3, 0), (-2.25, -2.598), (0, -2.598), (2.25, -2.598)
    # Total so far: 1 + 6 + 8 = 15. Too many.
    # We need 12.
    # Let's take the center, the 6 neighbors, and 5 of the 8 next-ring positions.
    # To be symmetric, we can't just drop one.
    # Maybe a different shape?
    # What if we use a 2x3 rectangle in the triangular lattice?
    # Or a hexagon of side 1 (1 hex) + 6 neighbors + 5 more?
    # Let's try a shape that is close to a hexagon of side 1.5? No, that's 7.
    # Let's use the 12 positions that form a "hexagon" of side 2 but missing 7 corners? No.
    
    # Let's stick to the 1 center + 6 neighbors + 5 from the next ring.
    # To maintain symmetry, maybe we don't use a perfect hexagonal lattice for all 12?
    # The inspiration mentions "Rigid cluster representation". This implies the relative positions are fixed.
    # The most compact packing of 12 unit hexagons is likely a subset of the triangular lattice.
    # Let's define the 12 positions as the 12 points in a hexagonal lattice that are closest to the origin.
    # Distances from origin in triangular lattice (edge 1.5):
    # 0: (0,0) -> 1
    # 1.5: 6 points
    # 3.0: 6 points (along axes)
    # 1.5*sqrt(3) ~ 2.598: 6 points (along diagonals)
    # 3*sqrt(3)/2 * sqrt(3)? No.
    # Let's list distances squared:
    # (0,0): 0
    # (1.5, 0): 2.25 (6 of these)
    # (0, 2.598): 6.75 (6 of these)
    # (3, 0): 9 (6 of these)
    # (2.25, 2.598): 5.0625 + 6.75 = 11.8125 (6 of these)
    
    # We want 12 points.
    # 1 (center) + 6 (dist 1.5) = 7.
    # Next closest are 6 points at dist 2.598.
    # Total 13.
    # If we take 1 + 6 + 5 of the 2.598 ring, we break symmetry.
    # However, the cluster can be rotated.
    # Let's try the 7-point hexagon (side 1.5) + 5 points from the next ring.
    # Which 5? To make it compact, we should probably pick ones that fill the "corners" of the next hexagon (side 2).
    # The 6 points at 2.598 are at angles 0, 60, ...
    # The 6 points at 3.0 are at angles 30, 90, ...
    # Wait, the lattice points are:
    # (1.5, 0) -> angle 0
    # (0.75, 1.299) -> angle 60
    # (-0.75, 1.299) -> angle 120
    # (-1.5, 0) -> angle 180
    # (-0.75, -1.299) -> angle 240
    # (0.75, -1.299) -> angle 300
    
    # Next ring (distance 2.598 from center? No, distance from center is sqrt(0.75^2 + 1.299^2) = sqrt(0.5625 + 1.687) = sqrt(2.25) = 1.5.
    # Wait. The neighbors are at distance 1.5.
    # The next ring in a hexagonal lattice (where nearest neighbor is 1.5) has distance 1.5 * sqrt(3) = 2.598?
    # Let's check (2.25, 2.598). Dist^2 = 2.25^2 + 2.598^2 = 5.0625 + 6.75 = 11.8125. Dist = 3.437.
    # Let's check (3, 0). Dist = 3.
    # Let's check (0, 2.598). Dist = 2.598.
    # So the points at (0, 2.598) are closer than (3,0) or (2.25, 2.598).
    # So the 12th point should be one of the (0, 2.598) type.
    # There are 6 such points.
    # So 1 + 6 + 5 of these 6.
    
    # Let's define the 12 positions:
    # 0: (0,0)
    # 1-6: The 6 neighbors
    # 7-11: 5 of the 6 points at distance 2.598.
    # Which one to omit? It doesn't matter for the rigid cluster, as we can rotate the whole cluster.
    # We will omit the one at angle 0 (i.e., (2.598, 0) ? No, (0, 2.598) is angle 90).
    # Let's just pick 5 arbitrarily, say angles 90, 150, 210, 270, 330. Omit 30?
    # Actually, due to the symmetry of the outer hexagon (60 deg), the specific choice of which 5 might not matter much if we optimize the cluster rotation.
    
    h = np.sqrt(3)/2 # 0.866
    # Neighbors
    p1 = np.array([1.5, 0])
    p2 = np.array([0.75, 1.5*h]) # 0.75, 1.299
    p3 = np.array([-0.75, 1.5*h])
    p4 = np.array([-1.5, 0])
    p5 = np.array([-0.75, -1.5*h])
    p6 = np.array([0.75, -1.5*h])
    
    # Next ring candidates (distance 1.5*sqrt(3) = 2.598)
    # These are at (1.5, 1.5*h*2)? No.
    # In triangular lattice with spacing 1.5:
    # Basis vectors u = (1.5, 0), v = (0.75, 1.5*h)
    # Points are i*u + j*v.
    # (0,0)
    # (1,0) -> (1.5, 0)
    # (0,1) -> (0.75, 1.299)
    # (-1,1) -> (-0.75, 1.299)
    # (-1,0) -> (-1.5, 0)
    # (0,-1) -> (0.75, -1.299)
    # (1,-1) -> (1.5, -1.299)
    
    # Next closest:
    # (2,0) -> (3, 0)
    # (1,1) -> (2.25, 2.598)
    # (0,2) -> (1.5, 3.897) -- Wait.
    # v = (0.75, 1.299). 2v = (1.5, 2.598).
    # So (0,2) is (1.5, 2.598). Dist^2 = 2.25 + 6.75 = 9. Dist=3.
    # (1,1) is (2.25, 2.598). Dist^2 = 5.0625 + 6.75 = 11.8125.
    # So the points at distance 3 are:
    # (3,0), (-3,0), (1.5, 2.598), (-1.5, 2.598), (1.5, -2.598), (-1.5, -2.598).
    # There are 6 of these.
    # Are there any closer?
    # (1, -1) is (0.75, -1.299) which is a neighbor.
    # So the 7th shell (distance 3) has 6 points.
    # So 1 + 6 + 5 of these 6 distance-3 points.
    
    p7 = np.array([3, 0])
    p8 = np.array([1.5, 2.598])
    p9 = np.array([-1.5, 2.598])
    p10 = np.array([-3, 0])
    p11 = np.array([-1.5, -2.598])
    # Omitting (1.5, -2.598)
    
    cluster_pos = np.array([
        [0, 0],
        p1, p2, p3, p4, p5, p6,
        p7, p8, p9, p10, p11
    ])
    
    # We will optimize:
    # 1. Translation of cluster (tx, ty)
    # 2. Rotation of cluster (theta)
    # 3. Rotation of outer hexagon (phi)
    
    # Objective: minimize s
    # s depends on the positions of the vertices of the rotated cluster and the angle phi.
    
    def objective(x):
        tx, ty, theta_deg, phi_deg = x
        theta_rad = np.deg2rad(theta_deg)
        phi_rad = np.deg2rad(phi_deg)
        
        # Rotate cluster
        c = np.cos(theta_rad)
        s_ = np.sin(theta_rad)
        R = np.array([[c, -s_], [s_, c]])
        
        pos_rot = cluster_pos @ R.T
        pos_trans = pos_rot + np.array([tx, ty])
        
        # Calculate min outer side
        s_val = _min_outer_side(pos_trans, phi_deg)
        
        return s_val
    
    # Multi-start
    # Since the problem is periodic and symmetric, we can sample theta and phi.
    # theta in [0, 60), phi in [0, 60).
    # tx, ty should be around 0.
    
    np.random.seed(42)
    
    num_restarts = 50
    for r in range(num_restarts):
        if time.time() > deadline:
            break
        
        # Initial guess
        if r == 0:
            x0 = np.array([0.0, 0.0, 0.0, 0.0])
        else:
            tx0 = np.random.uniform(-1, 1)
            ty0 = np.random.uniform(-1, 1)
            theta0 = np.random.uniform(0, 60)
            phi0 = np.random.uniform(0, 60)
            x0 = np.array([tx0, ty0, theta0, phi0])
            
        try:
            res = minimize(
                objective, 
                x0, 
                method='Nelder-Mead', 
                options={'maxiter': 2000, 'xatol': 1e-8, 'fatol': 1e-10, 'adaptive': True}
            )
        except Exception:
            continue
            
        x_opt = res.x
        tx, ty, theta_deg, phi_deg = x_opt
        
        # Construct candidate
        theta_rad = np.deg2rad(theta_deg)
        c = np.cos(theta_rad)
        s_ = np.sin(theta_rad)
        R = np.array([[c, -s_], [s_, c]])
        
        pos_rot = cluster_pos @ R.T
        pos_trans = pos_rot + np.array([tx, ty])
        
        inner_candidate = np.zeros((12, 3))
        inner_candidate[:, :2] = pos_trans
        inner_candidate[:, 2] = theta_deg
        
        outer_candidate = np.array([0, 0, phi_deg])
        
        s_candidate = _min_outer_side(pos_trans, phi_deg)
        
        # Check validity
        if _check_validity(inner_candidate, outer_candidate, s_candidate):
            score_candidate = 1.0 / s_candidate
            if score_candidate > best_score:
                best_score = score_candidate
                best_inner = inner_candidate
                best_outer = outer_candidate
                best_s = s_candidate

    # If we found a valid solution, return it
    if best_inner is not None:
        return best_inner, best_outer, best_s
    else:
        # Fallback to a known valid configuration
        inner_fallback = np.array([
            [0, 0, 0],
            [2.5, 0, 0],
            [-2.5, 0, 0],
            [1.25, 2.165, 0],
            [-1.25, 2.165, 0],
            [1.25, -2.165, 0],
            [-1.25, -2.165, 0],
            [3.75, 2.165, 0],
            [-3.75, 2.165, 0],
            [3.75, -2.165, 0],
            [-3.75, -2.165, 0],
            [0, 4.33, 0]
        ])
        outer_fallback = np.array([0, 0, 0])
        s_fallback = _min_outer_side(inner_fallback[:, :2], 0)
        
        if s_fallback < np.sqrt(12):
            s_fallback = np.sqrt(12)
            
        return inner_fallback, outer_fallback, s_fallback

# EVOLVE-BLOCK-END
