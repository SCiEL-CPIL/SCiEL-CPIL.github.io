import numpy as np
import time
from scipy.optimize import minimize

N = 12

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s limit

    def hexagon_vertices(cx, cy, side, angle_deg):
        angle_rad = np.radians(angle_deg)
        angles = angle_rad + np.radians(np.arange(0, 360, 60))
        return np.column_stack([
            cx + side * np.cos(angles),
            cy + side * np.sin(angles)
        ])

    def polygons_disjoint_sat(p1, p2, eps=1e-4):
        """
        Returns True if two convex polygons are disjoint (allowing touching with eps tolerance).
        Uses Separating Axis Theorem (SAT).
        """
        for i in range(len(p1)):
            j = (i + 1) % len(p1)
            edge = p1[j] - p1[i]
            normal = np.array([-edge[1], edge[0]])
            norm = np.linalg.norm(normal)
            if norm < 1e-10:
                continue
            normal /= norm
            
            proj1 = p1.dot(normal)
            proj2 = p2.dot(normal)
            
            min1, max1 = np.min(proj1), np.max(proj1)
            min2, max2 = np.min(proj2), np.max(proj2)
            
            if max1 < min2 - eps or max2 < min1 - eps:
                return True
        
        for i in range(len(p2)):
            j = (i + 1) % len(p2)
            edge = p2[j] - p2[i]
            normal = np.array([-edge[1], edge[0]])
            norm = np.linalg.norm(normal)
            if norm < 1e-10:
                continue
            normal /= norm
            
            proj1 = p1.dot(normal)
            proj2 = p2.dot(normal)
            
            min1, max1 = np.min(proj1), np.max(proj1)
            min2, max2 = np.min(proj2), np.max(proj2)
            
            if max1 < min2 - eps or max2 < min1 - eps:
                return True
        
        return False

    def check_validity(positions, angles):
        """
        positions: (N, 2) array
        angles: (N,) array
        Returns True if all are pairwise disjoint.
        """
        vertices_list = []
        for i in range(N):
            x, y = positions[i]
            ang = angles[i]
            verts = hexagon_vertices(x, y, 1.0, ang)
            vertices_list.append(verts)
        
        for i in range(N):
            for j in range(i + 1, N):
                if not polygons_disjoint_sat(vertices_list[i], vertices_list[j]):
                    return False
        return True

    def compute_outer_side(vertices_all, outer_center, outer_angle_deg):
        """
        Computes the minimum side length of a regular hexagon 
        centered at outer_center with rotation outer_angle_deg 
        that contains all points in vertices_all.
        
        vertices_all: (M, 2) array of points
        outer_center: (2,) array
        outer_angle_deg: float
        
        Returns: side length S
        """
        # Translate points relative to outer center
        pts = vertices_all - outer_center
        
        # Rotate points by -outer_angle to align with standard hexagon
        theta = np.radians(-outer_angle_deg)
        c, s = np.cos(theta), np.sin(theta)
        rot_pts = np.column_stack([
            pts[:, 0] * c - pts[:, 1] * s,
            pts[:, 0] * s + pts[:, 1] * c
        ])
        
        x = rot_pts[:, 0]
        y = rot_pts[:, 1]
        
        # For a standard regular hexagon (vertex at (S,0)), the constraints for a point (x,y) to be inside are:
        # |x| <= S
        # |y| <= S * sqrt(3) / 2  (Wait, this is for the bounding box, not the hexagon itself)
        
        # Correct inequalities for regular hexagon (flat top? No, vertex at right):
        # The hexagon is the intersection of 3 strips.
        # Normal 1: (1, 0). Distance |x| <= S * cos(pi/6) ? No.
        
        # Let's use the support function approach or projection.
        # A regular hexagon of side S centered at origin with one vertex at (S, 0) has vertices:
        # (S, 0), (S/2, S*sqrt(3)/2), (-S/2, S*sqrt(3)/2), (-S, 0), (-S/2, -S*sqrt(3)/2), (S/2, -S*sqrt(3)/2)
        
        # The width in the direction of the normal to a side (e.g., vertical side? No, there are no vertical sides in this orientation).
        # The sides are:
        # 1. Top-Right: connecting (S/2, S*sqrt(3)/2) and (S, 0). Vector: (S/2, -S*sqrt(3)/2). Normal: (sqrt(3)/2, 1/2).
        #    Distance from origin to this side is S * sqrt(3) / 2.
        #    So the constraint is: dot(p, n1) <= S * sqrt(3) / 2.
        
        # There are 6 sides. The normals are at angles 30, 90, 150, 210, 270, 330 degrees?
        # Let's check.
        # Vertex (S,0) is at angle 0.
        # Side 1 connects angle 0 and 60. Midpoint angle 30. Normal angle 30.
        # Side 2 connects angle 60 and 120. Midpoint angle 90. Normal angle 90.
        
        # So the normals are at 30, 90, 150, 210, 270, 330.
        # The distance from center to each side (apothem) is a = S * sqrt(3) / 2.
        
        # For a point p to be inside, we must have:
        # |dot(p, n_k)| <= a  for k=1..6
        
        # So S = (2 / sqrt(3)) * max_k ( |dot(p, n_k)| )
        
        normals = np.array([
            [np.cos(np.radians(30)), np.sin(np.radians(30))],
            [np.cos(np.radians(90)), np.sin(np.radians(90))],
            [np.cos(np.radians(150)), np.sin(np.radians(150))],
            [np.cos(np.radians(210)), np.sin(np.radians(210))],
            [np.cos(np.radians(270)), np.sin(np.radians(270))],
            [np.cos(np.radians(330)), np.sin(np.radians(330))]
        ])
        
        # Project all points onto all normals
        # dots: (M, 6)
        dots = np.abs(rot_pts.dot(normals.T))
        max_dot = np.max(dots)
        
        return max_dot * 2.0 / np.sqrt(3.0)

    def objective(params):
        if time.time() > deadline:
            return 1e10
            
        # params structure:
        # 0: r1 (radius of ring 1, 6 hexagons)
        # 1: r2 (radius of ring 2, 5 hexagons)
        # 2: outer_angle
        # 3: outer_center_x
        # 4: outer_center_y
        # 5..: angles of the 12 hexagons? 
        # For simplicity and speed, let's fix the relative angles of the inner hexagons to 0 initially, 
        # or allow a global rotation for the inner set? 
        # The current program allows individual rotation. 
        # However, for a "flower" packing, aligning them with the grid (0 deg) is usually optimal.
        # Let's allow a global rotation for the inner hexagons to optimize the fit with the outer.
        
        r1 = params[0]
        r2 = params[1]
        outer_angle = params[2]
        ocx = params[3]
        ocy = params[4]
        inner_global_rot = params[5]
        
        # Generate positions
        positions = np.zeros((N, 2))
        angles = np.zeros(N)
        
        # Center hexagon
        positions[0] = [0, 0]
        angles[0] = inner_global_rot
        
        # Ring 1: 6 hexagons at distance r1
        # Ideally r1 = sqrt(3) for touching.
        for i in range(6):
            theta = np.radians(60 * i)
            positions[i+1] = [r1 * np.cos(theta), r1 * np.sin(theta)]
            angles[i+1] = inner_global_rot
            
        # Ring 2: 5 hexagons at distance r2
        # To make a compact cluster, place them in the "gaps" or extending the arms.
        # A common compact shape for 12 is a hexagon of side 2 (which has 19 hexagons).
        # We need to remove 7.
        # Let's try a different topology: 
        # 1 center
        # 6 around
        # 5 more placed at angles 0, 72, 144, 216, 288? No, that's a pentagon.
        # Let's stick to the hexagonal lattice.
        # The 12 hexagons can be arranged as:
        # Row 0: 1
        # Row 1: 3
        # Row 2: 5
        # Row 3: 3
        # Total 12. This is a "lozenge" or rhombus shape?
        # Or simply take the first 12 from the hexagonal spiral.
        # Spiral order: (0,0), (1,0), (0,1), (-1,1), (-1,0), (0,-1), (1,-1), (2,0), (1,1), (0,2), (-1,2), (-2,1)
        # This shape is roughly circular.
        
        # Let's define the 12 positions in axial coords (q,r)
        # and scale them by a factor? No, the distances are fixed by the lattice if they are touching.
        # But we can "stretch" the lattice slightly if needed to allow rotation or better fit?
        # Actually, if they are disjoint, we can move them closer or further.
        # To minimize the outer hexagon, we want them as compact as possible.
        # So they should be touching or nearly touching.
        # The base positions are fixed by the lattice topology.
        # We can apply a uniform scaling factor 'scale' to the radii?
        # If we scale, we might create gaps, which is fine, but if we scale down, they overlap.
        # So we need to ensure disjointness.
        
        # Let's use the 12 axial coordinates from the spiral.
        coords = [
            (0, 0),
            (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
            (2, 0), (1, 1), (0, 2), (-1, 2), (-2, 1)
        ]
        
        # Calculate base positions
        base_pos = np.zeros((N, 2))
        for i, (q, r) in enumerate(coords):
            base_pos[i, 0] = np.sqrt(3) * (q + r / 2.0)
            base_pos[i, 1] = 1.5 * r
            
        # Apply scaling? 
        # If we scale by k < 1, they overlap.
        # If we scale by k > 1, they separate.
        # We want the smallest outer hexagon, so we want k as small as possible such that they are disjoint.
        # So k should be the minimum scaling that keeps them disjoint.
        # Since they are disjoint at k=1 (touching), any k >= 1 works.
        # So k=1 is optimal for compactness? 
        # Wait, if we rotate the whole cluster, the bounding box changes.
        # Also, we can rotate individual hexagons.
        
        # Let's optimize:
        # 1. Global rotation of the cluster (which effectively rotates the positions)
        # 2. Individual rotation of each hexagon (to allow tighter packing if the lattice allows? 
        #    In a hexagonal lattice, rotating individual hexagons by 30 degrees allows them to nest? 
        #    No, regular hexagons tile the plane. If you rotate one, it overlaps its neighbors unless you move them.
        #    So for a fixed lattice topology, the positions are likely fixed relative to each other 
        #    if we want them to touch.
        #    However, if we allow them to NOT touch, we might rotate them to fit better in the outer hexagon.
        #    But moving them to allow rotation will likely increase the outer size.
        #    So assume fixed relative positions (touching lattice) and optimize:
        #    - Global rotation of the cluster
        #    - Individual rotation of each hexagon (maybe 0 or 30 deg)
        #    - Outer hexagon rotation and center
        
        # Let's simplify: 
        # Fix the relative positions to the base lattice (k=1).
        # Optimize: 
        #   - Global rotation of the cluster (phi)
        #   - Outer hexagon rotation (psi)
        #   - Outer hexagon center (cx, cy)
        #   - Individual hexagon angles (alpha_i) - but this is 12 variables.
        #     Let's assume all inner hexagons have the same angle relative to the cluster?
        #     Or just 0?
        #     Let's try optimizing a global inner angle and the cluster rotation.
        
        # Revised params:
        # 0: cluster_rotation (phi)
        # 1: inner_hex_angle (alpha) - all inner hexagons rotated by this amount relative to cluster?
        #     Actually, if we rotate the cluster, the hexagons rotate with it.
        #     If we also rotate the hexagons individually, we change their shape relative to the lattice.
        #     Let's just use cluster_rotation and outer_rotation.
        #     And maybe a "stretch" factor? No, keep it simple.
        
        # Let's add a "gap" parameter? No, k=1 is minimal for disjointness.
        
        # So:
        # phi = params[0]
        # outer_angle = params[1]
        # ocx = params[2]
        # ocy = params[3]
        # inner_angle_offset = params[4] (all inner hexagons rotated by this)
        
        phi = params[0]
        outer_angle = params[1]
        ocx = params[2]
        ocy = params[3]
        inner_angle_offset = params[4]
        
        # Rotate base positions by phi
        c_phi, s_phi = np.cos(phi), np.sin(phi)
        rot_mat = np.array([[c_phi, -s_phi], [s_phi, c_phi]])
        positions = base_pos.dot(rot_mat.T)
        
        # Angles
        angles = np.full(N, inner_angle_offset + np.degrees(phi)) # If cluster rotates, hexagons should rotate to maintain orientation? 
        # Actually, if we rotate the positions, the "up" direction of the hexagon should probably rotate too 
        # if we want to maintain the "flat" side alignment relative to the lattice.
        # Let's assume the hexagons are rigid bodies attached to the lattice.
        
        # Check validity
        if not check_validity(positions, angles):
            return 1e10
            
        # Get all vertices
        all_verts = []
        for i in range(N):
            x, y = positions[i]
            ang = angles[i]
            verts = hexagon_vertices(x, y, 1.0, ang)
            all_verts.append(verts)
            
        all_verts = np.vstack(all_verts)
        
        # Compute outer side
        s = compute_outer_side(all_verts, np.array([ocx, ocy]), outer_angle)
        
        return s

    # Initial guess
    # Cluster rotation: 0
    # Outer angle: 0
    # Center: 0,0
    # Inner angle offset: 0
    x0 = np.array([0.0, 0.0, 0.0, 0.0, 0.0])
    
    # Bounds
    bounds = [
        (-np.pi, np.pi),   # cluster rotation
        (-np.pi/6, np.pi/6), # outer rotation (symmetry 60 deg)
        (-5, 5),           # center x
        (-5, 5),           # center y
        (-np.pi/6, np.pi/6) # inner angle offset
    ]
    
    best_result = None
    best_val = 1e10
    
    # Try a few starting points for outer rotation to avoid local minima
    start_angles = [0.0, np.pi/12, -np.pi/12, np.pi/6]
    
    for sa in start_angles:
        if time.time() > deadline - 10:
            break
            
        x0_try = x0.copy()
        x0_try[1] = sa
        
        try:
            res = minimize(
                objective,
                x0_try,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 200, 'ftol': 1e-8}
            )
            
            if res.fun < best_val:
                best_val = res.fun
                best_result = res.x
                
        except Exception:
            pass
            
    if best_result is None:
        # Fallback to initial valid config
        best_result = x0
        best_val = objective(x0)
        
    # Extract final parameters
    phi = best_result[0]
    outer_angle = best_result[1]
    ocx = best_result[2]
    ocy = best_result[3]
    inner_angle_offset = best_result[4]
    
    # Recalculate positions
    base_pos = np.zeros((N, 2))
    coords = [
        (0, 0),
        (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
        (2, 0), (1, 1), (0, 2), (-1, 2), (-2, 1)
    ]
    for i, (q, r) in enumerate(coords):
        base_pos[i, 0] = np.sqrt(3) * (q + r / 2.0)
        base_pos[i, 1] = 1.5 * r
        
    c_phi, s_phi = np.cos(phi), np.sin(phi)
    rot_mat = np.array([[c_phi, -s_phi], [s_phi, c_phi]])
    positions = base_pos.dot(rot_mat.T)
    
    angles = np.full(N, inner_angle_offset + np.degrees(phi))
    
    # Construct inner list
    inner = []
    for i in range(N):
        x = positions[i, 0]
        y = positions[i, 1]
        ang = angles[i]
        inner.append([float(x), float(y), float(ang)])
        
    # Outer hexagon
    return inner, (float(ocx), float(ocy)), float(best_val), float(np.degrees(outer_angle))

# EVOLVE-BLOCK-END
