import numpy as np
import math
import time

N = 12
START_TIME = time.time()
TIME_LIMIT = 130.0  # Keep a safety margin before 150s

def construct():
    """
    Place 12 unit regular hexagons inside a regular hexagon, minimizing outer side length.
    Returns: (inner, center, side, angle)
      inner: list of [x, y, angle_degrees] for each of the 12 inner hexagons
      center: (x, y) of the outer hexagon
      side: side length of the outer hexagon
      angle: rotation angle (degrees) of the outer hexagon
    """
    # Unit hexagon vertices (pointy-top, side=1)
    # Vertices at angles 30, 90, 150, 210, 270, 330 degrees
    hex_verts = []
    for i in range(6):
        a = math.pi/6 + i * math.pi/3
        hex_verts.append((math.cos(a), math.sin(a)))
    hex_verts = np.array(hex_verts)
    
    def support_fn(vertices, normal):
        """Support function: max dot product of vertices with normal."""
        return np.max(vertices @ normal)
    
    def hex_support(angle_deg):
        """Support function of a unit hexagon rotated by angle_deg."""
        # For a regular hexagon, the support function depends only on the angle 
        # relative to the nearest face normal.
        # Face normals are at 0, 60, 120, 180, 240, 300 degrees for flat-top
        # or 30, 90, ... for pointy-top.
        # Let's use a direct computation.
        theta = math.radians(angle_deg)
        R = np.array([[math.cos(theta), -math.sin(theta)],
                      [math.sin(theta), math.cos(theta)]])
        rotated = hex_verts @ R.T
        # The support in direction (1, 0) is max x-coordinate
        # But we need general support. Actually for the container, 
        # we'll compute widths along the 6 container normals.
        return rotated
    
    def width_along_normal(points, normal):
        """Width of point set along a given normal direction."""
        dots = points @ normal
        return np.max(dots) - np.min(dots)
    
    def compute_outer_side(points, inner_angles_deg, container_rot_deg):
        """
        Compute the minimal outer hexagon side length that contains all inner hexagons.
        points: Nx2 array of centers
        inner_angles_deg: array of rotation angles for each inner hex
        container_rot_deg: rotation of the outer hexagon
        Returns: minimal side length, or inf if impossible.
        """
        # Outer hexagon has 6 face normals at angles: container_rot + k*60
        container_normals = []
        base_rot = math.radians(container_rot_deg)
        for k in range(6):
            a = base_rot + k * math.pi/3
            container_normals.append(np.array([math.cos(a), math.sin(a)]))
        
        # For each container normal direction, compute the required half-width
        # The container hexagon with side L has inradius = L * sqrt(3)/2
        # So the support function of the container in any direction is L*sqrt(3)/2
        
        # For each inner hexagon, compute its support in each container normal direction
        # Then the total extent along normal k is:
        # max_i (center_i . n_k + support_i(n_k)) - min_i (center_i . n_k - support_i(n_k))
        # = max_i(center_i . n_k) + max_i(support_i(n_k)) - min_i(center_i . n_k) - max_i(support_i(n_k))
        # Wait, that's not right. Let me think again.
        
        # The container must contain all inner hexagons.
        # For direction n, the container's support is L*sqrt(3)/2.
        # The inner hexagons' combined support in direction n is:
        # max over all vertices of all inner hexagons of (vertex . n)
        # = max_i (max over vertices of hex_i of (vertex . n))
        # = max_i (center_i . n + support_hex_i(n))
        
        # Similarly in direction -n:
        # min_i (center_i . n - support_hex_i(-n)) = min_i (center_i . n - support_hex_i(n))
        # (since hex is centrally symmetric, support in -n = support in n)
        
        # So the total width required in direction n is:
        # max_i (center_i . n + s_i(n)) - min_i (center_i . n - s_i(n))
        # where s_i(n) is the support of hex i in direction n.
        
        # This width must be <= container width in direction n = L*sqrt(3)
        # (since inradius = L*sqrt(3)/2, width = 2*inradius = L*sqrt(3))
        
        max_L = 0.0
        for n in container_normals:
            # Compute support of each inner hex in direction n
            supports = []
            for i, angle in enumerate(inner_angles_deg):
                theta = math.radians(angle)
                R = np.array([[math.cos(theta), -math.sin(theta)],
                              [math.sin(theta), math.cos(theta)]])
                rotated = hex_verts @ R.T
                s = np.max(rotated @ n)
                supports.append(s)
            supports = np.array(supports)
            
            centers_dot = points @ n
            upper = np.max(centers_dot + supports)
            lower = np.min(centers_dot - supports)
            required_width = upper - lower
            
            L_required = required_width / math.sqrt(3)
            max_L = max(max_L, L_required)
        
        return max_L
    
    def is_valid_packing(points, inner_angles_deg, container_side):
        """Check if the packing is valid: no overlaps between inner hexagons, 
        and all fit within the container."""
        # Check pairwise non-overlap
        # Two hexagons overlap if the distance between centers is less than 
        # the sum of their "inradii" in the direction of the center-to-center vector?
        # No, for convex shapes, they don't overlap iff for some separating direction,
        # the projection intervals are disjoint.
        # Simpler: check if any vertex of one is inside the other, or edges intersect.
        # For speed, use a conservative check: if center distance < 2*inradius, might overlap.
        # Inradius of unit hex = sqrt(3)/2.
        # But this is only sufficient, not necessary.
        
        # Better: for each pair, check if they are separated.
        # For two regular hexagons, they are disjoint if and only if there exists 
        # a separating line. We can check the 6 face normals of each hex.
        
        # Actually, let's just check vertex-in-polygon and edge-edge intersection.
        # This is O(n^2 * 36) which is fine for n=12.
        
        def get_hex_vertices(center, angle_deg):
            theta = math.radians(angle_deg)
            R = np.array([[math.cos(theta), -math.sin(theta)],
                          [math.sin(theta), math.cos(theta)]])
            return center + hex_verts @ R.T
        
        def point_in_hex(p, center, angle_deg):
            """Check if point p is inside hexagon at center with angle_deg."""
            theta = math.radians(angle_deg)
            R_inv = np.array([[math.cos(-theta), -math.sin(-theta)],
                              [math.sin(-theta), math.cos(-theta)]])
            local_p = (p - center) @ R_inv
            # Check if local_p is inside unit hex
            # Unit hex vertices at 30, 90, ... degrees
            # Faces: x <= 0.5, -x+sqrt(3)*y <= sqrt(3)/2, etc.
            # Actually, the unit hex with vertices at (cos(30+60k), sin(30+60k)) 
            # has the property that |x| <= 0.5 and |y| <= sqrt(3)/2 is not quite right.
            # Let's use the support function approach: p is inside iff 
            # for all 6 face normals n_k, p . n_k <= h(n_k)
            # Face normals are at 0, 60, ... for flat-top, or 30, 90, ... for pointy-top.
            # Our hex_verts are pointy-top (vertices at 30, 90, ...).
            # So face normals are at 0, 60, 120, ...
            for k in range(6):
                a = k * math.pi/3
                n = np.array([math.cos(a), math.sin(a)])
                h = np.max(hex_verts @ n)  # Should be 0.5
                if local_p @ n > h + 1e-9:
                    return False
            return True
        
        def segments_intersect(p1, p2, p3, p4):
            """Check if segment p1-p2 intersects segment p3-p4."""
            def cross(o, a, b):
                return (a[0]-o[0])*(b[1]-o[1]) - (a[1]-o[1])*(b[0]-o[0])
            d1 = cross(p3, p4, p1)
            d2 = cross(p3, p4, p2)
            d3 = cross(p1, p2, p3)
            d4 = cross(p1, p2, p4)
            if ((d1 > 0 and d2 < 0) or (d1 < 0 and d2 > 0)) and \
               ((d3 > 0 and d4 < 0) or (d3 < 0 and d4 > 0)):
                return True
            return False
        
        verts_list = []
        for i in range(N):
            v = get_hex_vertices(points[i], inner_angles_deg[i])
            verts_list.append(v)
        
        for i in range(N):
            for j in range(i+1, N):
                # Check if any vertex of i is inside j, or vice versa
                for v in verts_list[i]:
                    if point_in_hex(v, points[j], inner_angles_deg[j]):
                        return False
                for v in verts_list[j]:
                    if point_in_hex(v, points[i], inner_angles_deg[i]):
                        return False
                # Check edge-edge intersections
                for e in range(6):
                    p1 = verts_list[i][e]
                    p2 = verts_list[i][(e+1)%6]
                    for f in range(6):
                        p3 = verts_list[j][f]
                        p4 = verts_list[j][(f+1)%6]
                        if segments_intersect(p1, p2, p3, p4):
                            return False
        
        # Check container containment
        # The container center should be the centroid of the inner hexagons
        # for optimal packing. Let's assume container center is at the centroid.
        cx = np.mean(points[:, 0])
        cy = np.mean(points[:, 1])
        # Shift points relative to container center
        centered = points - np.array([cx, cy])
        
        # Check if all hexagons fit in container of side L
        # We'll check for a few container rotations
        best_valid = False
        for rot in [0, 30]:
            L = compute_outer_side(centered, inner_angles_deg, rot)
            if L <= container_side + 1e-6:
                # Verify containment more carefully
                # Check each vertex of each inner hex is inside the container
                container_rot = math.radians(rot)
                # Container vertices
                c_verts = []
                for k in range(6):
                    a = container_rot + k * math.pi/3 + math.pi/6  # offset for vertices
                    r = container_side  # circumradius = side for regular hex
                    c_verts.append(np.array([r*math.cos(a), r*math.sin(a)]))
                c_verts = np.array(c_verts)
                
                for i in range(N):
                    v = get_hex_vertices(points[i] - np.array([cx, cy]), inner_angles_deg[i])
                    for pv in v:
                        # Check if pv is inside container
                        inside = True
                        for k in range(6):
                            edge1 = c_verts[k]
                            edge2 = c_verts[(k+1)%6]
                            # Check if pv is on the correct side of each edge
                            e = edge2 - edge1
                            n = np.array([-e[1], e[0]])  # inward normal
                            if n[0]*edge1[0] + n[1]*edge1[1] > 0:
                                n = -n  # ensure n points inward
                            if (pv - edge1) @ n < -1e-9:
                                inside = False
                                break
                        if not inside:
                            break
                    if not inside:
                        break
                if inside:
                    best_valid = True
                    break
        
        return best_valid
    
    # Initial configurations
    def hex_lattice_config():
        """Hexagonal lattice: center + ring 1 (6) + 5 from ring 2."""
        pts = [[0.0, 0.0]]
        # Ring 1
        for k in range(6):
            a = k * math.pi/3
            pts.append([math.sqrt(3)*math.cos(a), math.sqrt(3)*math.sin(a)])
        # Ring 2: pick 5 contiguous
        ring2 = []
        for k in range(12):
            a = k * math.pi/6 + math.pi/12  # offset
            # Actually, ring 2 points are at distance 2*sqrt(3) from center
            # Angles: 0, 30, 60, ... but offset
            r = 2 * math.sqrt(3)
            a = k * math.pi/6
            ring2.append([r*math.cos(a), r*math.sin(a)])
        # Pick 5 contiguous starting from angle 0
        for k in range(5):
            pts.append(ring2[k])
        return np.array(pts)
    
    def grid_config():
        """3x4 grid-like arrangement."""
        dx = math.sqrt(3)
        dy = 1.5
        pts = []
        for row in range(3):
            for col in range(4):
                x = col * dx
                y = row * dy
                if row % 2 == 1:
                    x += dx/2
                pts.append([x, y])
        return np.array(pts[:12])
    
    def staggered_config():
        """Staggered rows."""
        dx = math.sqrt(3)
        dy = 1.5
        pts = []
        # 4 rows of 3
        for row in range(4):
            for col in range(3):
                x = col * dx
                y = row * dy
                if row % 2 == 1:
                    x += dx/2
                pts.append([x, y])
        return np.array(pts[:12])
    
    configs = [hex_lattice_config(), grid_config(), staggered_config()]
    
    best_L = float('inf')
    best_points = None
    best_angles = None
    best_rot = 0
    
    # For each config, try to optimize
    for cfg_idx, points in enumerate(configs):
        if time.time() - START_TIME > TIME_LIMIT - 10:
            break
        
        # Normalize: center at centroid
        cx = np.mean(points[:, 0])
        cy = np.mean(points[:, 1])
        centered = points - np.array([cx, cy])
        
        # Try different rotations for inner hexagons
        # For hex lattice, angle 0 (pointy-top) or 30 (flat-top)
        for inner_angle in [0, 30]:
            angles = np.full(N, inner_angle, dtype=float)
            
            # Try different container rotations
            for cont_rot in [0, 30]:
                L = compute_outer_side(centered, angles, cont_rot)
                
                # Check validity
                if L < best_L:
                    if is_valid_packing(centered, angles, L):
                        best_L = L
                        best_points = centered.copy()
                        best_angles = angles.copy()
                        best_rot = cont_rot
    
    # Local optimization: perturb the configuration
    if best_points is not None and time.time() - START_TIME < TIME_LIMIT - 20:
        # Try small perturbations
        rng = np.random.RandomState(42)
        for restart in range(5):
            if time.time() - START_TIME > TIME_LIMIT - 15:
                break
            
            # Start from best or random
            if restart == 0:
                pts = best_points.copy()
                angs = best_angles.copy()
            else:
                # Random perturbation
                pts = best_points + rng.normal(0, 0.5, (N, 2))
                angs = best_angles + rng.uniform(-30, 30, N)
                angs = angs % 60  # Hex symmetry
        
        # Greedy improvement: move each hexagon slightly
        for iteration in range(50):
            if time.time() - START_TIME > TIME_LIMIT - 15:
                break
            
            improved = False
            for i in range(N):
                if time.time() - START_TIME > TIME_LIMIT - 15:
                    break
                
                # Try small moves
                for dx in [-0.3, -0.1, 0, 0.1, 0.3]:
                    for dy in [-0.3, -0.1, 0, 0.1, 0.3]:
                        new_pts = pts.copy()
                        new_pts[i] = new_pts[i] + np.array([dx, dy])
                        
                        for cont_rot in [best_rot, (best_rot+30)%60]:
                            L = compute_outer_side(new_pts, angs, cont_rot)
                            if L < best_L:
                                if is_valid_packing(new_pts, angs, L):
                                    best_L = L
                                    best_points = new_pts.copy()
                                    best_angles = angs.copy()
                                    best_rot = cont_rot
                                    pts = new_pts.copy()
                                    improved = True
                                    break
                    if improved:
                        break
    
    # Final result
    if best_points is None:
        # Fallback: use the hex lattice with best estimate
        points = hex_lattice_config()
        cx = np.mean(points[:, 0])
        cy = np.mean(points[:, 1])
        centered = points - np.array([cx, cy])
        angles = np.zeros(N)
        L = compute_outer_side(centered, angles, 0)
        inner = [[float(centered[i, 0]), float(centered[i, 1]), 0.0] for i in range(N)]
        return inner, (0.0, 0.0), float(L), 0.0
    
    inner = [[float(best_points[i, 0]), float(best_points[i, 1]), float(best_angles[i])] for i in range(N)]
    return inner, (0.0, 0.0), float(best_L), float(best_rot)
