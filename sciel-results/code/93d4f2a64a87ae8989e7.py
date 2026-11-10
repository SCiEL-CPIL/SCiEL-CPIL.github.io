import numpy as np
from scipy.optimize import minimize
import time

def hexagon_packing_11():
    start_time = time.time()
    
    # Initial guess: 7-hex flower + 4 more
    # Centers for 7-hex flower (side 1, dist sqrt(3))
    # Center: (0,0)
    # 6 neighbors at distance sqrt(3)
    sqrt3 = np.sqrt(3)
    centers = np.array([
        [0, 0],
        [sqrt3, 0],
        [-sqrt3, 0],
        [0, sqrt3], # Wait, triangular lattice angles are 60 deg.
        # Let's use standard triangular lattice vectors
        # v1 = (sqrt(3), 0)
        # v2 = (sqrt(3)/2, 1.5) ? No.
        # Distance is sqrt(3).
        # Angle 0: (sqrt(3), 0)
        # Angle 60: (sqrt(3)*cos(60), sqrt(3)*sin(60)) = (sqrt(3)/2, 1.5)
        # Angle 120: (-sqrt(3)/2, 1.5)
        # Angle 180: (-sqrt(3), 0)
        # Angle 240: (-sqrt(3)/2, -1.5)
        # Angle 300: (sqrt(3)/2, -1.5)
    ])
    # Correct neighbors for triangular lattice with spacing sqrt(3):
    # (sqrt(3), 0)
    # (sqrt(3)/2, 1.5) -> dist = sqrt(3/4 + 2.25) = sqrt(3)
    # (-sqrt(3)/2, 1.5)
    # (-sqrt(3), 0)
    # (-sqrt(3)/2, -1.5)
    # (sqrt(3)/2, -1.5)
    
    init_centers = np.array([
        [0, 0],
        [sqrt3, 0],
        [sqrt3/2, 1.5],
        [-sqrt3/2, 1.5],
        [-sqrt3, 0],
        [-sqrt3/2, -1.5],
        [sqrt3/2, -1.5],
        # Add 4 more, initially placed far out or in gaps
        [2.5*sqrt3, 0],
        [-2.5*sqrt3, 0],
        [0, 2.5*sqrt3],
        [0, -2.5*sqrt3]
    ])
    
    init_angles = np.zeros(11)
    init_S = 3.0
    
    x0 = np.concatenate([init_centers.flatten(), init_angles, [init_S]])
    
    def get_hex_vertices(cx, cy, angle_deg, s=1.0):
        angle_rad = np.deg2rad(angle_deg)
        verts = []
        for i in range(6):
            theta = angle_rad + np.deg2rad(60 * i)
            x = cx + s * np.cos(theta)
            y = cy + s * np.sin(theta)
            verts.append([x, y])
        return np.array(verts)

    def point_in_polygon(p, poly):
        # p: (2,), poly: (6, 2)
        # Returns True if p is inside poly
        n = len(poly)
        inside = False
        p1x, p1y = poly[0]
        p2x, p2y = poly[1]
        for i in range(n):
            p1x, p1y = poly[i]
            p2x, p2y = poly[(i + 1) % n]
            if ((p1y > p[1]) != (p2y > p[1])) and (p[0] < (p2x - p1x) * (p[1] - p1y) / (p2y - p1y) + p1x):
                inside = not inside
        return inside

    def min_dist_point_to_segment(p, a, b):
        # p, a, b are (2,)
        ab = b - a
        ap = p - a
        t = np.dot(ap, ab) / np.dot(ab, ab)
        t = np.clip(t, 0, 1)
        closest = a + t * ab
        return np.linalg.norm(p - closest)

    def polygon_distance_depth(A, B):
        # Returns 0 if separated, >0 if overlapping
        # Check if any vertex of A is inside B
        max_depth = 0
        for v in A:
            if point_in_polygon(v, B):
                # Calculate distance from v to boundary of B
                min_d = np.inf
                for i in range(6):
                    p1 = B[i]
                    p2 = B[(i+1)%6]
                    d = min_dist_point_to_segment(v, p1, p2)
                    if d < min_d:
                        min_d = d
                # If inside, the "depth" is roughly the distance to the boundary?
                # Actually, if it's inside, we want to push it out.
                # The penalty should be related to how deep it is.
                # Using min distance to boundary is a good proxy.
                max_depth = max(max_depth, min_d)
        
        # Check if any vertex of B is inside A
        for v in B:
            if point_in_polygon(v, A):
                min_d = np.inf
                for i in range(6):
                    p1 = A[i]
                    p2 = A[(i+1)%6]
                    d = min_dist_point_to_segment(v, p1, p2)
                    if d < min_d:
                        min_d = d
                max_depth = max(max_depth, min_d)
                
        # Check edge intersections (approximate depth as 0.1 if they cross)
        # This is complex to compute exactly. We'll rely on vertex checks for now.
        # If edges cross but no vertices are inside, it's a "crossing" which is rare in packing from a valid start.
        
        return max_depth

    def check_containment(inner_verts, outer_verts):
        # Check if all inner vertices are inside outer polygon
        # Returns max violation depth (0 if valid)
        max_violation = 0
        for v in inner_verts:
            if not point_in_polygon(v, outer_verts):
                # Find distance to boundary
                min_d = np.inf
                for i in range(6):
                    p1 = outer_verts[i]
                    p2 = outer_verts[(i+1)%6]
                    d = min_dist_point_to_segment(v, p1, p2)
                    if d < min_d:
                        min_d = d
                max_violation = max(max_violation, min_d)
        return max_violation

    def objective(x):
        if time.time() - start_time > 590:
            return 1e10 # Timeout guard
        
        centers = x[:22].reshape(11, 2)
        angles = x[22:33]
        S = x[33]
        
        # Ensure S is positive
        if S <= 0:
            return 1e10
            
        # Get outer vertices
        outer_verts = get_hex_vertices(0, 0, 0, S)
        
        total_penalty = 0
        valid = True
        
        inner_verts_list = []
        for i in range(11):
            verts = get_hex_vertices(centers[i, 0], centers[i, 1], angles[i])
            inner_verts_list.append(verts)
            
            # Check containment
            viol = check_containment(verts, outer_verts)
            if viol > 1e-6:
                total_penalty += 10 * viol
                valid = False
                
        # Check overlaps
        for i in range(11):
            for j in range(i+1, 11):
                depth = polygon_distance_depth(inner_verts_list[i], inner_verts_list[j])
                if depth > 1e-6:
                    total_penalty += 10 * depth
                    valid = False
                    
        if valid:
            return S
        else:
            return S + total_penalty

    # Run optimization
    # Use Nelder-Mead for derivative-free optimization
    result = minimize(objective, x0, method='Nelder-Mead', options={'maxiter': 2000, 'xatol': 1e-4, 'fatol': 1e-4})
    
    x_opt = result.x
    centers = x_opt[:22].reshape(11, 2)
    angles = x_opt[22:33]
    S = x_opt[33]
    
    # Verify validity
    outer_verts = get_hex_vertices(0, 0, 0, S)
    inner_verts_list = [get_hex_vertices(centers[i, 0], centers[i, 1], angles[i]) for i in range(11)]
    
    is_valid = True
    for i in range(11):
        if check_containment(inner_verts_list[i], outer_verts) > 1e-6:
            is_valid = False
            break
    if is_valid:
        for i in range(11):
            for j in range(i+1, 11):
                if polygon_distance_depth(inner_verts_list[i], inner_verts_list[j]) > 1e-6:
                    is_valid = False
                    break
            if not is_valid:
                break
                
    if not is_valid:
        # Fallback to seed if optimization failed
        return np.array([
            [0, 0, 0],
            [-2.5, 0, 0],
            [2.5, 0, 0],
            [-1.25, 2.17, 0],
            [1.25, 2.17, 0],
            [-1.25, -2.17, 0],
            [1.25, -2.17, 0],
            [-3.75, 2.17, 0],
            [3.75, 2.17, 0],
            [-3.75, -2.17, 0],
            [3.75, -2.17, 0]
        ]), np.array([0, 0, 0]), 8.0

    inner_hex_data = np.column_stack([centers, angles])
    outer_hex_data = np.array([0, 0, 0])
    
    return inner_hex_data, outer_hex_data, S
