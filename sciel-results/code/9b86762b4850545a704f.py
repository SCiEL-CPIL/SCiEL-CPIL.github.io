"""Improved program using contact-graph enumeration and LP-based enclosure optimization."""
import numpy as np
from scipy.optimize import linprog
import time

N = 11

# EVOLVE-BLOCK-START
def construct():
    t_start = time.time()
    deadline = t_start + 130.0  # safety margin before 150s kill

    # Precompute hexagon vertices for a unit regular hexagon with side length 1
    # A regular hexagon with side length 1 has vertices at distance 1 from center
    # Vertices at angles 0, 60, 120, 180, 240, 300 degrees
    hex_angles = np.array([0, 60, 120, 180, 240, 300]) * np.pi / 180.0
    base_vertices = np.column_stack([np.cos(hex_angles), np.sin(hex_angles)])
    
    def get_hex_vertices(cx, cy, angle_deg):
        """Get vertices of a unit hexagon centered at (cx,cy) rotated by angle_deg."""
        theta = np.radians(angle_deg)
        R = np.array([[np.cos(theta), -np.sin(theta)],
                      [np.sin(theta), np.cos(theta)]])
        return base_vertices @ R.T + np.array([cx, cy])
    
    def hexagon_contains_point(hex_center, hex_angle, point):
        """Check if point is inside or on the boundary of a hexagon."""
        # Transform point to hexagon's local frame
        theta = np.radians(hex_angle)
        R_inv = np.array([[np.cos(-theta), -np.sin(-theta)],
                          [np.sin(-theta), np.cos(-theta)]])
        local_point = R_inv @ (point - np.array(hex_center))
        # Check if inside unit hexagon (vertices at distance 1)
        # A point is inside a regular hexagon with circumradius 1 if
        # for all 6 edges, the point is on the correct side
        for i in range(6):
            v1 = base_vertices[i]
            v2 = base_vertices[(i+1) % 6]
            edge = v2 - v1
            normal = np.array([-edge[1], edge[0]])
            # The normal should point inward; check orientation
            # For a CCW polygon, inward normal is to the left of edge
            # Cross product of edge and (point - v1) should be >= 0 for CCW
            cross = edge[0] * (local_point[1] - v1[1]) - edge[1] * (local_point[0] - v1[0])
            if cross < -1e-9:
                return False
        return True
    
    def hexagons_disjoint(h1_center, h1_angle, h2_center, h2_angle, tol=1e-7):
        """Check if two unit hexagons are disjoint (no interior overlap)."""
        # Two regular hexagons of side 1 are disjoint if their centers are far enough
        # OR if they only touch at boundaries
        # Simple check: if distance between centers < 2*inradius, they might overlap
        # Inradius of unit hexagon = sqrt(3)/2 ≈ 0.866
        # But this is a necessary condition, not sufficient for all orientations
        
        # More robust: check if any vertex of one is inside the other
        # and if any edge of one intersects any edge of the other
        
        v1 = get_hex_vertices(h1_center[0], h1_center[1], h1_angle)
        v2 = get_hex_vertices(h2_center[0], h2_center[1], h2_angle)
        
        # Check if any vertex of hex1 is strictly inside hex2
        for pt in v1:
            if hexagon_contains_point(h2_center, h2_angle, pt):
                # Check if it's on the boundary
                # If strictly inside, they overlap
                # For boundary, we allow touching
                return False
        
        # Check if any vertex of hex2 is strictly inside hex1
        for pt in v2:
            if hexagon_contains_point(h1_center, h1_angle, pt):
                return False
        
        # Check edge intersections
        for i in range(6):
            e1 = [v1[i], v1[(i+1)%6]]
            for j in range(6):
                e2 = [v2[j], v2[(j+1)%6]]
                if segments_intersect(e1[0], e1[1], e2[0], e2[1]):
                    # They intersect; check if it's just a point touch
                    # For now, consider this as overlap
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
    
    def enclosing_hexagon_side_length(points, outer_angle_deg):
        """Compute the minimum side length of a regular hexagon with given angle 
        that encloses all given points."""
        theta = np.radians(outer_angle_deg)
        R = np.array([[np.cos(theta), -np.sin(theta)],
                      [np.sin(theta), np.cos(theta)]])
        
        # Transform points to outer hexagon's local frame
        local_points = (points - points.mean(axis=0))  # We'll optimize center too
        
        # Actually, we need to find the minimal enclosing hexagon with fixed orientation
        # For a regular hexagon with side length s, the inradius is s*sqrt(3)/2
        # The hexagon in local coordinates has vertices at distance s from center
        # and the edges are at distance s*sqrt(3)/2 from center
        
        # For a point to be inside the hexagon centered at c with side length s and angle theta:
        # In the rotated frame, for each of 6 edge normals n_i, we need n_i . (p - c) <= s*sqrt(3)/2
        
        # This is a linear program in variables (c_x, c_y, s)
        # Minimize s
        # Subject to: for each point p_j and each edge i:
        #   n_i . (p_j - c) <= s * sqrt(3)/2
        #   -n_i . (p_j - c) <= s * sqrt(3)/2
        #   s >= 0
        
        # Variables: c_x, c_y, s (3 variables)
        # Constraints: 2 * N * 6 = 132 inequalities
        
        # n_i are the outward normals of the 6 edges of a unit hexagon
        # Edge i goes from vertex i to vertex i+1
        # Outward normal is perpendicular to edge, pointing outward
        
        edge_normals = []
        for i in range(6):
            v1 = base_vertices[i]
            v2 = base_vertices[(i+1) % 6]
            edge = v2 - v1
            # Outward normal: rotate edge by -90 degrees (clockwise)
            n = np.array([edge[1], -edge[0]])
            n = n / np.linalg.norm(n)
            edge_normals.append(n)
        
        edge_normals = np.array(edge_normals)
        # These normals are in the local frame of the hexagon
        # In the global frame, they are rotated by theta
        global_normals = (edge_normals @ R.T)
        
        # Inradius = s * sqrt(3) / 2
        # Constraint: global_normals[i] . (p_j - c) <= s * sqrt(3)/2
        # And: -global_normals[i] . (p_j - c) <= s * sqrt(3)/2
        
        # Let's set up the LP:
        # Variables: x = [c_x, c_y, s]
        # Minimize: [0, 0, 1] . x
        # Constraints:
        # For each point j and each normal i:
        #   n_i[0]*c_x + n_i[1]*c_y - sqrt(3)/2 * s >= -n_i . p_j  (from n_i.(p-c) <= s*sqrt(3)/2)
        #   -n_i[0]*c_x - n_i[1]*c_y - sqrt(3)/2 * s >= n_i . p_j   (from -n_i.(p-c) <= s*sqrt(3)/2)
        
        # Rewrite as A_ub @ x <= b_ub:
        # n_i[0]*c_x + n_i[1]*c_y - sqrt(3)/2 * s <= n_i . p_j  ... wait
        
        # n_i . (p_j - c) <= s * sqrt(3)/2
        # n_i . p_j - n_i . c <= s * sqrt(3)/2
        # -n_i . c - s * sqrt(3)/2 <= -n_i . p_j
        # (-n_i[0]) * c_x + (-n_i[1]) * c_y + (-sqrt(3)/2) * s <= -n_i . p_j
        
        # Similarly for -n_i:
        # -n_i . (p_j - c) <= s * sqrt(3)/2
        # n_i . c - s * sqrt(3)/2 <= n_i . p_j
        # n_i[0] * c_x + n_i[1] * c_y + (-sqrt(3)/2) * s <= n_i . p_j
        
        n_points = len(points)
        A_ub = []
        b_ub = []
        
        for j in range(n_points):
            p_j = points[j]
            for i in range(6):
                n_i = global_normals[i]
                # First constraint: -n_i . c - s*sqrt(3)/2 <= -n_i . p_j
                A_ub.append([-n_i[0], -n_i[1], -np.sqrt(3)/2])
                b_ub.append(-n_i @ p_j)
                # Second constraint: n_i . c - s*sqrt(3)/2 <= n_i . p_j
                A_ub.append([n_i[0], n_i[1], -np.sqrt(3)/2])
                b_ub.append(n_i @ p_j)
        
        A_ub = np.array(A_ub)
        b_ub = np.array(b_ub)
        
        c_obj = np.array([0, 0, 1])
        
        # Bounds: s >= 0, c_x and c_y unbounded
        bounds = [(-np.inf, np.inf), (-np.inf, np.inf), (0, np.inf)]
        
        result = linprog(c_obj, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
        
        if result.success:
            return result.fun
        else:
            return np.inf
    
    # Generate candidate configurations using contact graph approach
    # For 11 hexagons, try several adjacency patterns
    
    best_side = np.inf
    best_config = None
    
    # Strategy: Try different arrangements of 11 hexagons
    # Common patterns: 
    # 1. A central hexagon with rings around it
    # 2. Linear chains
    # 3. Compact clusters
    
    # Let's try a systematic approach: generate positions by placing hexagons
    # in a hexagonal grid pattern, then optimize the enclosing hexagon
    
    # Hexagonal grid coordinates: use axial coordinates
    # For a hexagon at axial (q, r), the center is at:
    # x = 3/2 * q * s
    # y = sqrt(3)/2 * (r + q/2) * s
    # where s is the side length (1 for unit hexagons)
    
    def axial_to_xy(q, r, side=1.0):
        x = 1.5 * side * q
        y = np.sqrt(3)/2 * side * (r + q/2)
        return x, y
    
    # Generate candidate sets of 11 positions
    candidates = []
    
    # Pattern 1: Central hexagon + 6 in first ring + 4 in second ring
    # Positions in axial coordinates
    central = [(0,0)]
    ring1 = [(1,0), (0,1), (-1,1), (-1,0), (0,-1), (1,-1)]
    # Second ring partial: pick 4 from the 12 possible
    ring2_all = [(2,0), (1,1), (0,2), (-1,2), (-2,2), (-2,1), (-2,0), (-1,-1), (0,-2), (1,-2), (2,-1), (2,1)]
    
    # Try different subsets of ring2
    for i in range(len(ring2_all)):
        for j in range(i+1, len(ring2_all)):
            for k in range(j+1, len(ring2_all)):
                for l in range(k+1, len(ring2_all)):
                    subset = [ring2_all[i], ring2_all[j], ring2_all[k], ring2_all[l]]
                    positions = central + ring1 + subset
                    if len(positions) == 11:
                        candidates.append(positions)
    
    # Pattern 2: Two central hexagons + surrounding
    for dx in [-1, 0, 1]:
        for dy in [-1, 0, 1]:
            if dx == 0 and dy == 0:
                continue
            centers = [(0,0), (dx,dy)]
            # Find all hexagons adjacent to either center
            adj = set()
            for c in centers:
                for dq, dr in [(1,0),(0,1),(-1,1),(-1,0),(0,-1),(1,-1)]:
                    adj.add((c[0]+dq, c[1]+dr))
            adj = list(adj)
            # Need 9 more from adj
            if len(adj) >= 9:
                # Try first 9
                positions = centers + adj[:9]
                if len(positions) == 11:
                    candidates.append(positions)
    
    # Pattern 3: Linear chain
    for start_q in range(-2, 3):
        for start_r in range(-2, 3):
            positions = []
            for k in range(11):
                positions.append((start_q + k, start_r))
            candidates.append(positions)
    
    # Pattern 4: Diagonal chain
    for start_q in range(-2, 3):
        for start_r in range(-2, 3):
            positions = []
            for k in range(11):
                positions.append((start_q + k, start_r + k))
            candidates.append(positions)
    
    # Pattern 5: Compact 3x3-ish block
    for offset_q in range(-2, 3):
        for offset_r in range(-2, 3):
            positions = []
            for q in range(-1, 2):
                for r in range(-1, 2):
                    positions.append((q + offset_q, r + offset_r))
            # That's 9, need 2 more
            positions.append((2 + offset_q, 0 + offset_r))
            positions.append((-2 + offset_q, 0 + offset_r))
            if len(positions) == 11:
                candidates.append(positions)
    
    # Remove duplicates and limit candidates
    seen = set()
    unique_candidates = []
    for cand in candidates:
        key = tuple(sorted(cand))
        if key not in seen:
            seen.add(key)
            unique_candidates.append(cand)
    
    # Limit to reasonable number
    max_candidates = min(len(unique_candidates), 500)
    candidates_to_try = unique_candidates[:max_candidates]
    
    # For each candidate, compute the minimal enclosing hexagon
    for idx, positions in enumerate(candidates_to_try):
        if time.time() > deadline - 5.0:
            break
        
        # Convert axial to Cartesian
        points = np.array([axial_to_xy(q, r) for q, r in positions])
        
        # Try different outer hexagon angles
        best_for_candidate = np.inf
        for outer_angle in [0, 15, 30, 45, 60, 75, 90, 105, 120, 135, 150, 165]:
            side_len = enclosing_hexagon_side_length(points, outer_angle)
            if side_len < best_for_candidate:
                best_for_candidate = side_len
        
        if best_for_candidate < best_side:
            best_side = best_for_candidate
            # Store the configuration
            best_config = (positions, points)
    
    # If no valid configuration found, fall back to a simple arrangement
    if best_config is None:
        # Simple linear arrangement
        positions = [(k, 0) for k in range(11)]
        points = np.array([axial_to_xy(q, r) for q, r in positions])
        best_side = enclosing_hexagon_side_length(points, 0)
        best_config = (positions, points)
    
    # Extract the best configuration
    positions, points = best_config
    
    # Convert to the required format: (x, y, angle_degrees) for each inner hexagon
    inner = []
    for q, r in positions:
        x, y = axial_to_xy(q, r)
        inner.append([x, y, 0.0])
    
    # Find the optimal outer hexagon parameters
    # We need to determine the center and angle of the outer hexagon
    # Re-solve the LP to get the center
    theta = np.radians(0)  # Use the angle that gave the best result
    R = np.array([[np.cos(theta), -np.sin(theta)],
                  [np.sin(theta), np.cos(theta)]])
    
    edge_normals = []
    for i in range(6):
        v1 = base_vertices[i]
        v2 = base_vertices[(i+1) % 6]
        edge = v2 - v1
        n = np.array([edge[1], -edge[0]])
        n = n / np.linalg.norm(n)
        edge_normals.append(n)
    edge_normals = np.array(edge_normals)
    global_normals = (edge_normals @ R.T)
    
    n_points = len(points)
    A_ub = []
    b_ub = []
    
    for j in range(n_points):
        p_j = points[j]
        for i in range(6):
            n_i = global_normals[i]
            A_ub.append([-n_i[0], -n_i[1], -np.sqrt(3)/2])
            b_ub.append(-n_i @ p_j)
            A_ub.append([n_i[0], n_i[1], -np.sqrt(3)/2])
            b_ub.append(n_i @ p_j)
    
    A_ub = np.array(A_ub)
    b_ub = np.array(b_ub)
    c_obj = np.array([0, 0, 1])
    bounds = [(-np.inf, np.inf), (-np.inf, np.inf), (0, np.inf)]
    
    result = linprog(c_obj, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
    
    if result.success:
        outer_cx = result.x[0]
        outer_cy = result.x[1]
        outer_side = result.x[2]
    else:
        # Fallback: use bounding box
        outer_cx = points[:, 0].mean()
        outer_cy = points[:, 1].mean()
        outer_side = best_side
    
    return inner, (outer_cx, outer_cy), outer_side, 0.0

# EVOLVE-BLOCK-END
