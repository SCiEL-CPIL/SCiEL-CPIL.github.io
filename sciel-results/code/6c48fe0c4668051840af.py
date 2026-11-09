# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import time


def hexagon_packing_11():
    """
    Constructs a packing of 11 disjoint unit regular hexagons inside a larger regular hexagon,
    minimizing the outer hexagon side length (maximizing 1/side).
    
    Uses 6-fold symmetric layout: 1 center + 6 ring + 4 outer hexagons.
    Optimizes 3 DOF: ring radius r1, outer radius r2, global rotation angle.
    """
    start_time = time.time()
    deadline = start_time + 580  # safety margin before 630s limit

    def hex_vertices(cx, cy, s, a_deg):
        """Vertices of a regular hexagon with center (cx,cy), side s, angle a_deg."""
        a = np.radians(a_deg)
        angles = a + np.arange(6) * np.pi / 3
        return np.column_stack([cx + s * np.cos(angles), cy + s * np.sin(angles)])

    def hex_edge_normals(vertices):
        """Return 6 edge normals (unit vectors) for a hexagon given its vertices."""
        normals = np.zeros((6, 2))
        for i in range(6):
            v1 = vertices[i]
            v2 = vertices[(i + 1) % 6]
            edge = v2 - v1
            # Normal pointing outward (perpendicular to edge)
            n = np.array([-edge[1], edge[0]])
            n = n / (np.linalg.norm(n) + 1e-15)
            normals[i] = n
        return normals

    def check_separation(hex1_verts, hex2_verts):
        """Check if two hexagons are separated (no intersection).
        Returns True if separated, False if intersecting."""
        n1 = hex_edge_normals(hex1_verts)
        n2 = hex_edge_normals(hex2_verts)
        
        # Check all 12 separating axes
        for n in np.vstack([n1, n2]):
            proj1 = hex1_verts @ n
            proj2 = hex2_verts @ n
            max1 = proj1.max()
            min1 = proj1.min()
            max2 = proj2.max()
            min2 = proj2.min()
            # Intervals must be apart by more than 1e-6
            if max1 + 1e-6 < min2 or max2 + 1e-6 < min1:
                continue
            else:
                return False
        return True

    def check_inside_outer(inner_verts, outer_verts):
        """Check if all vertices of inner hex are inside or on the outer hexagon.
        Uses cross product test for each outer edge."""
        for i in range(6):
            p1 = outer_verts[i]
            p2 = outer_verts[(i + 1) % 6]
            edge = p2 - p1
            for p in inner_verts:
                cross = edge[0] * (p[1] - p1[1]) - edge[1] * (p[0] - p1[0])
                if cross < -1e-6:
                    return False
        return True

    def validate(inner_hex_data, outer_hex_data, side):
        """Full validation as per task contract."""
        if not np.all(np.isfinite(inner_hex_data)) or not np.all(np.isfinite(outer_hex_data)):
            return False
        if side < np.sqrt(11) - 1e-9:
            return False
        
        # Compute all vertices
        inner_verts_list = []
        for i in range(11):
            cx, cy, ang = inner_hex_data[i]
            verts = hex_vertices(cx, cy, 1.0, ang)
            inner_verts_list.append(verts)
        
        outer_cx, outer_cy, outer_ang = outer_hex_data
        outer_verts = hex_vertices(outer_cx, outer_cy, side, outer_ang)
        
        # Check all pairs of inner hexagons are separated
        for i in range(11):
            for j in range(i + 1, 11):
                if not check_separation(inner_verts_list[i], inner_verts_list[j]):
                    return False
        
        # Check all inner hexagons are inside outer
        for i in range(11):
            if not check_inside_outer(inner_verts_list[i], outer_verts):
                return False
        
        # Check edge lengths of shifted inner hexagons
        shifted_center = np.array([outer_cx, outer_cy])
        for i in range(11):
            cx, cy, ang = inner_hex_data[i]
            # Shifted position
            sx = cx - shifted_center[0]
            sy = cy - shifted_center[1]
            verts = hex_vertices(sx, sy, 1.0, ang)
            for k in range(6):
                v1 = verts[k]
                v2 = verts[(k + 1) % 6]
                edge_len = np.linalg.norm(v2 - v1)
                if abs(edge_len - 1.0) > 1e-8:
                    return False
        
        return True

    def compute_side_length(r1, r2, angle_deg):
        """Compute the minimum outer hexagon side length needed to contain all 11 hexagons
        in the 6-fold symmetric layout."""
        # Layout: center at origin, 6 ring hexagons at radius r1, 4 outer hexagons at radius r2
        # The 4 outer hexagons are at angles 0, 90, 180, 270 degrees (or similar)
        # Actually for 6-fold symmetry with 4 outer, let me think...
        
        # 11 = 1 + 6 + 4. The 4 outer ones break 6-fold symmetry.
        # Let's use: 6 ring at angles 0,60,120,180,240,300 at radius r1
        # 4 outer at angles 30, 150, 270, 330 at radius r2 (offset by 30 deg)
        
        inner_positions = []
        # Center
        inner_positions.append((0.0, 0.0, 0.0))
        
        # 6 ring hexagons
        for k in range(6):
            theta = np.radians(60 * k + angle_deg)
            x = r1 * np.cos(theta)
            y = r1 * np.sin(theta)
            inner_positions.append((x, y, 0.0))
        
        # 4 outer hexagons - placed at 30, 150, 270, 330 degrees
        for k in range(4):
            theta = np.radians(90 * k + 30 + angle_deg)
            x = r2 * np.cos(theta)
            y = r2 * np.sin(theta)
            inner_positions.append((x, y, 0.0))
        
        inner_hex_data = np.array(inner_positions)
        
        # Compute required outer side length
        # The outer hexagon is centered at origin with angle 0
        # We need to find minimum side s such that all inner hexagons fit inside
        
        # For a regular hexagon with side s centered at origin with angle 0,
        # the vertices are at distance s from center, at angles 0, 60, 120, 180, 240, 300
        # The inradius (distance from center to edge midpoint) is s * sqrt(3)/2
        
        # To check if a point is inside the outer hexagon, we can use the constraint:
        # For each edge, the point must be on the correct side.
        
        # Let's compute the required side by finding the maximum "support function" value
        # For a hexagon with angle 0 and side s, the 6 edge normals are at angles 30, 90, 150, 210, 270, 330
        # The support in direction theta is s * cos(theta - 30deg) for theta near 30, etc.
        # Actually, the distance from center to each edge along the normal direction is s * sqrt(3)/2
        
        # The constraint for a point p to be inside: for each of 6 edge normals n_i,
        # dot(p, n_i) <= s * sqrt(3)/2
        
        # So s >= max over all points and all edge normals of dot(p, n_i) / (sqrt(3)/2)
        
        # But we need to check all vertices of all inner hexagons, not just centers.
        
        # For simplicity, let's compute the required side by checking vertices
        # The edge normals of the outer hexagon (angle 0) are at angles 30, 90, 150, 210, 270, 330
        edge_normal_angles = np.radians([30, 90, 150, 210, 270, 330])
        edge_normals = np.column_stack([np.cos(edge_normal_angles), np.sin(edge_normal_angles)])
        
        max_dot = 0
        for (cx, cy, ang) in inner_positions:
            verts = hex_vertices(cx, cy, 1.0, ang)
            for n in edge_normals:
                dots = verts @ n
                max_dot = max(max_dot, dots.max())
        
        # s * sqrt(3)/2 >= max_dot, so s >= max_dot / (sqrt(3)/2)
        s = max_dot / (np.sqrt(3) / 2)
        
        return s, inner_hex_data

    def objective(params):
        """Objective function to minimize: the outer hexagon side length."""
        r1, r2, angle_deg = params
        
        # Basic constraints
        if r1 < 1.5 or r2 < 2.5:
            return 1e6
        if r2 < r1:
            return 1e6
        
        s, inner_hex_data = compute_side_length(r1, r2, angle_deg)
        
        # Check if the packing is valid (hexagons don't overlap)
        outer_hex_data = np.array([0.0, 0.0, 0.0])
        if not validate(inner_hex_data, outer_hex_data, s):
            return 1e6
        
        return s

    # Initial guess
    # For unit hexagons, the minimum distance between centers for non-overlapping is 2*sqrt(3) ≈ 3.46
    # Ring radius should be at least 2*sqrt(3)/2 * 2 = 2*sqrt(3) ≈ 3.46 for adjacent ring hexagons
    # Actually, adjacent hexagons in the ring are separated by 60 degrees, so distance = 2*r1*sin(30) = r1
    # So r1 >= 2*sqrt(3) for the ring hexagons to not overlap with each other
    # And r1 >= 2*sqrt(3) for ring to not overlap with center (distance from center to ring = r1, need r1 >= 2*sqrt(3))
    
    # Wait, let me reconsider. Two unit hexagons with centers at distance d apart:
    # They are non-overlapping iff d >= 2*sqrt(3) (the minimum distance between centers of two non-overlapping unit hexagons)
    
    # For the ring: adjacent ring hexagons are at distance 2*r1*sin(30°) = r1 from each other
    # So we need r1 >= 2*sqrt(3) ≈ 3.464
    
    # Ring to center: distance = r1, need r1 >= 2*sqrt(3)
    
    # Outer to ring: distance between outer at radius r2 and ring at radius r1
    # The angle difference is 30 degrees, so distance = sqrt(r1² + r2² - 2*r1*r2*cos(30°))
    # Need this >= 2*sqrt(3)
    
    # Outer to outer: adjacent outer hexagons are 90 degrees apart, distance = 2*r2*sin(45°) = r2*sqrt(2)
    # Need r2*sqrt(2) >= 2*sqrt(3), so r2 >= 2*sqrt(3)/sqrt(2) = 2*sqrt(3/2) ≈ 2.449
    
    # Let's start with r1 = 2*sqrt(3) + 0.1, r2 = r1 + 2*sqrt(3) + 0.1
    
    r1_init = 2 * np.sqrt(3) + 0.5
    r2_init = r1_init + 2 * np.sqrt(3) + 0.5
    angle_init = 0.0
    
    best_s = 1e10
    best_params = np.array([r1_init, r2_init, angle_init])
    best_inner = None
    best_outer = np.array([0.0, 0.0, 0.0])
    
    # Try multiple starting points
    initial_guesses = [
        [r1_init, r2_init, 0.0],
        [r1_init, r2_init, 30.0],
        [r1_init + 1.0, r2_init + 1.0, 0.0],
        [2*np.sqrt(3) + 0.1, 2*np.sqrt(3) + 0.1 + 2*np.sqrt(3) + 0.1, 0.0],
        [2*np.sqrt(3) + 0.1, 2*np.sqrt(3) + 0.1 + 2*np.sqrt(3) + 0.1, 15.0],
    ]
    
    for guess in initial_guesses:
        if time.time() > deadline - 60:
            break
        try:
            result = minimize(objective, guess, method='Nelder-Mead',
                             options={'maxiter': 5000, 'xatol': 1e-8, 'fatol': 1e-10})
            if result.fun < best_s:
                best_s = result.fun
                best_params = result.x
                s, inner_hex_data = compute_side_length(*best_params)
                if validate(inner_hex_data, best_outer, s):
                    best_inner = inner_hex_data
        except Exception:
            continue
    
    # If optimization didn't find a valid solution, use a conservative default
    if best_inner is None or best_s >= 1e9:
        # Fallback: use a known valid configuration
        # Place hexagons in a hexagonal grid pattern
        inner_hex_data = np.array([
            [0, 0, 0],
            [-2*np.sqrt(3), 0, 0],
            [2*np.sqrt(3), 0, 0],
            [-np.sqrt(3), 3, 0],
            [np.sqrt(3), 3, 0],
            [-np.sqrt(3), -3, 0],
            [np.sqrt(3), -3, 0],
            [-3*np.sqrt(3), 3, 0],
            [3*np.sqrt(3), 3, 0],
            [-3*np.sqrt(3), -3, 0],
            [3*np.sqrt(3), -3, 0],
        ])
        outer_hex_data = np.array([0.0, 0.0, 0.0])
        side = 10.0
        # Adjust side to be just large enough
        for _ in range(50):
            if validate(inner_hex_data, outer_hex_data, side):
                break
            side += 0.5
        return inner_hex_data, outer_hex_data, side
    
    # Final validation
    s_final, inner_final = compute_side_length(*best_params)
    if not validate(inner_final, best_outer, s_final):
        # Fallback
        inner_hex_data = np.array([
            [0, 0, 0],
            [-2*np.sqrt(3), 0, 0],
            [2*np.sqrt(3), 0, 0],
            [-np.sqrt(3), 3, 0],
            [np.sqrt(3), 3, 0],
            [-np.sqrt(3), -3, 0],
            [np.sqrt(3), -3, 0],
            [-3*np.sqrt(3), 3, 0],
            [3*np.sqrt(3), 3, 0],
            [-3*np.sqrt(3), -3, 0],
            [3*np.sqrt(3), -3, 0],
        ])
        outer_hex_data = np.array([0.0, 0.0, 0.0])
        side = 10.0
        for _ in range(50):
            if validate(inner_hex_data, outer_hex_data, side):
                break
            side += 0.5
        return inner_hex_data, outer_hex_data, side
    
    return inner_final, best_outer, s_final


# EVOLVE-BLOCK-END
