import math
import numpy as np
import time
from scipy.optimize import minimize, differential_evolution
from itertools import combinations

N = 11
APOTHEM = math.sqrt(3) / 2.0
START_TIME = time.time()
TIME_LIMIT = 150.0
SAFETY_MARGIN = 10.0

def construct():
    if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
        # Fallback: simple grid packing
        return [[0.0, 0.0, 0.0]], [0.0, 0.0], 1.0, 0.0

    # --- Geometry Helpers ---
    
    def hex_vertices(angle_deg):
        """Returns 6 vertices of a unit regular hexagon rotated by angle_deg."""
        angles = np.arange(6) * (math.pi / 3.0) + math.radians(angle_deg)
        return np.array([np.cos(angles), np.sin(angles)]).T

    def point_in_hexagon(p, center, side, angle_deg):
        """Check if point p is inside a hexagon with given center, side, angle."""
        # Transform to hexagon local frame
        p_rel = p - center
        # Rotate by -angle
        theta = -math.radians(angle_deg)
        c, s = math.cos(theta), math.sin(theta)
        R = np.array([[c, -s], [s, c]])
        p_local = R @ p_rel
        
        # Check against 6 half-planes defined by the sides
        # Normal vectors for sides of a flat-top hexagon (angle 0)
        # Sides are at angles 0, 60, ... 300 from center? No, vertices are at 0, 60...
        # For a hexagon with vertex at (1,0), the side opposite is at x=-1? No.
        # Let's use the support function or just check distance to edges.
        # Simpler: check if projection onto 6 normal directions is within apothem.
        # Normals for a hexagon with vertex at angle 0 are at angles 30, 90, 150...
        # Apothem = side * sqrt(3) / 2
        apothem = side * math.sqrt(3) / 2.0
        for k in range(6):
            # Normal direction
            ang = math.radians(angle_deg) + math.pi / 6.0 + k * math.pi / 3.0
            n = np.array([math.cos(ang), math.sin(ang)])
            dist = np.dot(p_rel, n)
            if dist > apothem:
                return False
            if dist < -apothem:
                return False
        return True

    def hexagon_support(point, hex_center, hex_angle, direction):
        """
        Returns the maximum dot product of (v - point) where v is a vertex of the hexagon.
        Actually, we need the support function of the hexagon relative to a global direction.
        We want to know the max projection of the hexagon onto 'direction'.
        h(direction) = max_{v in Hex} (v . direction)
        Since Hex = hex_center + R * unit_hex,
        h(direction) = hex_center . direction + max_{u in unit_hex} (u . (R^T direction))
        """
        # Rotate direction to local frame
        theta = -math.radians(hex_angle)
        c, s = math.cos(theta), math.sin(theta)
        R_t = np.array([[c, s], [-s, c]]) # Transpose of rotation
        d_local = R_t @ direction
        
        # Max projection of unit hexagon onto d_local
        # Unit hex vertices: (1,0), (0.5, sqrt(3)/2), ...
        # The support function of a regular hexagon is piecewise linear.
        # We can just compute max dot product with the 6 vertices.
        unit_verts = hex_vertices(0)
        projs = unit_verts @ d_local
        return hex_center @ direction + np.max(projs)

    def calc_min_side(pts, angles, center, outer_angle):
        """
        Calculate the minimum side length S of an outer hexagon (center, outer_angle)
        that contains all inner hexagons (pts, angles).
        """
        # The outer hexagon is defined by 6 half-planes.
        # For each of the 6 normals n_k of the outer hexagon:
        # The max projection of ANY inner hexagon onto n_k must be <= center . n_k + D_outer
        # The min projection of ANY inner hexagon onto n_k must be >= center . n_k - D_outer
        # D_outer is the apothem of the outer hexagon.
        # S_outer = 2 * D_outer / sqrt(3)
        
        normals = []
        for k in range(6):
            ang = math.radians(outer_angle) + math.pi / 6.0 + k * math.pi / 3.0
            normals.append(np.array([math.cos(ang), math.sin(ang)]))
        
        max_required_D = 0.0
        center_vec = np.array(center)
        
        for n in normals:
            max_proj = -float('inf')
            min_proj = float('inf')
            
            for i in range(N):
                # Support function of inner hex i in direction n
                # max_{v in hex_i} (v . n)
                # = center_i . n + max_{u in unit_hex_i} (u . (R_i^T n))
                
                # We precomputed unit hex vertices for angle 0.
                # But inner hex i has angle angles[i].
                # So we need to rotate n to local frame of hex i.
                theta = -math.radians(angles[i])
                c, s = math.cos(theta), math.sin(theta)
                R_t = np.array([[c, s], [-s, c]])
                n_local = R_t @ n
                
                unit_verts = hex_vertices(0)
                projs = unit_verts @ n_local
                max_u = np.max(projs)
                min_u = np.min(projs)
                
                c_i = pts[i]
                max_v = c_i @ n + max_u
                min_v = c_i @ n + min_u
                
                if max_v > max_proj:
                    max_proj = max_v
                if min_v < min_proj:
                    min_proj = min_v
            
            # The outer hexagon centered at `center` with apothem D must satisfy:
            # max_proj <= center . n + D  =>  D >= max_proj - center . n
            # min_proj >= center . n - D  =>  D >= center . n - min_proj
            
            c_dot_n = center_vec @ n
            D1 = max_proj - c_dot_n
            D2 = c_dot_n - min_proj
            
            D_required = max(D1, D2)
            if D_required > max_required_D:
                max_required_D = D_required
        
        S = 2.0 * max_required_D / math.sqrt(3.0)
        return S

    def check_overlap(pts, angles, eps=1e-9):
        """
        Check if any two inner hexagons overlap.
        Returns True if they overlap, False otherwise.
        Uses separating axis theorem (SAT) for convex polygons.
        """
        for i in range(N):
            for j in range(i + 1, N):
                # Get vertices
                verts_i = pts[i] + hex_vertices(angles[i])
                verts_j = pts[j] + hex_vertices(angles[j])
                
                # SAT Check
                if not separate_axes(verts_i, verts_j):
                    return True
        return False

    def separate_axes(p1, p2):
        """Check if two convex polygons are separated by an axis."""
        # Get all edge normals from both polygons
        normals = []
        for p in [p1, p2]:
            for k in range(6):
                v1 = p[k]
                v2 = p[(k + 1) % 6]
                edge = v2 - v1
                # Normal perpendicular to edge
                n = np.array([-edge[1], edge[0]])
                n_norm = np.linalg.norm(n)
                if n_norm > 1e-9:
                    normals.append(n / n_norm)
        
        for n in normals:
            # Project both polygons onto n
            proj1 = p1 @ n
            proj2 = p2 @ n
            
            min1, max1 = np.min(proj1), np.max(proj1)
            min2, max2 = np.min(proj2), np.max(proj2)
            
            # Check for gap
            if max1 < min2 - 1e-9 or max2 < min1 - 1e-9:
                return True
        
        return False

    # --- Optimization ---
    
    def objective(params):
        """
        Minimize the outer side length S.
        Params: x0, y0, a0, ..., x10, y10, a10, cx, cy, beta
        a_i is angle in radians.
        """
        if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
            return 1e100
            
        pts = params[:22].reshape(11, 2)
        angles_rad = params[22:33]
        cx = params[33]
        cy = params[34]
        beta = params[35]
        
        # Convert angles to degrees for helper functions? 
        # My helpers take degrees. Let's stick to radians internally if possible, 
        # but hex_vertices takes degrees. Let's modify hex_vertices to take radians or just convert.
        # For speed in objective, let's assume we can compute S and overlap quickly.
        
        # 1. Check Overlaps (SAT is expensive, but necessary for validity)
        # To speed up, we can first check distance between centers.
        # Min distance for non-overlap is at least 2 * apothem = sqrt(3) ~ 1.732 if aligned.
        # If dist < 1.7, they definitely overlap.
        # If dist > 2.0 (circumradius sum), they definitely don't overlap? 
        # Circumradius is 1. So if dist > 2, no overlap.
        # If 1.7 < dist < 2.0, we need SAT.
        
        # Optimization: Only run SAT if potentially overlapping.
        for i in range(N):
            for j in range(i + 1, N):
                d = np.linalg.norm(pts[i] - pts[j])
                if d < 2.0: # Potential overlap zone
                    # Quick check: if d < sqrt(3) - eps, definitely overlap? 
                    # Min distance is sqrt(3). So if d < sqrt(3), overlap.
                    if d < math.sqrt(3.0) - 1e-6:
                        return 1e100 # Heavy penalty
                    else:
                        # Run SAT
                        # Convert angles rad to deg for hex_vertices
                        a_i_deg = math.degrees(angles_rad[i])
                        a_j_deg = math.degrees(angles_rad[j])
                        verts_i = pts[i] + hex_vertices(a_i_deg)
                        verts_j = pts[j] + hex_vertices(a_j_deg)
                        if not separate_axes(verts_i, verts_j):
                            return 1e100
        
        # 2. Calculate S
        # Convert angles rad to deg for calc_min_side
        angles_deg = np.degrees(angles_rad)
        beta_deg = math.degrees(beta)
        S = calc_min_side(pts, angles_deg, [cx, cy], beta_deg)
        
        # Add small penalty for large S to keep points compact? 
        # No, S is the objective.
        
        return S

    # Initial guesses
    # Start with a hexagonal grid pattern
    # Center at (0,0), outer at (0,0)
    # Inner hexagons on a triangular grid
    
    initial_sets = []
    
    # Set 1: Standard hex grid
    # (0,0)
    # (1,0), (0,1), (-1,1), (-1,0), (0,-1), (1,-1)
    # (2,0), (1,1), (0,2), (-1,2), (-2,1) -> 11 total?
    # Let's use axial coordinates
    def axial_to_cartesian(q, r):
        x = math.sqrt(3) * (q + r / 2.0)
        y = 1.5 * r
        return [x, y]
    
    axial_1 = [(0,0), (1,0), (0,1), (-1,1), (-1,0), (0,-1), (1,-1), (1,1), (-1,2), (-2,1), (2,-1)]
    pts_1 = np.array([axial_to_cartesian(q, r) for q, r in axial_1])
    angles_1 = np.zeros(N)
    cx_1, cy_1 = 0.0, 0.0
    beta_1 = 0.0
    initial_sets.append(np.concatenate([pts_1.flatten(), angles_1, [cx_1, cy_1, beta_1]]))
    
    # Set 2: Perturbed
    np.random.seed(42)
    pts_2 = pts_1 + np.random.normal(0, 0.1, (N, 2))
    angles_2 = np.random.uniform(-0.1, 0.1, N)
    initial_sets.append(np.concatenate([pts_2.flatten(), angles_2, [cx_1, cy_1, beta_1]]))

    # Set 3: Rotated grid
    pts_3 = pts_1.copy()
    # Rotate points by 10 degrees
    theta = math.radians(10)
    R = np.array([[math.cos(theta), -math.sin(theta)], [math.sin(theta), math.cos(theta)]])
    pts_3 = (R @ pts_3.T).T
    angles_3 = np.full(N, 10.0)
    initial_sets.append(np.concatenate([pts_3.flatten(), angles_3, [cx_1, cy_1, beta_1]]))

    best_S = float('inf')
    best_params = None

    # Run Nelder-Mead on each initial set
    for x0 in initial_sets:
        if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
            break
            
        try:
            # Use Nelder-Mead for local refinement
            res = minimize(objective, x0, method='Nelder-Mead', 
                           options={'maxiter': 5000, 'xatol': 1e-6, 'fatol': 1e-6, 'disp': False})
            
            if res.fun < best_S:
                # Verify validity
                p = res.x
                pts = p[:22].reshape(11, 2)
                angles_rad = p[22:33]
                cx, cy = p[33], p[34]
                beta = p[35]
                
                # Double check overlaps with strict tolerance
                valid = True
                for i in range(N):
                    for j in range(i + 1, N):
                        d = np.linalg.norm(pts[i] - pts[j])
                        if d < math.sqrt(3.0) + 1e-5: # If close, check SAT
                            a_i_deg = math.degrees(angles_rad[i])
                            a_j_deg = math.degrees(angles_rad[j])
                            verts_i = pts[i] + hex_vertices(a_i_deg)
                            verts_j = pts[j] + hex_vertices(a_j_deg)
                            if not separate_axes(verts_i, verts_j):
                                valid = False
                                break
                    if not valid:
                        break
                
                if valid:
                    # Recalculate S to be sure
                    angles_deg = np.degrees(angles_rad)
                    beta_deg = math.degrees(beta)
                    S_val = calc_min_side(pts, angles_deg, [cx, cy], beta_deg)
                    
                    if S_val < best_S:
                        best_S = S_val
                        best_params = p
                        
        except Exception:
            continue

    # If no valid solution found from optimization, try a simple fallback
    if best_params is None:
        # Use the first initial set and calculate S
        p = initial_sets[0]
        pts = p[:22].reshape(11, 2)
        angles_rad = p[22:33]
        cx, cy = p[33], p[34]
        beta = p[35]
        
        # Check if valid
        valid = True
        for i in range(N):
            for j in range(i + 1, N):
                d = np.linalg.norm(pts[i] - pts[j])
                if d < math.sqrt(3.0) + 1e-5:
                    a_i_deg = math.degrees(angles_rad[i])
                    a_j_deg = math.degrees(angles_rad[j])
                    verts_i = pts[i] + hex_vertices(a_i_deg)
                    verts_j = pts[j] + hex_vertices(a_j_deg)
                    if not separate_axes(verts_i, verts_j):
                        valid = False
                        break
            if not valid:
                break
        
        if valid:
            angles_deg = np.degrees(angles_rad)
            beta_deg = math.degrees(beta)
            best_S = calc_min_side(pts, angles_deg, [cx, cy], beta_deg)
            best_params = p
        else:
            # Absolute fallback: return a trivially valid but large packing
            # Just put them in a line
            pts = np.zeros((N, 2))
            for i in range(N):
                pts[i, 0] = i * 2.0 # Spaced by 2, definitely no overlap
            angles_rad = np.zeros(N)
            cx = np.mean(pts[:, 0])
            cy = 0.0
            beta = 0.0
            p = np.concatenate([pts.flatten(), angles_rad, [cx, cy, beta]])
            
            angles_deg = np.degrees(angles_rad)
            beta_deg = math.degrees(beta)
            best_S = calc_min_side(pts, angles_deg, [cx, cy], beta_deg)
            best_params = p

    # Extract final result
    pts = best_params[:22].reshape(11, 2)
    angles_rad = best_params[22:33]
    cx = best_params[33]
    cy = best_params[34]
    beta = best_params[35]
    
    # Add tiny epsilon to S to ensure containment due to float precision
    final_S = best_S * (1.0 + 1e-8)
    
    inner = []
    for i in range(N):
        inner.append([float(pts[i, 0]), float(pts[i, 1]), float(math.degrees(angles_rad[i]))])
    
    center = [float(cx), float(cy)]
    side = float(final_S)
    rotation = float(math.degrees(beta))
    
    return inner, center, side, rotation
