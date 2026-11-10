# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


def hexagon_packing_11():
    """
    Constructs a packing of 11 disjoint unit regular hexagons inside a larger regular hexagon, maximizing 1/outer_hex_side_length.
    Returns
        inner_hex_data: np.ndarray of shape (11,3), where each row is of the form (x, y, angle_degrees) containing the (x,y) coordinates and angle_degree of the respective inner hexagon.
        outer_hex_data: np.ndarray of shape (3,) of form (x,y,angle_degree) containing the (x,y) coordinates and angle_degree of the outer hexagon.
        outer_hex_side_length: float representing the side length of the outer hexagon.
    """
    start_time = time.time()
    deadline = start_time + 600.0  # 600s safety margin

    def hex_vertices(cx, cy, s, a_deg):
        a_rad = np.deg2rad(a_deg)
        angles = a_rad + np.deg2rad(np.array([0, 60, 120, 180, 240, 300]))
        return np.column_stack([cx + s * np.cos(angles), cy + s * np.sin(angles)])

    def outer_hex_vertices(cx, cy, s, a_deg):
        return hex_vertices(cx, cy, s, a_deg)

    def check_validity(inner, outer, side):
        """Check if the packing is valid."""
        # All values finite
        if not np.all(np.isfinite(inner)) or not np.all(np.isfinite(outer)):
            return False
        
        # side >= sqrt(11)
        if side < np.sqrt(11) - 1e-9:
            return False
        
        n_inner = len(inner)
        n_outer = 3
        
        # Get outer center
        ocx, ocy, oangle = outer[0], outer[1], outer[2]
        
        # Outer hexagon vertices
        outer_verts = outer_hex_vertices(ocx, ocy, side, oangle)
        
        # Check edge lengths of inner hexagons (should be ~1)
        for i in range(n_inner):
            ix, iy, iangle = inner[i]
            iverts = hex_vertices(ix, iy, 1.0, iangle)
            for j in range(6):
                p1 = iverts[j]
                p2 = iverts[(j + 1) % 6]
                edge_len = np.linalg.norm(p2 - p1)
                if abs(edge_len - 1.0) > 1e-8:
                    return False
        
        # Check separation between every pair of inner hexagons
        for i in range(n_inner):
            for j in range(i + 1, n_inner):
                ix1, iy1, ia1 = inner[i]
                ix2, iy2, ia2 = inner[j]
                
                v1 = hex_vertices(ix1, iy1, 1.0, ia1)
                v2 = hex_vertices(ix2, iy2, 1.0, ia2)
                
                # For each of the 12 edge normals (6 from each hex), check separation
                separated = False
                for hex_v in [v1, v2]:
                    for k in range(6):
                        p1 = hex_v[k]
                        p2 = hex_v[(k + 1) % 6]
                        edge = p2 - p1
                        # Normal (perpendicular to edge)
                        normal = np.array([-edge[1], edge[0]])
                        normal = normal / np.linalg.norm(normal)
                        
                        # Project both hexagons onto this normal
                        proj1 = np.dot(v1, normal)
                        proj2 = np.dot(v2, normal)
                        
                        min1 = np.min(proj1)
                        max1 = np.max(proj1)
                        min2 = np.min(proj2)
                        max2 = np.max(proj2)
                        
                        # Check if intervals are apart by more than 1e-6
                        # Either max1 < min2 - 1e-6 or max2 < min1 - 1e-6
                        if max1 < min2 - 1e-6 or max2 < min1 - 1e-6:
                            separated = True
                            break
                    if separated:
                        break
                if not separated:
                    return False
        
        # Check that all inner vertices are inside the outer hexagon
        for i in range(n_inner):
            ix, iy, iangle = inner[i]
            iverts = hex_vertices(ix, iy, 1.0, iangle)
            for p in iverts:
                # Shift by outer center
                p_shifted = p - np.array([ocx, ocy])
                # For each outer edge from vertex p1 to p2, check cross product
                for k in range(6):
                    p1 = outer_verts[k] - np.array([ocx, ocy])
                    p2 = outer_verts[(k + 1) % 6] - np.array([ocx, ocy])
                    cross = (p2[0] - p1[0]) * (p_shifted[1] - p1[1]) - (p2[1] - p1[1]) * (p_shifted[0] - p1[0])
                    if cross < -1e-6:
                        return False
        
        return True

    def compute_side_length(inner, outer):
        """Compute the minimum side length of outer hexagon that contains all inner hexagons."""
        ocx, ocy, oangle = outer[0], outer[1], outer[2]
        
        # For a regular hexagon centered at (ocx, ocy) with angle oangle,
        # the distance from center to each side (apothem) is s * cos(30 deg) = s * sqrt(3)/2
        # The distance from center to each vertex is s
        
        # For each inner hexagon vertex, compute the required side length
        max_required = 0.0
        for i in range(len(inner)):
            ix, iy, iangle = inner[i]
            iverts = hex_vertices(ix, iy, 1.0, iangle)
            for p in iverts:
                # Distance from outer center to this point
                d = np.linalg.norm(p - np.array([ocx, ocy]))
                # The point must be inside the hexagon.
                # For a regular hexagon with side s, the maximum distance from center to any point inside is s (at vertices)
                # But we need to account for the angular position.
                # Actually, for a regular hexagon, the boundary in polar coordinates is:
                # r_max(theta) = s * cos(30 deg) / cos(theta - nearest_vertex_angle)
                # where nearest_vertex_angle is the angle of the closest vertex.
                
                # Simpler approach: for each outer edge, compute the distance from the point to the edge
                # and ensure it's on the correct side.
                outer_verts = outer_hex_vertices(ocx, ocy, 1.0, oangle)  # unit hexagon
                for k in range(6):
                    p1 = outer_verts[k]
                    p2 = outer_verts[(k + 1) % 6]
                    edge = p2 - p1
                    normal = np.array([-edge[1], edge[0]])
                    normal = normal / np.linalg.norm(normal)
                    # Point relative to edge
                    rel = (p - np.array([ocx, ocy])) - p1
                    dist = np.dot(rel, normal)
                    # The apothem of a unit hexagon is sqrt(3)/2
                    apothem = np.sqrt(3) / 2
                    # Required scale factor
                    if dist > 0:
                        required_s = dist / apothem
                    else:
                        required_s = 0.0
                    max_required = max(max_required, required_s)
        
        return max_required

    # Initial arrangement: hexagonal grid pattern
    # 11 hexagons in a pattern: center + 6 around + 4 more
    # Let's use a known good starting configuration
    
    # Hexagonal lattice: positions for 11 hexagons
    # Center at (0,0), then 6 neighbors, then 4 more at distance 2
    # In hexagonal coordinates, unit hexagons have centers at:
    # Using axial coordinates converted to Cartesian
    
    def hex_to_cart(q, r, size=1.0):
        x = size * (q + r / 2.0) * 2.0  # Actually, for unit hexagons, spacing is 2*sqrt(3)/2 = sqrt(3) for flat-top
        # For pointy-top hexagons with side 1:
        # x = sqrt(3) * (q + r/2)
        # y = 3/2 * r
        x = np.sqrt(3) * (q + r / 2.0)
        y = 1.5 * r
        return x, y
    
    # 11 hexagons in a compact arrangement
    # Using hex coordinates (q, r):
    # (0,0), (1,0), (-1,0), (0,1), (0,-1), (1,-1), (-1,1),
    # (2,-1), (-2,1), (2,-2), (-2,2)
    hex_coords = [
        (0, 0),
        (1, 0), (-1, 0),
        (0, 1), (0, -1),
        (1, -1), (-1, 1),
        (2, -1), (-2, 1),
        (2, -2), (-2, 2)
    ]
    
    inner_init = np.zeros((11, 3))
    for i, (q, r) in enumerate(hex_coords):
        x, y = hex_to_cart(q, r)
        inner_init[i] = [x, y, 0]
    
    outer_init = np.array([0.0, 0.0, 0.0])
    
    # Compute initial side length
    side_init = compute_side_length(inner_init, outer_init)
    
    # Optimize positions to minimize the required side length
    def objective(params):
        """Objective function to minimize: the required outer side length."""
        if time.time() > deadline - 60:
            return np.inf
        
        # Extract positions
        inner_pos = params[:22].reshape(11, 2)
        angles = params[22:33] * 180.0 / np.pi  # Convert to degrees
        ocx, ocy = params[33], params[34]
        oangle = params[35] * 180.0 / np.pi
        
        inner = np.column_stack([inner_pos, angles])
        outer = np.array([ocx, ocy, oangle])
        
        side = compute_side_length(inner, outer)
        return side
    
    # Initial parameters
    x0 = np.concatenate([
        inner_init[:, :2].flatten(),
        np.deg2rad(inner_init[:, 2]),
        [0.0, 0.0, 0.0]
    ])
    
    # Try optimization
    best_inner = inner_init.copy()
    best_outer = outer_init.copy()
    best_side = side_init
    best_score = 1.0 / best_side if best_side > 0 else 0.0
    
    # Use Nelder-Mead for optimization (no gradient needed)
    try:
        result = minimize(objective, x0, method='Nelder-Mead',
                         options={'maxiter': 500, 'xatol': 1e-6, 'fatol': 1e-8,
                                  'adaptive': True})
        
        if result.fun < best_side and np.isfinite(result.fun):
            inner_pos = result.x[:22].reshape(11, 2)
            angles = result.x[22:33] * 180.0 / np.pi
            ocx, ocy = result.x[33], result.x[34]
            oangle = result.x[35] * 180.0 / np.pi
            
            candidate_inner = np.column_stack([inner_pos, angles])
            candidate_outer = np.array([ocx, ocy, oangle])
            candidate_side = result.fun
            
            if check_validity(candidate_inner, candidate_outer, candidate_side):
                best_inner = candidate_inner
                best_outer = candidate_outer
                best_side = candidate_side
                best_score = 1.0 / best_side
    except Exception as e:
        pass
    
    # Try a few random restarts if time permits
    if time.time() < deadline - 120:
        np.random.seed(42)
        for restart in range(5):
            if time.time() > deadline - 60:
                break
            
            # Perturb initial positions
            perturbation = 0.5
            x0_perturbed = x0.copy()
            x0_perturbed[:22] += np.random.randn(22) * perturbation
            
            try:
                result = minimize(objective, x0_perturbed, method='Nelder-Mead',
                                 options={'maxiter': 300, 'xatol': 1e-6, 'fatol': 1e-8,
                                          'adaptive': True})
                
                if result.fun < best_side and np.isfinite(result.fun):
                    inner_pos = result.x[:22].reshape(11, 2)
                    angles = result.x[22:33] * 180.0 / np.pi
                    ocx, ocy = result.x[33], result.x[34]
                    oangle = result.x[35] * 180.0 / np.pi
                    
                    candidate_inner = np.column_stack([inner_pos, angles])
                    candidate_outer = np.array([ocx, ocy, oangle])
                    candidate_side = result.fun
                    
                    if check_validity(candidate_inner, candidate_outer, candidate_side):
                        best_inner = candidate_inner
                        best_outer = candidate_outer
                        best_side = candidate_side
                        best_score = 1.0 / best_side
            except Exception as e:
                pass
    
    # Ensure the result is valid; if not, fall back to a known valid configuration
    if not check_validity(best_inner, best_outer, best_side):
        # Use the initial configuration with a large enough side
        best_inner = inner_init.copy()
        best_outer = outer_init.copy()
        best_side = compute_side_length(best_inner, best_outer)
        if not check_validity(best_inner, best_outer, best_side):
            # Last resort: very large side
            best_side = 15.0
    
    return best_inner, best_outer, float(best_side)


# EVOLVE-BLOCK-END
