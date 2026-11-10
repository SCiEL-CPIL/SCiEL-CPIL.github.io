# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize


def hexagon_packing_12():
    """
    Constructs a packing of 12 disjoint unit regular hexagons inside a larger regular hexagon, maximizing 1/outer_hex_side_length.
    Returns
        inner_hex_data: np.ndarray of shape (12,3), where each row is of the form (x, y, angle_degrees)
        outer_hex_data: np.ndarray of shape (3,) of form (x,y,angle_degree)
        outer_hex_side_length: float
    """
    n = 12
    
    # Initial configuration: hexagonal arrangement
    # Unit hexagons have "radius" (center to vertex) = 1
    # Distance between centers of touching hexagons = 2 (for same orientation)
    
    # Start with a hexagonal grid pattern
    # Positions in a hex lattice with spacing 2
    sqrt3 = np.sqrt(3)
    
    # Hexagonal arrangement: center + 6 around + 5 more
    # Let's use a compact arrangement
    initial_positions = np.array([
        [0, 0],           # 0: center
        [2, 0],           # 1: right
        [-2, 0],          # 2: left
        [1, sqrt3],       # 3: top-right
        [-1, sqrt3],      # 4: top-left
        [1, -sqrt3],      # 5: bottom-right
        [-1, -sqrt3],     # 6: bottom-left
        [3, sqrt3],       # 7: far top-right
        [-3, sqrt3],      # 8: far top-left
        [3, -sqrt3],      # 9: far bottom-right
        [-3, -sqrt3],     # 10: far bottom-left
        [0, 2*sqrt3],     # 11: top
    ])
    
    initial_angles = np.zeros(12)
    initial_outer_center = np.array([0.0, 0.0])
    initial_outer_angle = 0.0
    initial_s = 6.0  # start with a reasonable size
    
    # Pack all parameters into a single vector
    # [x0, y0, a0, x1, y1, a1, ..., x11, y11, a11, outer_x, outer_y, outer_angle, s]
    x0 = np.concatenate([initial_positions, initial_angles, 
                         initial_outer_center, [initial_outer_angle], [initial_s]])
    
    def get_hex_vertices(cx, cy, side, angle_deg):
        """Get vertices of a regular hexagon"""
        angle_rad = np.radians(angle_deg)
        vertices = []
        for i in range(6):
            theta = angle_rad + np.radians(60 * i)
            x = cx + side * np.cos(theta)
            y = cy + side * np.sin(theta)
            vertices.append([x, y])
        return np.array(vertices)
    
    def get_hex_edges(vertices):
        """Get edges of a hexagon as (p1, p2) pairs"""
        edges = []
        for i in range(6):
            p1 = vertices[i]
            p2 = vertices[(i + 1) % 6]
            edges.append((p1, p2))
        return edges
    
    def edge_normal(p1, p2):
        """Get outward normal of edge from p1 to p2"""
        edge = p2 - p1
        length = np.linalg.norm(edge)
        if length < 1e-12:
            return np.array([0.0, 0.0])
        # Normal perpendicular to edge
        normal = np.array([-edge[1], edge[0]]) / length
        return normal
    
    def project_vertex_on_normal(v, normal):
        """Project vertex v onto normal direction"""
        return np.dot(v, normal)
    
    def check_separation(hex1_vertices, hex2_vertices, tol=1e-6):
        """
        Check if two hexagons are separated using SAT.
        Returns the minimum gap (positive if separated, negative if overlapping)
        """
        min_gap = np.inf
        
        all_edges = []
        all_edges.extend(get_hex_edges(hex1_vertices))
        all_edges.extend(get_hex_edges(hex2_vertices))
        
        for p1, p2 in all_edges:
            normal = edge_normal(p1, p2)
            
            # Project all vertices of hex1
            proj1 = [project_vertex_on_normal(v, normal) for v in hex1_vertices]
            min1, max1 = min(proj1), max(proj1)
            
            # Project all vertices of hex2
            proj2 = [project_vertex_on_normal(v, normal) for v in hex2_vertices]
            min2, max2 = min(proj2), max(proj2)
            
            # Check overlap
            if max1 < min2 or max2 < min1:
                # Separated along this axis
                gap = min(min2 - max1, min1 - max2)
                min_gap = min(min_gap, gap)
            else:
                # Overlapping on this axis
                min_gap = min(min_gap, 0.0)
        
        return min_gap
    
    def objective(params):
        """
        Objective function: minimize outer hexagon side length s,
        with penalties for constraint violations
        """
        # Unpack parameters
        inner_params = params[:36].reshape(12, 3)
        outer_x = params[36]
        outer_y = params[37]
        outer_angle = params[38]
        s = params[39]
        
        # Ensure s is positive
        s = max(s, 1.0)
        
        penalty = 0.0
        
        # Get outer hexagon vertices
        outer_vertices = get_hex_vertices(outer_x, outer_y, s, outer_angle)
        
        # Check outer vertices are distinct
        for i in range(6):
            for j in range(i+1, 6):
                if np.linalg.norm(outer_vertices[i] - outer_vertices[j]) < 1e-6:
                    penalty += 1000.0
        
        # Get inner hexagon vertices
        inner_vertices_list = []
        for i in range(12):
            cx, cy, angle = inner_params[i]
            vertices = get_hex_vertices(cx, cy, 1.0, angle)
            inner_vertices_list.append(vertices)
        
        # Check that inner vertices are inside outer hexagon
        for i in range(12):
            vertices = inner_vertices_list[i]
            for v in vertices:
                # Check if v is inside outer hexagon
                # For each edge of outer hexagon, check cross product
                inside = True
                for k in range(6):
                    p1 = outer_vertices[k]
                    p2 = outer_vertices[(k + 1) % 6]
                    cross = (p2[0] - p1[0]) * (v[1] - p1[1]) - (p2[1] - p1[1]) * (v[0] - p1[0])
                    # For a convex polygon with vertices in CCW order, 
                    # point is inside if all cross products have same sign
                    if cross < -1e-6:
                        inside = False
                        break
                if not inside:
                    # Penalty for being outside
                    # Find distance to nearest edge
                    min_dist = np.inf
                    for k in range(6):
                        p1 = outer_vertices[k]
                        p2 = outer_vertices[(k + 1) % 6]
                        edge = p2 - p1
                        edge_len = np.linalg.norm(edge)
                        if edge_len > 1e-12:
                            # Distance from point to line
                            t = np.dot(v - p1, edge) / (edge_len * edge_len)
                            t = np.clip(t, 0, 1)
                            closest = p1 + t * edge
                            dist = np.linalg.norm(v - closest)
                            min_dist = min(min_dist, dist)
                    penalty += min_dist * 10.0
        
        # Check separation between all pairs of inner hexagons
        for i in range(12):
            for j in range(i+1, 12):
                gap = check_separation(inner_vertices_list[i], inner_vertices_list[j])
                if gap < 1e-6:
                    # Penalty for not being separated
                    penalty += (1e-6 - gap) * 100.0
        
        # Check that unit edges have correct length (should be 1)
        for i in range(12):
            vertices = inner_vertices_list[i]
            for k in range(6):
                p1 = vertices[k]
                p2 = vertices[(k + 1) % 6]
                edge_len = np.linalg.norm(p2 - p1)
                if abs(edge_len - 1.0) > 1e-8:
                    penalty += (abs(edge_len - 1.0) - 1e-8) * 1000.0
        
        # Objective: minimize s + penalty
        return s + penalty
    
    # Initial guess
    x0 = np.concatenate([
        initial_positions.flatten(), 
        initial_angles, 
        [initial_outer_center[0], initial_outer_center[1]], 
        [initial_outer_angle], 
        [initial_s]
    ])
    
    # Use Nelder-Mead for robustness
    result = minimize(objective, x0, method='Nelder-Mead', 
                      options={'maxiter': 10000, 'xatol': 1e-8, 'fatol': 1e-10})
    
    # Extract optimal parameters
    params = result.x
    inner_params = params[:36].reshape(12, 3)
    outer_x = params[36]
    outer_y = params[37]
    outer_angle = params[38]
    s = params[39]
    
    # Ensure s is at least the minimum possible
    s = max(s, np.sqrt(12))
    
    # Build output arrays
    inner_hex_data = inner_params.copy()
    outer_hex_data = np.array([outer_x, outer_y, outer_angle])
    
    return inner_hex_data, outer_hex_data, s


# EVOLVE-BLOCK-END
