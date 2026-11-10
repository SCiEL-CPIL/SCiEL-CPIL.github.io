# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize


def hexagon_packing_12():
    """
    Constructs a packing of 12 disjoint unit regular hexagons inside a larger regular hexagon,
    using a hexagonal cluster arrangement and optimizing the outer hexagon size.
    Returns
        inner_hex_data: np.ndarray of shape (12,3), where each row is of the form (x, y, angle_degrees)
        outer_hex_data: np.ndarray of shape (3,) of form (x, y, angle_degrees)
        outer_hex_side_length: float
    """
    # Hexagonal cluster of 12 unit hexagons: center + ring of 6 + ring of 6
    # For unit hexagons, center-to-center distance for adjacent hexes is sqrt(3)
    d = np.sqrt(3)  # distance between adjacent hex centers
    
    # Generate hexagonal lattice positions for a cluster of radius 2 (12 hexagons)
    centers = []
    # Center
    centers.append([0.0, 0.0])
    # Ring 1: 6 hexagons at distance d
    for i in range(6):
        angle = np.pi / 3 * i
        centers.append([d * np.cos(angle), d * np.sin(angle)])
    # Ring 2: 6 hexagons at distance 2*d
    for i in range(6):
        angle = np.pi / 3 * i
        centers.append([2 * d * np.cos(angle), 2 * d * np.sin(angle)])
    
    centers = np.array(centers)
    
    # All inner hexagons have angle 0 (aligned with outer)
    inner_hex_data = np.column_stack([centers, np.zeros(12)])
    
    # Optimize the outer hexagon: find minimal side length s such that all inner hexagons fit
    # The outer hexagon is centered at origin with some angle
    
    def get_hex_vertices(cx, cy, side, angle_deg):
        """Get 6 vertices of a regular hexagon."""
        angle_rad = np.radians(angle_deg)
        vertices = []
        for i in range(6):
            a = angle_rad + np.pi / 3 * i
            vertices.append([cx + side * np.cos(a), cy + side * np.sin(a)])
        return np.array(vertices)
    
    def check_fit(outer_side, outer_angle_deg):
        """Check if all inner hexagons fit inside the outer hexagon.
        Returns the minimum required outer side length given the orientation."""
        outer_verts = get_hex_vertices(0, 0, outer_side, outer_angle_deg)
        
        # For each inner hexagon vertex, check if it's inside the outer hexagon
        # A point is inside a convex polygon if it's on the same side of all edges
        max_required_side = 0.0
        
        for i in range(12):
            inner_verts = get_hex_vertices(centers[i, 0], centers[i, 1], 1.0, 0.0)
            for v in inner_verts:
                # Check if v is inside outer hexagon with side = outer_side
                # For a regular hexagon centered at origin, the condition is:
                # For each edge, the cross product must be consistent
                inside = True
                for j in range(6):
                    p1 = outer_verts[j]
                    p2 = outer_verts[(j + 1) % 6]
                    edge = p2 - p1
                    # Normal pointing inward
                    cross = edge[0] * (v[1] - p1[1]) - edge[1] * (v[0] - p1[0])
                    # For a convex polygon, all cross products should have the same sign
                    # For CCW polygon, interior is to the left, so cross >= 0
                    if cross < -1e-8:
                        inside = False
                        break
                if not inside:
                    # Need to increase outer_side to include this vertex
                    # The distance from origin to this vertex projected onto the appropriate direction
                    # For a regular hexagon, the minimum side to contain a point at distance r
                    # in direction theta is r / cos(angle_diff) where angle_diff is the angle
                    # between theta and the nearest apothem direction
                    r = np.linalg.norm(v)
                    theta = np.arctan2(v[1], v[0])
                    outer_angle_rad = np.radians(outer_angle_deg)
                    # The apothem directions are at outer_angle + 30 + 60*k degrees
                    # Find the closest apothem direction
                    best_cos = 0.0
                    for k in range(6):
                        apothem_angle = outer_angle_rad + np.pi / 6 + np.pi / 3 * k
                        cos_diff = np.cos(theta - apothem_angle)
                        if cos_diff > best_cos:
                            best_cos = cos_diff
                    if best_cos > 1e-8:
                        required = r / best_cos
                        if required > max_required_side:
                            max_required_side = required
                    else:
                        # Point is in a corner direction, need larger side
                        # The maximum distance from center to vertex of outer hex is s
                        # So we need s >= r
                        if r > max_required_side:
                            max_required_side = r
        
        return max_required_side
    
    # Optimize over the outer hexagon angle to minimize the required side length
    def objective(angle_params):
        angle_deg = angle_params[0]
        # Start with a reasonable estimate
        s = 5.0
        for _ in range(20):
            required = check_fit(s, angle_deg)
            if required <= s:
                break
            s = required
        return s
    
    # Try different angles
    best_s = np.inf
    best_angle = 0.0
    for angle_deg in np.linspace(0, 60, 61):
        s = objective([angle_deg])
        if s < best_s:
            best_s = s
            best_angle = angle_deg
    
    # Fine-tune the angle
    result = minimize(objective, [best_angle], bounds=[(best_angle - 5, best_angle + 5)],
                      method='Nelder-Mead', options={'xatol': 1e-4, 'fatol': 1e-8})
    if result.fun < best_s:
        best_s = result.fun
        best_angle = result.x[0]
    
    outer_hex_data = np.array([0.0, 0.0, best_angle])
    outer_hex_side_length = best_s
    
    return inner_hex_data, outer_hex_data, outer_hex_side_length


# EVOLVE-BLOCK-END
