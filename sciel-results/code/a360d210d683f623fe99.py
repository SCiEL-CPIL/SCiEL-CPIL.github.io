import numpy as np


def hexagon_packing_12():
    """
    Constructs a packing of 12 disjoint unit regular hexagons inside a larger regular hexagon, maximizing 1/outer_hex_side_length.
    Returns
        inner_hex_data: np.ndarray of shape (12,3), where each row is (x, y, angle_degrees)
        outer_hex_data: np.ndarray of shape (3,) of form (x,y,angle_degree)
        outer_hex_side_length: float
    """
    sqrt3 = np.sqrt(3)
    
    # Flower pattern: 1 center + 6 ring1 + 5 ring2 (corners, leaving one gap)
    # Ring 1: distance sqrt3 from center, at angles 0, 60, 120, 180, 240, 300
    # Ring 2 corners: distance 2*sqrt3, at same angles
    # Leave out the corner at angle 0 (right side) to allow tighter outer hex
    
    inner_centers = []
    # Center
    inner_centers.append((0.0, 0.0))
    
    # Ring 1: 6 hexagons
    for i in range(6):
        angle = np.radians(60 * i)
        cx = sqrt3 * np.cos(angle)
        cy = sqrt3 * np.sin(angle)
        inner_centers.append((cx, cy))
    
    # Ring 2: 5 corners (skip angle 0, i.e., skip i=0)
    for i in range(1, 6):
        angle = np.radians(60 * i)
        cx = 2 * sqrt3 * np.cos(angle)
        cy = 2 * sqrt3 * np.sin(angle)
        inner_centers.append((cx, cy))
    
    inner_hex_data = np.array([[cx, cy, 0.0] for cx, cy in inner_centers])
    
    # Compute the minimal enclosing regular hexagon
    # The outer hexagon is centered at origin, with some angle and side s
    # We need to find s such that all vertices of all inner hexagons are inside
    
    def get_hex_vertices(cx, cy, side, angle_deg):
        angle_rad = np.radians(angle_deg)
        verts = []
        for i in range(6):
            a = angle_rad + np.radians(60 * i)
            verts.append((cx + side * np.cos(a), cy + side * np.sin(a)))
        return np.array(verts)
    
    # Collect all vertices of all inner hexagons
    all_verts = []
    for i in range(12):
        cx, cy, ang = inner_hex_data[i]
        verts = get_hex_vertices(cx, cy, 1.0, ang)
        all_verts.extend(verts.tolist())
    all_verts = np.array(all_verts)
    
    # The outer hexagon has center (0,0), angle 0, side s
    # For a point (px, py) to be inside a regular hexagon centered at origin with side s and angle 0:
    # The hexagon has vertices at angles 0, 60, 120, 180, 240, 300 at distance s from center
    # The inradius is s * sqrt(3) / 2
    # A point is inside if its distance to each edge is non-negative
    
    def point_in_hexagon(px, py, s, angle_deg=0.0):
        """Check if point (px,py) is inside hexagon centered at origin with side s and angle angle_deg."""
        angle_rad = np.radians(angle_deg)
        # Get the 6 edge normals
        for i in range(6):
            # Edge i is between vertex i and vertex i+1
            # Vertex i is at angle (angle_rad + 60*i)
            # The edge from vertex i to vertex i+1 has a normal pointing inward
            # Normal direction is at angle (angle_rad + 60*i + 30)
            normal_angle = angle_rad + np.radians(60 * i + 30)
            nx = np.cos(normal_angle)
            ny = np.sin(normal_angle)
            # The distance from center to this edge is s * sqrt(3) / 2
            # The point must satisfy: nx*px + ny*py <= s * sqrt(3) / 2
            if nx * px + ny * py > s * sqrt3 / 2 + 1e-10:
                return False
        return True
    
    # Find minimal s such that all vertices are inside
    # Binary search on s
    lo = 0.0
    hi = 10.0
    for _ in range(100):
        mid = (lo + hi) / 2
        all_inside = True
        for px, py in all_verts:
            if not point_in_hexagon(px, py, mid):
                all_inside = False
                break
        if all_inside:
            hi = mid
        else:
            lo = mid
    
    s = hi + 1e-9  # tiny margin
    
    outer_hex_data = np.array([0.0, 0.0, 0.0])
    
    return inner_hex_data, outer_hex_data, s
