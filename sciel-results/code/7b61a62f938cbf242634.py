import numpy as np
from scipy.optimize import minimize_scalar


def hexagon_packing_11():
    """
    Constructs a packing of 11 disjoint unit regular hexagons inside a larger regular hexagon,
    maximizing 1/outer_hex_side_length.
    """
    sqrt3 = np.sqrt(3)

    # Compact 11-hexagon cluster defined in hexagonal lattice coordinates.
    # Adjacent hexagon centers in a hex lattice are distance 2 apart (for unit hexagons).
    # We use a compact arrangement and then scale slightly to ensure gap > 1e-6.
    
    # Define centers for a tight 11-hexagon cluster
    # Using a pattern that minimizes the bounding regular hexagon
    centers_base = np.array([
        # Row y = 2*sqrt(3): 2 hexagons
        [0.0, 2*sqrt3],
        [2.0, 2*sqrt3],
        # Row y = sqrt(3): 3 hexagons
        [-1.0, sqrt3],
        [1.0, sqrt3],
        [3.0, sqrt3],
        # Row y = 0: 3 hexagons
        [-2.0, 0.0],
        [0.0, 0.0],
        [2.0, 0.0],
        # Row y = -sqrt(3): 2 hexagons
        [-1.0, -sqrt3],
        [1.0, -sqrt3],
        # Row y = -2*sqrt(3): 1 hexagon
        [0.0, -2*sqrt3],
    ])

    # Check minimum distance between any two centers
    min_dist = np.inf
    for i in range(11):
        for j in range(i+1, 11):
            d = np.linalg.norm(centers_base[i] - centers_base[j])
            if d < min_dist:
                min_dist = d

    # Scale factor to ensure all gaps > 1e-6
    # Gap = min_dist - 2 (for touching hexagons, gap = 0 when dist = 2)
    # We need gap > 1e-6, so min_dist > 2 + 1e-6
    # Scale factor = (2 + 1e-6) / min_dist if min_dist <= 2 + 1e-6
    if min_dist <= 2.0 + 1e-6:
        scale = (2.0 + 1e-5) / min_dist  # small margin
        centers = centers_base * scale
    else:
        centers = centers_base

    # Recompute min_dist after scaling
    min_dist_scaled = np.inf
    for i in range(11):
        for j in range(i+1, 11):
            d = np.linalg.norm(centers[i] - centers[j])
            if d < min_dist_scaled:
                min_dist_scaled = d

    # Compute all vertices of all 11 hexagons for a given rotation
    vertex_angles = np.pi/3 * np.arange(6)
    vx = np.cos(vertex_angles)
    vy = np.sin(vertex_angles)

    def compute_required_side(rotation_angle):
        """Compute the minimum side length of enclosing regular hexagon given rotation."""
        cos_r = np.cos(rotation_angle)
        sin_r = np.sin(rotation_angle)
        rot_matrix = np.array([[cos_r, -sin_r], [sin_r, cos_r]])

        rotated_centers = centers @ rot_matrix.T

        # All vertices: shape (66, 2)
        all_vertices = np.zeros((66, 2))
        for i in range(11):
            all_vertices[i*6:(i+1)*6, 0] = rotated_centers[i, 0] + vx
            all_vertices[i*6:(i+1)*6, 1] = rotated_centers[i, 1] + vy

        # Outer hexagon centered at origin.
        # Edge normal directions: angle = rotation_angle + pi/6 + pi/3 * k for k=0..5
        normal_angles = rotation_angle + np.pi/6 + np.pi/3 * np.arange(6)
        normals = np.column_stack([np.cos(normal_angles), np.sin(normal_angles)])

        # Max projection of all vertices along each normal
        projections = all_vertices @ normals.T
        max_projections = np.max(projections, axis=1)

        apothem = np.max(max_projections)
        side = 2.0 * apothem / sqrt3

        return side

    # Optimize rotation to minimize side length
    result = minimize_scalar(compute_required_side, bounds=(0, np.pi/3), method='bounded',
                             options={'xatol': 1e-12})

    optimal_rotation = result.x
    optimal_side = result.fun

    # Get the rotated centers
    cos_r = np.cos(optimal_rotation)
    sin_r = np.sin(optimal_rotation)
    rot_matrix = np.array([[cos_r, -sin_r], [sin_r, cos_r]])
    rotated_centers = centers @ rot_matrix.T

    # Inner hex data
    inner_angle_deg = np.degrees(optimal_rotation)
    inner_hex_data = np.zeros((11, 3))
    for i in range(11):
        inner_hex_data[i, 0] = rotated_centers[i, 0]
        inner_hex_data[i, 1] = rotated_centers[i, 1]
        inner_hex_data[i, 2] = inner_angle_deg

    # Outer hexagon
    outer_hex_data = np.array([0.0, 0.0, inner_angle_deg])
    outer_hex_side_length = float(optimal_side)

    return inner_hex_data, outer_hex_data, outer_hex_side_length
