# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize


def _hexagon_vertices(cx, cy, d, angle_deg):
    """Return the 6 vertices of a regular hexagon centered at (cx, cy) with side d and angle angle_deg."""
    a = np.deg2rad(angle_deg)
    angles = a + np.deg2rad(60 * np.arange(6))
    return np.column_stack([cx + d * np.cos(angles), cy + d * np.sin(angles)])


def _hexagon_edge_normals(vertices):
    """Return the outward edge normals of a convex polygon given its vertices in CCW order."""
    n = len(vertices)
    normals = np.zeros((n, 2))
    for i in range(n):
        p1 = vertices[i]
        p2 = vertices[(i + 1) % n]
        edge = p2 - p1
        normals[i] = np.array([edge[1], -edge[0]])
        norm = np.linalg.norm(normals[i])
        if norm > 1e-15:
            normals[i] /= norm
    return normals


def _separation_check(hex1_verts, hex2_verts):
    """
    Check if two convex hexagons are separated (not overlapping).
    Uses the separating axis theorem.
    Returns True if separated, False if overlapping.
    """
    normals1 = _hexagon_edge_normals(hex1_verts)
    normals2 = _hexagon_edge_normals(hex2_verts)
    all_normals = np.vstack([normals1, normals2])
    
    for normal in all_normals:
        proj1 = hex1_verts @ normal
        proj2 = hex2_verts @ normal
        min1, max1 = np.min(proj1), np.max(proj1)
        min2, max2 = np.min(proj2), np.max(proj2)
        if max1 < min2 - 1e-6 or max2 < min1 - 1e-6:
            return True
    return False


def _is_inside_outer(v, outer_verts):
    """
    Check if point v is inside the outer hexagon (CCW vertices).
    For each edge (p1, p2), cross(p2-p1, v-p1) must be >= -1e-6.
    """
    n = len(outer_verts)
    for i in range(n):
        p1 = outer_verts[i]
        p2 = outer_verts[(i + 1) % n]
        edge = p2 - p1
        vec = v - p1
        cross = edge[0] * vec[1] - edge[1] * vec[0]
        if cross < -1e-6:
            return False
    return True


def _get_lattice_positions(e1, e2, rotation_deg):
    """
    Get positions of 12 hexagons on a triangular lattice.
    Uses a compact hexagonal arrangement of 12 points.
    Lattice coordinates (i,j) where position = i*e1 + j*e2.
    """
    # 12 points in a compact hexagonal cluster
    lattice_points = [
        (0, 0),
        (1, 0), (-1, 0), (0, 1), (0, -1), (1, -1), (-1, 1),
        (2, 0), (-2, 0), (0, 2), (0, -2), (2, -2),
    ]
    
    rot = np.deg2rad(rotation_deg)
    rot_matrix = np.array([[np.cos(rot), -np.sin(rot)],
                           [np.sin(rot), np.cos(rot)]])
    
    positions = []
    for i, j in lattice_points:
        pos = i * e1 + j * e2
        pos = rot_matrix @ pos
        positions.append(pos)
    return np.array(positions)


def _compute_objective(params):
    """
    params = [a, b, theta, rotation_deg, s]
    a, b: lattice basis vector lengths
    theta: angle between basis vectors (radians)
    rotation_deg: global rotation of the lattice
    s: outer hexagon side length
    
    Returns: negative of 1/s (to minimize) or a large penalty if invalid.
    """
    a, b, theta, rotation_deg, s = params
    
    if s < np.sqrt(12) or s <= 0 or a <= 0 or b <= 0:
        return 1e10
    
    e1 = np.array([a, 0])
    e2 = np.array([b * np.cos(theta), b * np.sin(theta)])
    
    positions = _get_lattice_positions(e1, e2, rotation_deg)
    
    for i in range(12):
        for j in range(i + 1, 12):
            verts_i = _hexagon_vertices(positions[i][0], positions[i][1], 1.0, 0)
            verts_j = _hexagon_vertices(positions[j][0], positions[j][1], 1.0, 0)
            if not _separation_check(verts_i, verts_j):
                return 1e10
    
    outer_verts = _hexagon_vertices(0, 0, s, 0)
    for i in range(12):
        verts_i = _hexagon_vertices(positions[i][0], positions[i][1], 1.0, 0)
        for v in verts_i:
            if not _is_inside_outer(v, outer_verts):
                return 1e10
    
    return -1.0 / s


def hexagon_packing_12():
    """
    Constructs a packing of 12 disjoint unit regular hexagons inside a larger regular hexagon, maximizing 1/outer_hex_side_length.
    Returns
        inner_hex_data: np.ndarray of shape (12,3), where each row is of the form (x, y, angle_degrees)
        outer_hex_data: np.ndarray of shape (3,) of form (x,y,angle_degree)
        outer_hex_side_length: float
    """
    x0 = np.array([2.0, 2.0, np.pi / 3, 0.0, 8.0])
    
    bounds = [(1.5, 5.0), (1.5, 5.0), (np.pi / 4, 2 * np.pi / 3), (-180, 180), (np.sqrt(12), 20.0)]
    
    best_result = None
    best_fun = np.inf
    
    initial_guesses = [
        np.array([2.0, 2.0, np.pi / 3, 0.0, 8.0]),
        np.array([2.0, 2.0, np.pi / 3, 30.0, 8.0]),
        np.array([2.0, 2.0, np.pi / 3, 60.0, 8.0]),
        np.array([2.0, 2.0, np.pi / 3, -30.0, 8.0]),
        np.array([2.0, 2.0, np.pi / 3, -60.0, 8.0]),
    ]
    
    for x0_guess in initial_guesses:
        result = minimize(_compute_objective, x0_guess, method='L-BFGS-B', bounds=bounds,
                          options={'maxiter': 1000, 'ftol': 1e-12, 'gtol': 1e-10})
        
        if result.fun < best_fun:
            best_fun = result.fun
            best_result = result
    
    a, b, theta, rotation_deg, s = best_result.x
    
    if not best_result.success or best_fun > -0.1:
        inner_hex_data = np.array(
            [
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
                [3.75, -2.17, 0],
                [0, -4, 0],
            ]
        )
        outer_hex_data = np.array([0, 0, 0])
        outer_hex_side_length = 8.0
        return inner_hex_data, outer_hex_data, outer_hex_side_length
    
    e1 = np.array([a, 0])
    e2 = np.array([b * np.cos(theta), b * np.sin(theta)])
    positions = _get_lattice_positions(e1, e2, rotation_deg)
    
    inner_hex_data = np.zeros((12, 3))
    for idx, pos in enumerate(positions):
        inner_hex_data[idx, 0] = pos[0]
        inner_hex_data[idx, 1] = pos[1]
        inner_hex_data[idx, 2] = 0.0
    
    outer_hex_data = np.array([0.0, 0.0, 0.0])
    outer_hex_side_length = float(s)
    
    return inner_hex_data, outer_hex_data, outer_hex_side_length


# EVOLVE-BLOCK-END
