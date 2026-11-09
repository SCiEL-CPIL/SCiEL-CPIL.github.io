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


def _compute_objective(params):
    """
    params = [d, rotation_deg, s]
    d: lattice spacing (distance between adjacent hexagon centers)
    rotation_deg: global rotation of the lattice
    s: outer hexagon side length
    
    Returns: negative of 1/s (to minimize) or a large penalty if invalid.
    """
    d, rot_deg, s = params
    
    if s < np.sqrt(12) or s <= 0 or d <= 0:
        return 1e10
    
    # Triangular lattice basis vectors with spacing d and 60 degree angle
    # e1 and e2 have length d and are at 60 degrees
    e1_base = np.array([d, 0])
    e2_base = np.array([d * 0.5, d * np.sqrt(3) / 2])
    
    # Apply global rotation
    theta = np.deg2rad(rot_deg)
    R = np.array([[np.cos(theta), -np.sin(theta)],
                  [np.sin(theta), np.cos(theta)]])
    
    e1 = R @ e1_base
    e2 = R @ e2_base
    
    # Define 12 lattice points in a compact hexagonal-like arrangement
    # Using a 3x4 parallelogram centered at origin
    # Lattice coords: i in {-1, 0, 1, 2}, j in {-1, 0, 1}
    # That gives 12 points
    lattice_points = []
    for i in range(-1, 3):
        for j in range(-1, 2):
            lattice_points.append((i, j))
    
    # Compute positions and center them
    positions = np.array([i * e1 + j * e2 for i, j in lattice_points])
    center = np.mean(positions, axis=0)
    positions = positions - center
    
    # Check separation between all pairs of inner hexagons
    for i in range(12):
        for j in range(i + 1, 12):
            verts_i = _hexagon_vertices(positions[i][0], positions[i][1], 1.0, 0)
            verts_j = _hexagon_vertices(positions[j][0], positions[j][1], 1.0, 0)
            if not _separation_check(verts_i, verts_j):
                return 1e10
    
    # Check all inner hexagons are inside the outer hexagon
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
    Uses a crystallographic lattice approach: 12 hexagons on a triangular lattice,
    optimizing lattice spacing, global rotation, and outer hexagon size.
    """
    # Initial guess: d ~ sqrt(3) for touching hexagons, rotation 0, s ~ 6
    x0 = np.array([np.sqrt(3), 0.0, 6.0])
    
    # Bounds
    bounds = [(1.5, 5.0), (0.0, 60.0), (np.sqrt(12), 20.0)]
    
    # Optimize
    result = minimize(_compute_objective, x0, method='L-BFGS-B', bounds=bounds,
                      options={'maxiter': 1000, 'ftol': 1e-14, 'gtol': 1e-12})
    
    d, rot_deg, s = result.x
    
    # If optimization failed, use a fallback configuration
    if not result.success or result.fun > -0.1:
        # Fallback: use a valid configuration
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
    
    # Compute final positions
    e1_base = np.array([d, 0])
    e2_base = np.array([d * 0.5, d * np.sqrt(3) / 2])
    
    theta = np.deg2rad(rot_deg)
    R = np.array([[np.cos(theta), -np.sin(theta)],
                  [np.sin(theta), np.cos(theta)]])
    
    e1 = R @ e1_base
    e2 = R @ e2_base
    
    lattice_points = []
    for i in range(-1, 3):
        for j in range(-1, 2):
            lattice_points.append((i, j))
    
    positions = np.array([i * e1 + j * e2 for i, j in lattice_points])
    center = np.mean(positions, axis=0)
    positions = positions - center
    
    inner_hex_data = np.zeros((12, 3))
    for idx in range(12):
        inner_hex_data[idx, 0] = positions[idx][0]
        inner_hex_data[idx, 1] = positions[idx][1]
        inner_hex_data[idx, 2] = 0.0
    
    outer_hex_data = np.array([0.0, 0.0, 0.0])
    outer_hex_side_length = float(s)
    
    return inner_hex_data, outer_hex_data, outer_hex_side_length


# EVOLVE-BLOCK-END
