# EVOLVE-BLOCK-START
import numpy as np
import time


def _hex_vertices(cx, cy, d, angle_deg):
    """Return vertices of a regular hexagon centered at (cx,cy) with side d and angle angle_deg."""
    a = np.radians(angle_deg)
    angles = a + np.radians(60 * np.arange(6))
    return np.column_stack([cx + d * np.cos(angles), cy + d * np.sin(angles)])


def _min_outer_side_for_rotation(inner_centers, rot_deg):
    """
    Given inner hex centers (unit hexagons, all axis-aligned internally),
    find minimum side length s of a regular hexagon (centered at origin, rotated by rot_deg)
    that contains all 12 inner hexagons.
    Returns (s, best_center_offset) or (inf, None) if infeasible.
    """
    rot = np.radians(rot_deg)
    
    # Vertices of unit hexagon (side 1) centered at origin, axis-aligned
    unit_verts = np.array([[np.cos(np.radians(60*i)), np.sin(np.radians(60*i))] for i in range(6)])
    
    # For a regular hexagon with side s centered at (ox, oy) with rotation rot:
    # A point p is inside iff for each edge i (0..5):
    #   cos(rot + 60*i + 30) * (p_x - ox) + sin(rot + 60*i + 30) * (p_y - oy) <= s * sqrt(3) / 2
    
    # We need this for all inner vertices p = c_j + u_k
    
    # This is a linear program: minimize s subject to linear constraints in (ox, oy, s)
    
    from scipy.optimize import linprog
    
    # All inner vertices (72 total)
    all_verts = []
    for c in inner_centers:
        for u in unit_verts:
            all_verts.append(c + u)
    all_verts = np.array(all_verts)  # shape (72, 2)
    
    n_constraints = 6 * len(all_verts)
    A = np.zeros((n_constraints, 3))  # variables: [ox, oy, s]
    b = np.zeros(n_constraints)
    
    idx = 0
    for i in range(6):
        angle = rot + np.radians(60*i + 30)
        ci = np.cos(angle)
        si = np.sin(angle)
        for j in range(len(all_verts)):
            # ci*(vx - ox) + si*(vy - oy) <= s*sqrt(3)/2
            # -ci*ox - si*oy - sqrt(3)/2 * s <= -(ci*vx + si*vy)
            A[idx, 0] = -ci
            A[idx, 1] = -si
            A[idx, 2] = -np.sqrt(3)/2
            b[idx] = -(ci * all_verts[j, 0] + si * all_verts[j, 1])
            idx += 1
    
    # Objective: minimize s
    c_obj = np.array([0, 0, 1])
    
    # Bounds: ox, oy free, s >= 0
    bounds = [(None, None), (None, None), (0, None)]
    
    res = linprog(c_obj, A_ub=A, b_ub=b, bounds=bounds, method='highs')
    
    if res.status == 0:
        s = res.x[2]
        ox = res.x[0]
        oy = res.x[1]
        return s, (ox, oy)
    else:
        return np.inf, None


def hexagon_packing_12():
    """
    Constructs a packing of 12 disjoint unit regular hexagons inside a larger regular hexagon, maximizing 1/outer_hex_side_length.
    Returns
        inner_hex_data: np.ndarray of shape (12,3), where each row is of the form (x, y, angle_degrees) containing the (x,y) coordinates and angle_degree of the respective inner hexagon.
        outer_hex_data: np.ndarray of shape (3,) of form (x,y,angle_degree) containing the (x,y) coordinates and angle_degree of the outer hexagon.
        outer_hex_side_length: float representing the side length of the outer hexagon.
    """
    start_time = time.time()
    deadline = start_time + 600.0
    
    sqrt3 = np.sqrt(3)
    
    # Define 12 hexagon centers in a hexagonal lattice pattern with spacing 2
    # This ensures non-overlap since the minimum distance between parallel edges of unit hexagons is sqrt(3),
    # and with center distance 2, we have margin.
    centers = np.array([
        [0, 0],
        [2, 0],
        [1, sqrt3],
        [-1, sqrt3],
        [-2, 0],
        [-1, -sqrt3],
        [1, -sqrt3],
        [3, sqrt3],
        [0, 2*sqrt3],
        [-3, sqrt3],
        [-3, -sqrt3],
        [0, -2*sqrt3],
    ])
    
    # All inner hexagons are axis-aligned (angle = 0)
    inner_hex_data = np.column_stack([centers, np.zeros(12)])
    
    # Now optimize the outer hexagon rotation to minimize its side length
    best_s = np.inf
    best_rot = 0
    best_center = (0, 0)
    
    # Try rotations from 0 to 60 degrees with steps of 0.5 degrees
    for rot_deg in np.linspace(0, 60, 121):
        if time.time() > deadline - 30:
            break
        s, center = _min_outer_side_for_rotation(centers, rot_deg)
        if s < best_s:
            best_s = s
            best_rot = rot_deg
            best_center = center
    
    # Refine with finer search around best_rot
    if time.time() < deadline - 60:
        for rot_deg in np.linspace(max(0, best_rot - 1), min(60, best_rot + 1), 41):
            if time.time() > deadline - 30:
                break
            s, center = _min_outer_side_for_rotation(centers, rot_deg)
            if s < best_s:
                best_s = s
                best_rot = rot_deg
                best_center = center
    
    outer_hex_data = np.array([best_center[0], best_center[1], best_rot])
    
    return inner_hex_data, outer_hex_data, best_s


# EVOLVE-BLOCK-END
