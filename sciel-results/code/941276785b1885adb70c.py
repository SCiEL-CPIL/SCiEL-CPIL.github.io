# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import time


def _hexagon_vertices(cx, cy, d, a_deg):
    """Compute vertices of a regular hexagon with center (cx,cy), side d, angle a_deg."""
    a = np.radians(a_deg)
    angles = a + np.radians(60 * np.arange(6))
    return np.column_stack([cx + d * np.cos(angles), cy + d * np.sin(angles)])


def _edge_normals(vertices):
    """Compute outward normals for each edge of a polygon given as (6,2) vertices in CCW order."""
    edges = np.roll(vertices, -1, axis=0) - vertices
    # For CCW polygon, outward normal is (dy, -dx) normalized
    normals = np.column_stack([edges[:, 1], -edges[:, 0]])
    norms = np.linalg.norm(normals, axis=1, keepdims=True)
    norms[norms < 1e-15] = 1.0
    return normals / norms


def _separation_distance(v1, n1, v2, n2):
    """Compute minimum separation using SAT (separating axis test).
    Returns negative value if overlapping, positive if separated.
    """
    min_sep = np.inf
    # Project both polygons onto each normal of both polygons
    for normals, verts in [(n1, v1), (n2, v2)]:
        for i in range(6):
            n = normals[i]
            proj1 = verts @ n
            proj2 = v2 @ n
            min1, max1 = proj1.min(), proj1.max()
            min2, max2 = proj2.min(), proj2.max()
            # Distance between intervals
            if max1 <= min2:
                sep = min2 - max1
            elif max2 <= min1:
                sep = min1 - max2
            else:
                sep = -max(min1 - max2, min2 - max1)  # negative means overlap
            min_sep = min(min_sep, sep)
    return min_sep


def _check_containment(inner_verts, outer_verts):
    """Check if all inner vertices are inside or on the outer hexagon.
    Returns minimum clearance (negative if outside).
    """
    min_clearance = np.inf
    for i in range(6):
        p1 = outer_verts[i]
        p2 = outer_verts[(i + 1) % 6]
        edge = p2 - p1
        # Inward normal for CCW polygon
        normal = np.array([edge[1], -edge[0]])
        norm_len = np.linalg.norm(normal)
        if norm_len < 1e-15:
            continue
        normal = normal / norm_len
        # For each inner vertex, compute signed distance to this edge
        for v in inner_verts:
            d = np.dot(v - p1, normal)
            min_clearance = min(min_clearance, d)
    return min_clearance


def _packing_cost(params):
    """Cost function: minimize outer hexagon side length while ensuring valid packing.
    params: [x_outer, y_outer, angle_outer_deg, s_outer, 
             x1, y1, a1, x2, y2, a2, ..., x12, y12, a12]
    """
    x_outer = params[0]
    y_outer = params[1]
    angle_outer = params[2]
    s_outer = params[3]
    
    if s_outer < np.sqrt(12) - 0.01:
        return 1e10
    
    inner_params = params[4:]
    inner_hexes = inner_params.reshape(12, 3)
    
    # Compute outer vertices
    outer_verts = _hexagon_vertices(x_outer, y_outer, s_outer, angle_outer)
    
    # Compute all inner vertices
    all_inner_verts = []
    all_inner_normals = []
    for i in range(12):
        cx, cy, a_deg = inner_hexes[i]
        verts = _hexagon_vertices(cx, cy, 1.0, a_deg)
        all_inner_verts.append(verts)
        all_inner_normals.append(_edge_normals(verts))
    
    # Check containment: all inner vertices must be inside outer hexagon
    total_containment_penalty = 0.0
    for i in range(12):
        clearance = _check_containment(all_inner_verts[i], outer_verts)
        if clearance < 0:
            total_containment_penalty += clearance * clearance
    
    # Check separation between all pairs of inner hexagons
    total_separation_penalty = 0.0
    for i in range(12):
        for j in range(i + 1, 12):
            sep = _separation_distance(all_inner_verts[i], all_inner_normals[i], 
                                       all_inner_verts[j], all_inner_normals[j])
            if sep < 0:
                total_separation_penalty += sep * sep
    
    # Objective: minimize s_outer + penalties
    cost = s_outer + 1000.0 * total_containment_penalty + 1000.0 * total_separation_penalty
    
    return cost


def _initial_honeycomb_params():
    """Generate initial parameters for a 3-4-3 honeycomb arrangement."""
    # Outer hexagon centered at origin
    x_outer, y_outer, angle_outer = 0.0, 0.0, 0.0
    
    # Initial guess for outer side length
    s_outer = 5.0
    
    # Honeycomb positions (3-4-3 arrangement)
    # Using axial coordinates for hexagonal grid
    # Center hexagon at origin
    positions = []
    angles = []
    
    # Row 0 (top): 3 hexagons
    for k in range(-1, 2):
        x = k * 2.5
        y = 2.1650635094610965
        positions.append([x, y])
        angles.append(0.0)
    
    # Row 1 (middle): 4 hexagons
    for k in range(-2, 2):
        x = k * 2.5
        y = 0.0
        positions.append([x, y])
        angles.append(0.0)
    
    # Row 2 (bottom): 3 hexagons
    for k in range(-1, 2):
        x = k * 2.5
        y = -2.1650635094610965
        positions.append([x, y])
        angles.append(0.0)
    
    params = [x_outer, y_outer, angle_outer, s_outer]
    for pos, ang in zip(positions, angles):
        params.extend([pos[0], pos[1], ang])
    
    return np.array(params)


def hexagon_packing_12():
    """
    Constructs a packing of 12 disjoint unit regular hexagons inside a larger regular hexagon, maximizing 1/outer_hex_side_length.
    Returns
        inner_hex_data: np.ndarray of shape (12,3), where each row is of the form (x, y, angle_degrees) containing the (x,y) coordinates and angle_degree of the respective inner hexagon.
        outer_hex_data: np.ndarray of shape (3,) of form (x,y,angle_degree) containing the (x,y) coordinates and angle_degree of the outer hexagon.
        outer_hex_side_length: float representing the side length of the outer hexagon.
    """
    deadline = time.time() + 600.0
    
    # Start with honeycomb initial guess
    x0 = _initial_honeycomb_params()
    
    # Try multiple random initializations with different rotations
    best_cost = np.inf
    best_params = x0.copy()
    
    n_restarts = 10
    
    for restart in range(n_restarts):
        if time.time() > deadline - 60:
            break
        
        if restart == 0:
            params_init = x0.copy()
        else:
            params_init = x0.copy()
            # Add some randomness to angles and positions
            noise_scale = 0.5
            params_init[4:] += np.random.randn(36) * noise_scale
            # Keep outer hexagon roughly centered
            params_init[0] = np.random.randn() * 0.5
            params_init[1] = np.random.randn() * 0.5
            params_init[2] = np.random.uniform(-30, 30)
            params_init[3] = np.random.uniform(4.0, 6.0)
        
        try:
            result = minimize(_packing_cost, params_init, method='L-BFGS-B', 
                             options={'maxiter': 500, 'maxls': 50, 'ftol': 1e-10, 'gtol': 1e-8})
            
            if result.fun < best_cost:
                best_cost = result.fun
                best_params = result.x.copy()
        except Exception:
            continue
    
    # Extract best solution
    x_outer, y_outer, angle_outer, s_outer = best_params[0], best_params[1], best_params[2], best_params[3]
    inner_params = best_params[4:].reshape(12, 3)
    
    # Verify the solution is valid; if not, fall back to a known valid configuration
    outer_verts = _hexagon_vertices(x_outer, y_outer, s_outer, angle_outer)
    
    valid = True
    all_inner_verts = []
    all_inner_normals = []
    for i in range(12):
        cx, cy, a_deg = inner_params[i]
        if not (np.isfinite(cx) and np.isfinite(cy) and np.isfinite(a_deg)):
            valid = False
            break
        verts = _hexagon_vertices(cx, cy, 1.0, a_deg)
        all_inner_verts.append(verts)
        all_inner_normals.append(_edge_normals(verts))
    
    if valid:
        # Check containment
        for i in range(12):
            clearance = _check_containment(all_inner_verts[i], outer_verts)
            if clearance < -1e-6:
                valid = False
                break
    
    if valid:
        # Check separation
        for i in range(12):
            if not valid:
                break
            for j in range(i + 1, 12):
                sep = _separation_distance(all_inner_verts[i], all_inner_normals[i],
                                           all_inner_verts[j], all_inner_normals[j])
                if sep < -1e-6:
                    valid = False
                    break
    
    if valid and s_outer >= np.sqrt(12):
        inner_hex_data = inner_params
        outer_hex_data = np.array([x_outer, y_outer, angle_outer])
        return inner_hex_data, outer_hex_data, s_outer
    
    # Fallback: use the original grid arrangement (known to be valid)
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


# EVOLVE-BLOCK-END
