# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import differential_evolution, minimize, NonlinearConstraint

_START_TIME = time.time()
_DEADLINE = _START_TIME + 580  # safety margin

def _hex_vertices(cx, cy, s, a_deg):
    """Return 6 vertices of a regular hexagon with center (cx,cy), side s, angle a_deg."""
    a = np.radians(a_deg)
    angles = a + np.radians(60) * np.arange(6)
    return np.column_stack([cx + s * np.cos(angles), cy + s * np.sin(angles)])

def _hex_edge_normals(vertices):
    """Return 12 edge normal vectors (6 edges, each with inward and outward)."""
    normals = []
    for i in range(6):
        p1 = vertices[i]
        p2 = vertices[(i + 1) % 6]
        edge = p2 - p1
        # Perpendicular (outward normal)
        n = np.array([edge[1], -edge[0]])
        n /= np.linalg.norm(n)
        # Determine if outward: check direction from center
        center = np.mean(vertices, axis=0)
        mid = (p1 + p2) / 2
        if np.dot(n, mid - center) < 0:
            n = -n
        normals.append(n)
        normals.append(-n)
    return np.array(normals)

def _check_separation(vertices_i, normals_i, vertices_j, normals_j):
    """Check if two hexagons are separated (non-intersecting)."""
    for n in normals_i:
        proj_i = np.dot(vertices_i, n)
        proj_j = np.dot(vertices_j, n)
        min_i, max_i = proj_i.min(), proj_i.max()
        min_j, max_j = proj_j.min(), proj_j.max()
        if max_i + 1e-7 < min_j or max_j + 1e-7 < min_i:
            continue  # Separated on this normal
        else:
            return False
    return True

def _check_all_separated(inner_hex_data):
    """Check all pairs of inner hexagons are separated."""
    n = len(inner_hex_data)
    vertices_list = []
    normals_list = []
    for i in range(n):
        x, y, ang = inner_hex_data[i]
        verts = _hex_vertices(x, y, 1.0, ang)
        normals = _hex_edge_normals(verts)
        vertices_list.append(verts)
        normals_list.append(normals)
    
    for i in range(n):
        for j in range(i + 1, n):
            if not _check_separation(vertices_list[i], normals_list[i], vertices_list[j], normals_list[j]):
                return False
    return True

def _check_in_outer(inner_hex_data, outer_center, outer_angle, outer_side):
    """Check all inner vertices are inside the outer hexagon."""
    outer_verts = _hex_vertices(outer_center[0], outer_center[1], outer_side, outer_angle)
    
    # Get edges of outer hexagon
    for i in range(6):
        p1 = outer_verts[i]
        p2 = outer_verts[(i + 1) % 6]
        edge = p2 - p1
        # Inward normal
        n = np.array([edge[1], -edge[0]])
        n /= np.linalg.norm(n)
        mid = (p1 + p2) / 2
        center = np.mean(outer_verts, axis=0)
        if np.dot(n, mid - center) < 0:
            n = -n
        
        # Check all inner vertices
        for hex_idx in range(len(inner_hex_data)):
            x, y, ang = inner_hex_data[hex_idx]
            verts = _hex_vertices(x, y, 1.0, ang)
            for v in verts:
                cross = np.cross(p2 - p1, v - p1)
                if cross < -1e-6:
                    return False
    return True

def _validate_solution(inner_hex_data, outer_center, outer_angle, outer_side):
    """Full validation of a solution."""
    # Check finite values
    if not np.all(np.isfinite(inner_hex_data)):
        return False
    if not np.all(np.isfinite(outer_center)) or not np.isfinite(outer_angle):
        return False
    if not np.isfinite(outer_side):
        return False
    if outer_side < np.sqrt(11) - 1e-9:
        return False
    
    # Check separation
    if not _check_all_separated(inner_hex_data):
        return False
    
    # Check containment
    if not _check_in_outer(inner_hex_data, outer_center, outer_angle, outer_side):
        return False
    
    return True

def _compute_min_outer_side(inner_hex_data, outer_center, outer_angle):
    """Compute minimum outer side length that contains all inner hexagons."""
    # For a given outer center and angle, find minimum side length
    # This is the maximum over all edges of the outer hexagon of the max projection
    
    best_side = 0
    for i in range(6):
        # Edge i of outer hexagon
        a1 = np.radians(outer_angle + 60 * i)
        a2 = np.radians(outer_angle + 60 * (i + 1))
        p1 = outer_center + best_side * np.array([np.cos(a1), np.sin(a1)])
        p2 = outer_center + best_side * np.array([np.cos(a2), np.sin(a2)])
        
        edge = p2 - p1
        if np.linalg.norm(edge) < 1e-10:
            continue
        
        n = np.array([edge[1], -edge[0]])
        n /= np.linalg.norm(n)
        mid = (p1 + p2) / 2
        if np.dot(n, mid - outer_center) < 0:
            n = -n
        
        # Max projection of all inner vertices on this normal
        max_proj = -np.inf
        for hex_idx in range(len(inner_hex_data)):
            x, y, ang = inner_hex_data[hex_idx]
            verts = _hex_vertices(x, y, 1.0, ang)
            proj = np.dot(verts, n)
            max_proj = max(max_proj, proj.max())
        
        # The outer vertex on this edge must be at least max_proj
        # For a regular hexagon with center c and side s, the distance from center to each edge is s * sqrt(3)/2
        # So we need: dot(outer_center, n) + s * sqrt(3)/2 >= max_proj
        center_proj = np.dot(outer_center, n)
        required_s = (max_proj - center_proj) / (np.sqrt(3) / 2)
        best_side = max(best_side, required_s)
    
    return best_side

def _objective(params):
    """Objective function: minimize outer side length."""
    if time.time() > _DEADLINE:
        return 1e10
    
    # Parse parameters
    inner_data = params[:33].reshape(11, 3)
    outer_center = params[33:35]
    outer_angle = params[35]
    
    # Compute minimum outer side
    try:
        side = _compute_min_outer_side(inner_data, outer_center, outer_angle)
    except:
        return 1e10
    
    if not np.isfinite(side) or side < np.sqrt(11) - 1e-6:
        return 1e10
    
    # Check separation
    if not _check_all_separated(inner_data):
        return 1e10
    
    # Check containment with this side
    if not _check_in_outer(inner_data, outer_center, outer_angle, side):
        return 1e10
    
    return side

def hexagon_packing_11():
    """
    Constructs a packing of 11 disjoint unit regular hexagons inside a larger regular hexagon,
    maximizing 1/outer_hex_side_length.
    
    Returns:
        inner_hex_data: np.ndarray of shape (11,3), rows (x, y, angle_deg)
        outer_hex_data: np.ndarray of shape (3,), (x, y, angle_deg)
        outer_hex_side_length: float
    """
    # Initial known valid arrangement (seed program)
    inner_init = np.array([
        [0, 0, 0],
        [-2.5, 0, 0],
        [2.5, 0, 0],
        [-1.25, 2.165, 0],
        [1.25, 2.165, 0],
        [-1.25, -2.165, 0],
        [1.25, -2.165, 0],
        [-3.75, 2.165, 0],
        [3.75, 2.165, 0],
        [-3.75, -2.165, 0],
        [3.75, -2.165, 0],
    ])
    outer_center_init = np.array([0.0, 0.0])
    outer_angle_init = 0.0
    
    # Compute initial side
    best_params = np.concatenate([inner_init.flatten(), outer_center_init, [outer_angle_init]])
    best_side = _compute_min_outer_side(inner_init, outer_center_init, outer_angle_init)
    
    # Verify initial solution
    if not _validate_solution(inner_init, outer_center_init, outer_angle_init, best_side):
        # Try to fix by expanding
        best_side = 10.0
        if not _check_in_outer(inner_init, outer_center_init, outer_angle_init, best_side):
            # Use a very large side as fallback
            best_side = 20.0
    
    # Differential evolution for global optimization
    # Bounds for each parameter
    bounds = []
    for i in range(11):
        bounds.append((-5, 5))  # x
        bounds.append((-5, 5))  # y
        bounds.append((-180, 180))  # angle
    bounds.append((-5, 5))  # outer center x
    bounds.append((-5, 5))  # outer center y
    bounds.append((-180, 180))  # outer angle
    
    if time.time() < _DEADLINE - 120:
        try:
            de_result = differential_evolution(
                _objective,
                bounds,
                seed=42,
                maxiter=100,
                tol=1e-8,
                popsize=15,
                mutation=(0.3, 1.0),
                recombination=0.8,
                polish=False,
                init='latinhypercube'
            )
            if de_result.fun < best_side and np.isfinite(de_result.fun):
                best_params = de_result.x
                best_side = de_result.fun
        except Exception as e:
            pass
    
    # Local refinement with Nelder-Mead
    if time.time() < _DEADLINE - 60:
        try:
            nm_result = minimize(
                _objective,
                best_params,
                method='Nelder-Mead',
                options={'maxiter': 2000, 'xatol': 1e-8, 'fatol': 1e-10, 'adaptive': True}
            )
            if nm_result.fun < best_side and np.isfinite(nm_result.fun):
                best_params = nm_result.x
                best_side = nm_result.fun
        except Exception as e:
            pass
    
    # Extract best solution
    best_inner = best_params[:33].reshape(11, 3)
    best_outer_center = best_params[33:35]
    best_outer_angle = best_params[35]
    
    # Recompute side for final solution
    final_side = _compute_min_outer_side(best_inner, best_outer_center, best_outer_angle)
    
    # Validate final solution
    if not _validate_solution(best_inner, best_outer_center, best_outer_angle, final_side):
        # Fallback to initial valid solution
        best_inner = inner_init
        best_outer_center = outer_center_init
        best_outer_angle = outer_angle_init
        final_side = _compute_min_outer_side(inner_init, outer_center_init, outer_angle_init)
    
    outer_hex_data = np.array([best_outer_center[0], best_outer_center[1], best_outer_angle])
    
    return best_inner, outer_hex_data, float(final_side)

# EVOLVE-BLOCK-END
