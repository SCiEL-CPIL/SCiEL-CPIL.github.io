# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize


def _hex_vertices(cx, cy, d, angle_deg):
    """Compute vertices of a regular hexagon with center (cx,cy), side d, angle angle_deg."""
    a = np.radians(angle_deg)
    angles = a + np.radians(60) * np.arange(6)
    return np.column_stack([cx + d * np.cos(angles), cy + d * np.sin(angles)])


def _hex_edge_normals(vertices):
    """Compute outward edge normals for a convex polygon given in CCW order."""
    n = len(vertices)
    normals = np.zeros((n, 2))
    for i in range(n):
        p1 = vertices[i]
        p2 = vertices[(i + 1) % n]
        edge = p2 - p1
        # Outward normal for CCW polygon: rotate edge by -90 degrees
        normals[i] = np.array([edge[1], -edge[0]])
        normals[i] /= (np.linalg.norm(normals[i]) + 1e-15)
    return normals


def _separation_distance(v1, n1, v2, n2):
    """
    Check if two convex polygons are separated using SAT.
    Returns the minimum separation (positive means separated, negative means overlapping).
    v1, n1: vertices and normals of polygon 1
    v2, n2: vertices and normals of polygon 2
    """
    min_sep = float('inf')
    # Test all normals from both polygons
    for normals in [n1, n2]:
        for normal in normals:
            proj1 = np.dot(v1, normal)
            proj2 = np.dot(v2, normal)
            min1, max1 = np.min(proj1), np.max(proj1)
            min2, max2 = np.min(proj2), np.max(proj2)
            # Separation along this axis
            sep = max(min2 - max1, min1 - max2)
            min_sep = min(min_sep, sep)
    return min_sep


def _point_in_hexagon(v, outer_vertices):
    """
    Check if point v is inside or on the boundary of the outer hexagon.
    outer_vertices: (6, 2) array of vertices in order.
    Returns True if v is inside or on boundary (with tolerance).
    """
    for i in range(6):
        p1 = outer_vertices[i]
        p2 = outer_vertices[(i + 1) % 6]
        edge = p2 - p1
        to_v = v - p1
        cross = edge[0] * to_v[1] - edge[1] * to_v[0]
        if cross < -1e-6:
            return False
    return True


def _objective(params, s):
    """
    Objective function for optimization.
    params: (12*2, ) array of [x1,y1, x2,y2, ..., x12,y12]
    s: outer hexagon side length
    """
    n = 12
    centers = params.reshape(n, 2)
    
    # Outer hexagon centered at origin with side length s
    outer_vertices = _hex_vertices(0, 0, s, 0)
    outer_normals = _hex_edge_normals(outer_vertices)
    
    penalty = 0.0
    
    # Check inner-inner separation
    for i in range(n):
        vi = _hex_vertices(centers[i, 0], centers[i, 1], 1.0, 0)
        ni = _hex_edge_normals(vi)
        for j in range(i + 1, n):
            vj = _hex_vertices(centers[j, 0], centers[j, 1], 1.0, 0)
            nj = _hex_edge_normals(vj)
            sep = _separation_distance(vi, ni, vj, nj)
            if sep < 1e-6:
                # Overlapping or too close: penalize
                penalty += max(0, 1e-6 - sep) ** 2
    
    # Check inner vertices inside outer hexagon
    for i in range(n):
        vi = _hex_vertices(centers[i, 0], centers[i, 1], 1.0, 0)
        for v in vi:
            if not _point_in_hexagon(v, outer_vertices):
                # Compute violation distance
                # Find the edge with maximum violation
                max_violation = 0
                for k in range(6):
                    p1 = outer_vertices[k]
                    p2 = outer_vertices[(k + 1) % 6]
                    edge = p2 - p1
                    to_v = v - p1
                    cross = edge[0] * to_v[1] - edge[1] * to_v[0]
                    if cross < 0:
                        # Distance from point to line
                        edge_len = np.linalg.norm(edge)
                        dist = -cross / edge_len
                        max_violation = max(max_violation, dist)
                penalty += max_violation ** 2
    
    return penalty


def _objective_and_grad(params, s):
    """
    Objective function with gradient for L-BFGS-B.
    """
    n = 12
    centers = params.reshape(n, 2)
    
    # Outer hexagon centered at origin with side length s
    outer_vertices = _hex_vertices(0, 0, s, 0)
    
    penalty = 0.0
    grad = np.zeros_like(params)
    
    # For simplicity, use finite differences for gradient if analytical is too complex
    # But let's try to compute it properly
    
    # Actually, let's just use the penalty and let L-BFGS-B handle gradients via finite differences
    # or we can provide an approximate gradient
    
    return penalty


def hexagon_packing_12():
    """
    Constructs a packing of 12 disjoint unit regular hexagons inside a larger regular hexagon,
    minimizing the outer hexagon side length.
    Returns
        inner_hex_data: np.ndarray of shape (12,3), where each row is of the form (x, y, angle_degrees)
        outer_hex_data: np.ndarray of shape (3,) of form (x,y,angle_degree)
        outer_hex_side_length: float representing the side length of the outer hexagon.
    """
    n = 12
    
    # Start with a reasonable initial guess
    # Use a hexagonal lattice arrangement as initial guess
    sqrt3 = np.sqrt(3)
    initial_centers = np.array([
        [0, 0],
        [-2.5, 0],
        [2.5, 0],
        [-1.25, 2.165],
        [1.25, 2.165],
        [-1.25, -2.165],
        [1.25, -2.165],
        [-3.75, 2.165],
        [3.75, 2.165],
        [-3.75, -2.165],
        [3.75, -2.165],
        [0, -4.33],
    ])
    
    # We'll try different outer hexagon sizes and optimize
    best_s = float('inf')
    best_centers = None
    
    # Try a range of s values
    for s in np.linspace(4.0, 7.0, 30):
        # Initial guess: scale the initial centers to fit
        # The bounding radius of the initial arrangement
        max_r = np.max(np.linalg.norm(initial_centers, axis=1))
        # The inradius of the outer hexagon is s * sqrt(3) / 2
        inradius = s * sqrt3 / 2
        # Scale factor to fit inside
        scale = min(1.0, (inradius - 1.0) / max_r) if max_r > 0 else 1.0
        x0 = (initial_centers * scale).flatten()
        
        # Define objective
        def obj(params):
            return _objective(params, s)
        
        # Optimize
        try:
            result = minimize(
                obj, x0, method='L-BFGS-B',
                options={'maxiter': 500, 'ftol': 1e-10, 'gtol': 1e-8}
            )
            
            if result.fun < 1e-6:  # Valid packing
                if s < best_s:
                    best_s = s
                    best_centers = result.x.reshape(n, 2)
        except Exception:
            pass
    
    # If we found a valid packing, use it
    if best_centers is not None and best_s < float('inf'):
        inner_hex_data = np.column_stack([
            best_centers[:, 0], best_centers[:, 1], np.zeros(n)
        ])
        outer_hex_data = np.array([0.0, 0.0, 0.0])
        return inner_hex_data, outer_hex_data, best_s
    else:
        # Fallback to the original solution
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
        outer_hex_side_length = 8
        return inner_hex_data, outer_hex_data, outer_hex_side_length


# EVOLVE-BLOCK-END
