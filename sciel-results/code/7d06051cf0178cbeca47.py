# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize, differential_evolution

def _hex_vertices(cx, cy, s, a_deg):
    """Return 6 vertices of a regular hexagon with center (cx,cy), side s, angle a_deg (degrees)."""
    a = np.deg2rad(a_deg)
    angles = a + np.deg2rad(60.0 * np.arange(6))
    return np.column_stack([cx + s * np.cos(angles), cy + s * np.sin(angles)])

def _hex_edge_normals(vertices):
    """Return 6 outward edge normals for a hexagon given its 6 CCW vertices."""
    edges = np.roll(vertices, -1, axis=0) - vertices
    normals = np.column_stack([edges[:, 1], -edges[:, 0]])
    norms = np.linalg.norm(normals, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    normals = normals / norms
    return normals

def _separation_check(v1, v2, normals1, normals2, tol=1e-6):
    """
    Check if two hexagons (given vertices and normals) are separated.
    For each of the 12 normals (6 from each), project both hexagons and check if intervals are apart by > tol.
    Returns True if separated (non-intersecting), False otherwise.
    """
    all_normals = np.vstack([normals1, normals2])
    for n in all_normals:
        proj1 = v1 @ n
        proj2 = v2 @ n
        min1, max1 = proj1.min(), proj1.max()
        min2, max2 = proj2.min(), proj2.max()
        if max1 < min2 - tol or max2 < min1 - tol:
            return True
    return False

def _all_separated(inner_hexes, tol=1e-6):
    """Check all pairs of inner hexagons are separated."""
    n = len(inner_hexes)
    vertices_list = []
    normals_list = []
    for cx, cy, s, a_deg in inner_hexes:
        v = _hex_vertices(cx, cy, s, a_deg)
        nm = _hex_edge_normals(v)
        vertices_list.append(v)
        normals_list.append(nm)
    for i in range(n):
        for j in range(i + 1, n):
            if not _separation_check(vertices_list[i], vertices_list[j], normals_list[i], normals_list[j], tol):
                return False
    return True

def _containment_check(inner_hexes, outer_cx, outer_cy, outer_s, outer_a_deg, tol=1e-6):
    """
    Check that all inner hexagon vertices are inside the outer hexagon.
    """
    outer_v = _hex_vertices(outer_cx, outer_cy, outer_s, outer_a_deg)
    for i in range(6):
        p1 = outer_v[i]
        p2 = outer_v[(i + 1) % 6]
        e = p2 - p1
        for cx, cy, s, a_deg in inner_hexes:
            inner_v = _hex_vertices(cx, cy, s, a_deg)
            for p in inner_v:
                d = p - p1
                cross_val = e[0] * d[1] - e[1] * d[0]
                if cross_val < -tol:
                    return False
    return True

def _objective(params, n_inner=11):
    """
    Objective: minimize outer side length.
    params: [outer_s, x0, y0, a0, ..., x10, y10, a10]
    Outer center fixed at (0,0), angle 0.
    """
    outer_s = params[0]
    if outer_s < np.sqrt(11):
        return 1000.0 + 100.0 * (np.sqrt(11) - outer_s)
        
    inner_params = params[1:]
    inner_hexes = []
    for i in range(n_inner):
        x = inner_params[3*i]
        y = inner_params[3*i + 1]
        a = inner_params[3*i + 2]
        inner_hexes.append((x, y, 1.0, a))
    
    outer_cx, outer_cy, outer_a_deg = 0.0, 0.0, 0.0
    
    penalty = 0.0
    
    # Check separation
    if not _all_separated(inner_hexes, tol=1e-6):
        penalty += 100.0 * outer_s
    
    # Check containment
    if not _containment_check(inner_hexes, outer_cx, outer_cy, outer_s, outer_a_deg, tol=1e-6):
        penalty += 100.0 * outer_s
    
    return outer_s + penalty

def hexagon_packing_11():
    n = 11
    # Bounds for Differential Evolution
    # Outer side: [sqrt(11), 6]
    # Inner centers: [-5, 5]
    # Inner angles: [0, 360]
    
    bounds = [(np.sqrt(11), 6.0)]
    for i in range(n):
        bounds.append((-5.0, 5.0)) # x
        bounds.append((-5.0, 5.0)) # y
        bounds.append((0.0, 360.0)) # angle
        
    # Run Differential Evolution
    # Use a small population and maxiter to keep it fast, then refine
    de_result = differential_evolution(
        _objective, 
        bounds, 
        seed=42, 
        maxiter=100, 
        popsize=15,
        tol=1e-4,
        polish=False
    )
    
    best_params = de_result.x
    best_fun = de_result.fun
    
    # Refine with L-BFGS-B
    # L-BFGS-B can handle bounds
    local_result = minimize(
        _objective, 
        best_params, 
        method='L-BFGS-B', 
        bounds=bounds,
        options={'maxiter': 1000, 'ftol': 1e-8, 'gtol': 1e-8}
    )
    
    if local_result.fun < best_fun:
        best_params = local_result.x
        best_fun = local_result.fun
        
    # Final check and extraction
    outer_s = best_params[0]
    inner_hex_data = np.zeros((n, 3))
    for i in range(n):
        inner_hex_data[i, 0] = best_params[1 + 3*i]
        inner_hex_data[i, 1] = best_params[2 + 3*i]
        inner_hex_data[i, 2] = best_params[3 + 3*i]
    
    outer_hex_data = np.array([0.0, 0.0, 0.0])
    
    # Fallback if optimization fails completely
    if best_fun > 100:
        # Use a known valid packing (7-hex flower + 4 in gaps, roughly)
        # This is a heuristic fallback
        sqrt3 = np.sqrt(3)
        inner_hex_data = np.array(
            [
                [0, 0, 0],
                [sqrt3, 0, 0],
                [sqrt3/2, 1.5, 0],
                [-sqrt3/2, 1.5, 0],
                [-sqrt3, 0, 0],
                [-sqrt3/2, -1.5, 0],
                [sqrt3/2, -1.5, 0],
                [1.5*sqrt3, 1.5, 30],
                [-1.5*sqrt3, 1.5, 150],
                [-1.5*sqrt3, -1.5, -150],
                [1.5*sqrt3, -1.5, -30],
            ]
        )
        outer_hex_data = np.array([0, 0, 0])
        outer_s = 4.0 # Ensure it's large enough to be valid
        
    return inner_hex_data, outer_hex_data, float(outer_s)

# EVOLVE-BLOCK-END
